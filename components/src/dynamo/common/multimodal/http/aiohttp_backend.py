# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""aiohttp backend for the multimodal HTTP facade.

Singleton ``aiohttp.ClientSession`` mirroring vLLM's
``HTTPConnection.get_async_client`` pattern, with an explicit ``TCPConnector``
so single-origin fan-out has all 100 pool slots available.

Defaults:
  - ``limit=100`` — total pool (matches old httpx cap).
  - ``limit_per_host=0`` — unlimited per host, load-bearing for single-origin.
  - ``keepalive_timeout=15`` — ride through request bursts.
  - ``enable_cleanup_closed=True`` — purge half-closed TLS sockets.
  - ``trust_env=False`` — matches today's behavior; flip in a follow-up PR
    after a proxy-impact review.

Operators can override at session-creation time via env vars:

  - ``DYN_MM_HTTP_MAX_CONNECTIONS`` (default 100) — connector ``limit``.
  - ``DYN_MM_HTTP_LIMIT_PER_HOST`` (default 0).
  - ``DYN_MM_HTTP_KEEPALIVE_TIMEOUT`` (default 15s).
  - ``DYN_MM_HTTP_READ_TIMEOUT`` — when set, replaces the caller-supplied
    per-request timeout; otherwise the caller's ``timeout`` arg is used.

Per-request timeout is a ``ClientTimeout(total=timeout)`` — covers pool
wait + connect + read + body in one budget, same shape as vLLM.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp
from yarl import URL

from . import (
    MmHttpConnectionError,
    MmHttpStatusError,
    MmHttpTimeout,
    _env_float,
    _env_int,
)

logger = logging.getLogger(__name__)


class _Config:
    __slots__ = (
        "limit",
        "limit_per_host",
        "keepalive_timeout",
        "read_timeout_override",
    )

    def __init__(self) -> None:
        self.limit = _env_int("DYN_MM_HTTP_MAX_CONNECTIONS", 100)
        self.limit_per_host = _env_int("DYN_MM_HTTP_LIMIT_PER_HOST", 0)
        self.keepalive_timeout = _env_float("DYN_MM_HTTP_KEEPALIVE_TIMEOUT", 15.0)
        # ``read_timeout_override`` is None unless the env var is set; when None
        # the caller's per-call ``timeout`` arg is used as the total budget.
        raw_read = _env_float("DYN_MM_HTTP_READ_TIMEOUT", -1.0)
        self.read_timeout_override = raw_read if raw_read >= 0 else None


_config: Optional[_Config] = None
_session: Optional[aiohttp.ClientSession] = None
_session_lock = asyncio.Lock()


def _get_config() -> _Config:
    global _config
    if _config is None:
        _config = _Config()
    return _config


def _effective_timeout(timeout: float) -> aiohttp.ClientTimeout:
    cfg = _get_config()
    total = (
        cfg.read_timeout_override
        if cfg.read_timeout_override is not None
        else timeout
    )
    return aiohttp.ClientTimeout(total=total)


def _build_session() -> aiohttp.ClientSession:
    cfg = _get_config()
    connector = aiohttp.TCPConnector(
        limit=cfg.limit,
        limit_per_host=cfg.limit_per_host,
        keepalive_timeout=cfg.keepalive_timeout,
        enable_cleanup_closed=True,
    )
    return aiohttp.ClientSession(connector=connector, trust_env=False)


async def _get_session() -> aiohttp.ClientSession:
    global _session
    async with _session_lock:
        if _session is None or _session.closed:
            _session = _build_session()
            cfg = _get_config()
            logger.info(
                "aiohttp backend initialized: limit=%d, limit_per_host=%d, "
                "keepalive_timeout=%.1fs%s",
                cfg.limit,
                cfg.limit_per_host,
                cfg.keepalive_timeout,
                f", read timeout forced to {cfg.read_timeout_override:.1f}s via env"
                if cfg.read_timeout_override is not None
                else "; total timeout set per-request",
            )
    return _session


async def fetch_bytes(url: str, timeout: float) -> bytes:
    session = await _get_session()
    client_timeout = _effective_timeout(timeout)
    try:
        async with session.get(
            url, timeout=client_timeout, allow_redirects=True
        ) as response:
            response.raise_for_status()
            return await response.read()
    except aiohttp.ClientResponseError as e:
        raise MmHttpStatusError(e.status, e.message or "", url) from e
    except (asyncio.TimeoutError, aiohttp.ServerTimeoutError) as e:
        raise MmHttpTimeout(f"Timeout loading {url}") from e
    except (
        aiohttp.ClientConnectionError,
        aiohttp.ClientConnectorError,
        aiohttp.ServerDisconnectedError,
    ) as e:
        raise MmHttpConnectionError(f"Connection error loading {url}: {e}") from e
    except aiohttp.ClientError as e:
        raise MmHttpConnectionError(f"HTTP error loading {url}: {e}") from e


_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


async def fetch_body_or_redirect(
    url: str, timeout: float
) -> tuple[bytes | None, str | None]:
    """Single hop with ``allow_redirects=False``.

    Used by the facade's policy-aware path. Returns ``(body, None)`` for a
    terminal response (2xx, or 3xx without a ``Location`` header), or
    ``(None, absolute_next_url)`` for a followable redirect. Raises
    :class:`MmHttpStatusError` for 4xx/5xx and the usual timeout /
    connection classes on transport failure.
    """
    session = await _get_session()
    client_timeout = _effective_timeout(timeout)
    try:
        async with session.get(
            url, timeout=client_timeout, allow_redirects=False
        ) as response:
            if response.status in _REDIRECT_STATUSES:
                location = response.headers.get("Location")
                if location:
                    next_url = str(response.url.join(URL(location)))
                    return None, next_url
                return await response.read(), None

            try:
                response.raise_for_status()
            except aiohttp.ClientResponseError as e:
                raise MmHttpStatusError(e.status, e.message or "", url) from e
            return await response.read(), None
    except (asyncio.TimeoutError, aiohttp.ServerTimeoutError) as e:
        raise MmHttpTimeout(f"Timeout loading {url}") from e
    except (
        aiohttp.ClientConnectionError,
        aiohttp.ClientConnectorError,
        aiohttp.ServerDisconnectedError,
    ) as e:
        raise MmHttpConnectionError(f"Connection error loading {url}: {e}") from e
    except aiohttp.ClientError as e:
        raise MmHttpConnectionError(f"HTTP error loading {url}: {e}") from e


async def close() -> None:
    global _session, _config
    async with _session_lock:
        if _session is not None and not _session.closed:
            await _session.close()
        _session = None
        _config = None
