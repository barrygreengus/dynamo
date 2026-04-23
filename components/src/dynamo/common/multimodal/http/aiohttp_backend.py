# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""aiohttp backend for the multimodal HTTP facade.

Singleton ``aiohttp.ClientSession`` mirroring vLLM's
``HTTPConnection.get_async_client`` pattern, with an explicit ``TCPConnector``
so single-origin fan-out (e.g. Pinterest CDN) has all 100 pool slots
available.

Config:
  - ``limit=100`` — total pool (matches old httpx cap).
  - ``limit_per_host=0`` — unlimited per host, load-bearing for single-origin.
  - ``keepalive_timeout=15`` — ride through request bursts.
  - ``enable_cleanup_closed=True`` — purge half-closed TLS sockets.
  - ``trust_env=False`` — matches today's behavior; flip in a follow-up PR
    after a proxy-impact review.

Per-request timeout is a ``ClientTimeout(total=timeout)`` — covers pool
wait + connect + read + body in one budget, same shape as vLLM.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, NoReturn, Optional

import aiohttp
from yarl import URL

from . import MmHttpConnectionError, MmHttpStatusError, MmHttpTimeout

logger = logging.getLogger(__name__)

_LIMIT: int = 100
_LIMIT_PER_HOST: int = 0
_KEEPALIVE_TIMEOUT: float = 15.0

_session: Optional[aiohttp.ClientSession] = None
_session_lock = asyncio.Lock()


def _build_session() -> aiohttp.ClientSession:
    connector = aiohttp.TCPConnector(
        limit=_LIMIT,
        limit_per_host=_LIMIT_PER_HOST,
        keepalive_timeout=_KEEPALIVE_TIMEOUT,
        enable_cleanup_closed=True,
    )
    return aiohttp.ClientSession(connector=connector, trust_env=False)


async def _get_session() -> aiohttp.ClientSession:
    global _session
    async with _session_lock:
        if _session is None or _session.closed:
            _session = _build_session()
            logger.info(
                "aiohttp backend initialized: limit=%d, limit_per_host=%d, "
                "keepalive_timeout=%.1fs",
                _LIMIT,
                _LIMIT_PER_HOST,
                _KEEPALIVE_TIMEOUT,
            )
    return _session


async def fetch_bytes(url: str, timeout: float) -> bytes:
    session = await _get_session()
    client_timeout = aiohttp.ClientTimeout(total=timeout)
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
    client_timeout = aiohttp.ClientTimeout(total=timeout)
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
    global _session
    async with _session_lock:
        if _session is not None and not _session.closed:
            await _session.close()
        _session = None


def get_raw_client(timeout: float) -> NoReturn:
    """aiohttp backend has no drop-in replacement for ``httpx.AsyncClient``.

    ``aiohttp.ClientSession.get(url)`` returns an async context manager;
    callers that expect ``await client.get(url)`` would break. Callers
    should migrate to :func:`..fetch_bytes`. Set
    ``DYN_MM_HTTP_BACKEND=httpx`` to opt back into the httpx call shape.
    """
    _ = timeout  # unused
    raise NotImplementedError(
        "The aiohttp backend does not expose a raw-client shim. "
        "Use dynamo.common.multimodal.http.fetch_bytes(url, timeout) instead, "
        "or set DYN_MM_HTTP_BACKEND=httpx to opt back into the httpx.AsyncClient shape."
    )
