# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""httpx backend for the multimodal HTTP facade.

Fixed config (not as-shipped):
  - ``httpx.Timeout(connect=5, read=<per-call>, write=10, pool=60)`` applied
    per-request, so each fetch sees its caller-supplied read timeout instead
    of the first caller's value getting baked into the singleton.
  - ``httpx.Limits(max_connections=100, max_keepalive_connections=100)`` —
    keepalive matches pool size so idle connections get reused under fan-out
    instead of churning TLS handshakes.

An operator who flips ``DYN_MM_HTTP_BACKEND=httpx`` gets a working backend,
not the original PoolTimeout bug.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from . import MmHttpConnectionError, MmHttpStatusError, MmHttpTimeout

logger = logging.getLogger(__name__)

_MAX_CONNECTIONS: int = 100
_CONNECT_TIMEOUT: float = 5.0
_WRITE_TIMEOUT: float = 10.0
_POOL_TIMEOUT: float = 60.0

_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()


def _per_call_timeout(read_timeout: float) -> httpx.Timeout:
    return httpx.Timeout(
        connect=_CONNECT_TIMEOUT,
        read=read_timeout,
        write=_WRITE_TIMEOUT,
        pool=_POOL_TIMEOUT,
    )


def _build_client() -> httpx.AsyncClient:
    # The singleton's default timeout is overridden on every request by
    # per-call ``httpx.Timeout(...)`` passed to ``client.get(...)``; we set a
    # conservative default here only for direct callers of ``get_raw_client``.
    return httpx.AsyncClient(
        timeout=_per_call_timeout(60.0),
        follow_redirects=True,
        limits=httpx.Limits(
            max_connections=_MAX_CONNECTIONS,
            max_keepalive_connections=_MAX_CONNECTIONS,
        ),
    )


async def _get_client() -> httpx.AsyncClient:
    global _client
    async with _client_lock:
        if _client is None or _client.is_closed:
            _client = _build_client()
            logger.info(
                "httpx backend initialized: max_connections=%d, max_keepalive=%d, "
                "timeout(connect=%.1fs, write=%.1fs, pool=%.1fs); read timeout "
                "set per-request",
                _MAX_CONNECTIONS,
                _MAX_CONNECTIONS,
                _CONNECT_TIMEOUT,
                _WRITE_TIMEOUT,
                _POOL_TIMEOUT,
            )
    return _client


async def fetch_bytes(url: str, timeout: float) -> bytes:
    client = await _get_client()
    try:
        response = await client.get(url, timeout=_per_call_timeout(timeout))
        response.raise_for_status()
        return response.content
    except httpx.HTTPStatusError as e:
        raise MmHttpStatusError(e.response.status_code, str(e), url) from e
    except httpx.TimeoutException as e:
        raise MmHttpTimeout(f"Timeout loading {url}") from e
    except (httpx.ConnectError, httpx.NetworkError) as e:
        raise MmHttpConnectionError(f"Connection error loading {url}: {e}") from e
    except httpx.HTTPError as e:
        raise MmHttpConnectionError(f"HTTP error loading {url}: {e}") from e


async def fetch_body_or_redirect(
    url: str, timeout: float
) -> tuple[bytes | None, str | None]:
    """Single hop with redirects disabled.

    Used by the facade's policy-aware path. Returns ``(body, None)`` for a
    terminal response (2xx, or 3xx without a ``Location`` header), or
    ``(None, absolute_next_url)`` for a followable redirect. Raises
    :class:`MmHttpStatusError` for 4xx/5xx and the usual timeout /
    connection classes on transport failure.
    """
    client = await _get_client()
    try:
        request = client.build_request("GET", url)
        response = await client.send(
            request, follow_redirects=False, timeout=_per_call_timeout(timeout)
        )
    except httpx.TimeoutException as e:
        raise MmHttpTimeout(f"Timeout loading {url}") from e
    except (httpx.ConnectError, httpx.NetworkError) as e:
        raise MmHttpConnectionError(f"Connection error loading {url}: {e}") from e
    except httpx.HTTPError as e:
        raise MmHttpConnectionError(f"HTTP error loading {url}: {e}") from e

    try:
        if response.is_redirect:
            location = response.headers.get("location")
            if location:
                next_url = str(response.url.join(location))
                return None, next_url
            # 3xx without Location: treat as terminal, surface the body.
            return response.content, None

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise MmHttpStatusError(e.response.status_code, str(e), url) from e
        return response.content, None
    finally:
        await response.aclose()


async def close() -> None:
    global _client
    async with _client_lock:
        if _client is not None and not _client.is_closed:
            await _client.aclose()
        _client = None


def get_raw_client(timeout: float) -> httpx.AsyncClient:
    """Return the underlying ``httpx.AsyncClient`` (creating it if needed).

    Synchronous accessor for back-compat. The ``timeout`` parameter is
    retained for ABI but unused — the session uses per-request timeouts and
    callers who want a specific budget should pass ``timeout=httpx.Timeout(...)``
    directly on each ``.get(...)``.
    """
    _ = timeout  # legacy ABI; actual timeouts are per-request
    global _client
    if _client is None or _client.is_closed:
        _client = _build_client()
    return _client
