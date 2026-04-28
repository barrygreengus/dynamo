"""Minimal mirror of Dynamo's multimodal HTTP client (httpx).

Singleton httpx.AsyncClient with Dynamo main's defaults:
  - Limits(max_connections=100, max_keepalive_connections=20)
  - timeout=30.0 (float -> httpx applies to connect/read/write/pool ALL)

Saturating the pool raises httpx.PoolTimeout after 30s. The 20-slot
keepalive cap with 100-slot pool means up to 80 connections churn
(re-handshake TLS) per cycle under load — amplifies the saturation.
"""

from __future__ import annotations

import asyncio

import httpx

# These are the default behavior of Dynamo
MAX_CONNECTIONS = 100
MAX_KEEPALIVE = 20
TIMEOUT = 30.0  # float -> applies to pool, connect, read, write

_client: httpx.AsyncClient | None = None
_lock = asyncio.Lock()
_first_response_logged = False


async def get_client() -> httpx.AsyncClient:
    global _client
    async with _lock:
        if _client is None or _client.is_closed:
            print(
                f"[dynamo] build AsyncClient: "
                f"max_connections={MAX_CONNECTIONS} "
                f"max_keepalive={MAX_KEEPALIVE} "
                f"timeout={TIMEOUT}"
            )
            _client = httpx.AsyncClient(
                timeout=TIMEOUT,
                limits=httpx.Limits(
                    max_connections=MAX_CONNECTIONS,
                    max_keepalive_connections=MAX_KEEPALIVE,
                ),
            )
    return _client


async def reset() -> None:
    global _client, _first_response_logged
    async with _lock:
        if _client is not None and not _client.is_closed:
            await _client.aclose()
        _client = None
        _first_response_logged = False


async def fetch(url: str, timeout: float | None = None) -> bytes:
    global _first_response_logged
    client = await get_client()
    r = await client.get(url, timeout=timeout) if timeout is not None else await client.get(url)
    if not _first_response_logged:
        _first_response_logged = True
        print(
            f"[dynamo] first response headers: "
            f"http={r.http_version} "
            f"connection={r.headers.get('connection')!r} "
            f"keep-alive={r.headers.get('keep-alive')!r} "
            f"server={r.headers.get('server')!r}"
        )
    r.raise_for_status()
    return r.content
