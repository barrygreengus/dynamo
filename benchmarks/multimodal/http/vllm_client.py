"""Minimal mirror of native vLLM's async HTTP fetch path (aiohttp).

Singleton aiohttp.ClientSession with vLLM's defaults:
  - trust_env=True
  - default TCPConnector (limit=100, limit_per_host=0)
  - per-request ClientTimeout(total=5.0)

Saturating raises asyncio.TimeoutError after 5s total. vLLM wraps this
in a 3-attempt, 4x backoff retry loop upstream — omitted here for
minimality. aiohttp has NO separate keepalive cap: up to `limit` idle
connections stay warm, so no re-handshake churn.
"""

from __future__ import annotations

import asyncio

import aiohttp

TIMEOUT = 5.0  # VLLM_IMAGE_FETCH_TIMEOUT default

_session: aiohttp.ClientSession | None = None
_lock = asyncio.Lock()


async def get_session() -> aiohttp.ClientSession:
    global _session
    async with _lock:
        if _session is None or _session.closed:
            _session = aiohttp.ClientSession(trust_env=True)
    return _session


async def reset() -> None:
    global _session
    async with _lock:
        if _session is not None and not _session.closed:
            await _session.close()
        _session = None


async def fetch(url: str, timeout: float | None = None) -> bytes:
    session = await get_session()
    t = aiohttp.ClientTimeout(total=timeout if timeout is not None else TIMEOUT)
    async with session.get(url, timeout=t) as r:
        r.raise_for_status()
        return await r.read()
