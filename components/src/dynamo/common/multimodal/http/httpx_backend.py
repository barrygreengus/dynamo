# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""httpx backend for the multimodal HTTP facade.

Behavior notes (operator-tunable knobs live in :mod:`.args`):

  - ``httpx.Timeout`` is built per-request so each fetch sees its
    caller-supplied read timeout instead of the first caller's value
    getting baked into the singleton. ``write`` is unbounded — fan-out
    GET workloads have no meaningful write phase.
  - ``max_keepalive_connections`` is sized to match ``max_connections``
    so idle connections get reused under fan-out instead of churning
    TLS handshakes.
  - A process-wide semaphore (``DYN_MM_HTTP_CONCURRENCY``) wraps every
    fetch so a burst can't push ``PoolTimeout`` up the stack. aiohttp
    has no equivalent because its connector queues natively.

An operator who flips ``DYN_MM_HTTP_BACKEND=httpx`` gets a working backend,
not the original PoolTimeout bug.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import httpx

from . import MmHttpConnectionError, MmHttpStatusError, MmHttpTimeout
from .args import HttpArgs, from_env

logger = logging.getLogger(__name__)


_args: Optional[HttpArgs] = None
_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()
_semaphore: Optional[asyncio.Semaphore] = None


def _get_args() -> HttpArgs:
    global _args
    if _args is None:
        _args = from_env()
    return _args


def _get_semaphore() -> asyncio.Semaphore:
    """Return the process-wide concurrency cap on in-flight httpx fetches.

    Lazy-init so the bound (``DYN_MM_HTTP_CONCURRENCY``) is read once and
    the ``asyncio.Semaphore`` binds to whichever loop first awaits it.
    """
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(_get_args().concurrency)
    return _semaphore


def _per_call_timeout(read_timeout: float) -> httpx.Timeout:
    args = _get_args()
    effective_read = (
        args.read_timeout_override
        if args.read_timeout_override is not None
        else read_timeout
    )
    return httpx.Timeout(
        connect=args.connect_timeout,
        read=effective_read,
        write=None,
        pool=args.pool_timeout,
    )


def _build_client() -> httpx.AsyncClient:
    args = _get_args()
    return httpx.AsyncClient(
        timeout=_per_call_timeout(60.0),
        follow_redirects=True,
        limits=httpx.Limits(
            max_connections=args.max_connections,
            max_keepalive_connections=args.max_keepalive,
        ),
    )


async def _get_client() -> httpx.AsyncClient:
    global _client
    async with _client_lock:
        if _client is None or _client.is_closed:
            _client = _build_client()
            args = _get_args()
            logger.info(
                "httpx backend initialized: max_connections=%d, max_keepalive=%d, "
                "timeout(connect=%.1fs, write=None, pool=%.1fs); read timeout %s",
                args.max_connections,
                args.max_keepalive,
                args.connect_timeout,
                args.pool_timeout,
                f"forced to {args.read_timeout_override:.1f}s via env"
                if args.read_timeout_override is not None
                else "set per-request",
            )
    return _client


async def fetch_bytes(url: str, timeout: float) -> bytes:
    client = await _get_client()
    async with _get_semaphore():
        try:
            response = await client.get(url, timeout=_per_call_timeout(timeout))
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as e:
            raise MmHttpStatusError(e.response.status_code, str(e), url) from e
        except httpx.TimeoutException as e:
            raise MmHttpTimeout(f"Timeout loading {url}") from e
        except (httpx.ConnectError, httpx.NetworkError) as e:
            raise MmHttpConnectionError(
                f"Connection error loading {url}: {e}"
            ) from e
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
    async with _get_semaphore():
        try:
            request = client.build_request("GET", url)
            response = await client.send(
                request, follow_redirects=False, timeout=_per_call_timeout(timeout)
            )
        except httpx.TimeoutException as e:
            raise MmHttpTimeout(f"Timeout loading {url}") from e
        except (httpx.ConnectError, httpx.NetworkError) as e:
            raise MmHttpConnectionError(
                f"Connection error loading {url}: {e}"
            ) from e
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
    global _client, _args, _semaphore
    async with _client_lock:
        if _client is not None and not _client.is_closed:
            await _client.aclose()
        _client = None
        _args = None
        _semaphore = None
