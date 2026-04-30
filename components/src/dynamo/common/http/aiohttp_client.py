# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""aiohttp implementation of :class:`MmHttpClient` — the default backend.

Why aiohttp is the default
--------------------------

Under high concurrency (e.g. one request fanning out to 100 image
URLs) the httpx backend hits ``httpx.PoolTimeout``. aiohttp scales
markedly better on the same workload — see
https://docs.nvidia.com/nemo/gym/latest/infrastructure/engineering-notes/aiohttp-vs-httpx.html
and https://github.com/openai/openai-python/issues/1596.

Root cause is in httpx's pool. The maintenance routine in
``httpcore._async.connection_pool`` (around L303-309) runs
"whenever a new request is added or removed from the pool" and is
``O(queue_size × pool_size)`` per call —
https://github.com/encode/httpcore/blob/master/httpcore/_async/connection_pool.py#L303-L309
— so cost grows quadratically with backlog. aiohttp's connector
queues natively in O(1), which is why our httpx backend needed a
process-wide semaphore (``DYN_MM_HTTP_CONCURRENCY``) in front of
the pool to keep ``PoolTimeout`` from leaking up the stack. That
semaphore is redundant under aiohttp.

Measured at 500 rps × 10k requests across server-processing-time
buckets (lower is better, all latencies ms)::

    [sweep] mean_ms=50  order=['aiohttp', 'httpx']
    backend  wall(s)     avg     p50     p90     p99
    httpx       20.1    75.7    51.0   158.7   241.1
    aiohttp     20.1    50.7    50.6    50.8    51.1

    [sweep] mean_ms=100  order=['httpx', 'aiohttp']
    httpx       23.2  1550.4  1484.3  2430.2  3164.9
    aiohttp     20.1   101.4   100.6   100.9   101.2

    [sweep] mean_ms=200  order=['aiohttp', 'httpx']
    httpx       43.2 11833.2 11762.6 21167.1 23025.2
    aiohttp     20.8   320.5   231.2   733.3   830.3

    [sweep] mean_ms=300  order=['httpx', 'aiohttp']
    httpx       62.2 21229.8 21318.0 37834.0 41728.4
    aiohttp     32.2  6355.0  6395.1 11119.6 12071.7

Both backends are equivalent below saturation (mean_ms=50). Past
saturation httpx degrades super-linearly while aiohttp stays close
to the offered rate.

Effective-defaults mapping (httpx → aiohttp)
--------------------------------------------

Operator-tunable knobs live in :mod:`.args`. The table below names
every behaviorally-relevant knob: each row is either a *match*, a
*semantic difference*, or an *intentional change*.

================================  =========================================  =========================================  ====================================================================
dimension                         httpx (old default)                        aiohttp (new default)                       parity
================================  =========================================  =========================================  ====================================================================
total pool size                   ``Limits(max_connections=100)``            ``TCPConnector(limit=100)``                 match
per-host cap                      none                                       ``limit_per_host=0`` (unlimited)            match — load-bearing for single-origin fan-out, not env-tunable
keepalive cap                     ``max_keepalive_connections=100`` (count)  ``keepalive_timeout=15s`` (time)            different (semantics) — aiohttp ages out idle conns by time, no count cap
per-request timeout shape         ``Timeout(connect=5s, read=per-call,
                                  write=None, pool=60s)``                    ``ClientTimeout(total=per-call)``           different (intentional) — one budget covering pool wait + connect + read + body, same shape as vLLM ``HTTPConnection``
follow redirects                  ``follow_redirects=True`` (client)         ``allow_redirects=True`` (per call)         match
concurrency cap                   process-wide ``Semaphore(50)``
                                  (``DYN_MM_HTTP_CONCURRENCY``)              none — connector queues in O(1)            different (intentional) — semaphore was a workaround for httpx's O(queue × pool) cost; redundant under aiohttp
proxy honored from env            ``trust_env=True`` (httpx default)         ``trust_env=True``                          match
half-closed TLS cleanup           handled internally                         ``enable_cleanup_closed=True``              aiohttp-only knob
================================  =========================================  =========================================  ====================================================================
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp
from yarl import URL

from .base import MmHttpClient, MmHttpConnectionError, MmHttpStatusError, MmHttpTimeout

logger = logging.getLogger(__name__)


_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


class AiohttpClient(MmHttpClient):
    """aiohttp-backed concrete client."""

    def __init__(self, args=None) -> None:
        super().__init__(args)
        self._session: Optional[aiohttp.ClientSession] = None

    def _effective_timeout(self, timeout: float) -> aiohttp.ClientTimeout:
        total = (
            self._args.read_timeout_override
            if self._args.read_timeout_override is not None
            else timeout
        )
        return aiohttp.ClientTimeout(total=total)

    def _build_session(self) -> aiohttp.ClientSession:
        connector = aiohttp.TCPConnector(
            limit=self._args.max_connections,
            # Single-origin fan-out is the whole point; capping per-host
            # would defeat it. Hard-coded rather than env-tunable.
            limit_per_host=0,
            keepalive_timeout=self._args.keepalive_timeout,
            enable_cleanup_closed=True,
        )
        return aiohttp.ClientSession(connector=connector, trust_env=True)

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._lock:
            if self._session is None or self._session.closed:
                self._session = self._build_session()
                logger.info(
                    "aiohttp backend initialized: limit=%d, limit_per_host=0, "
                    "keepalive_timeout=%.1fs%s",
                    self._args.max_connections,
                    self._args.keepalive_timeout,
                    f", read timeout forced to {self._args.read_timeout_override:.1f}s via env"
                    if self._args.read_timeout_override is not None
                    else "; total timeout set per-request",
                )
        return self._session

    async def _fetch_simple(self, url: str, timeout: float) -> bytes:
        session = await self._get_session()
        client_timeout = self._effective_timeout(timeout)
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

    async def fetch_body_or_redirect(
        self, url: str, timeout: float
    ) -> tuple[bytes | None, str | None]:
        session = await self._get_session()
        client_timeout = self._effective_timeout(timeout)
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

    async def close(self) -> None:
        async with self._lock:
            if self._session is not None and not self._session.closed:
                await self._session.close()
            self._session = None
