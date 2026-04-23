# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral facade for Dynamo's multimodal HTTP client.

Env var ``DYN_MM_HTTP_BACKEND`` selects the implementation used by
:func:`fetch_bytes`:

  - ``aiohttp`` (default) — ``aiohttp.ClientSession`` singleton.
  - ``httpx`` — ``httpx.AsyncClient`` singleton (fixed config: decoupled pool
    timeout, ``max_keepalive_connections`` matches ``max_connections``).

Callers use :func:`fetch_bytes` and catch the unified exception classes
(``MmHttpTimeout``, ``MmHttpConnectionError``, ``MmHttpStatusError``).

For SSRF-safe fetches (e.g. from ``ImageLoader``) pass a
``UrlValidationPolicy`` — :func:`fetch_bytes` then follows redirects
manually and revalidates each hop against the policy, matching the
behavior of the old httpx-only ``url_validator.fetch_with_revalidation``
but backend-neutrally.

:func:`get_http_client` is kept as a back-compat symbol; it always
returns an ``httpx.AsyncClient`` regardless of the selected backend
(unused in-tree after ImageLoader migrated to :func:`fetch_bytes`).
"""

from __future__ import annotations

import logging
import os
from types import ModuleType
from typing import Any

from ..url_validator import (
    UrlValidationError,
    UrlValidationPolicy,
    _MAX_REDIRECTS,
    validate_url,
)

logger = logging.getLogger(__name__)


class MmHttpError(Exception):
    """Base class for all multimodal HTTP fetch failures."""


class MmHttpTimeout(MmHttpError):
    """Timeout during connect / read / pool-wait."""


class MmHttpConnectionError(MmHttpError):
    """Network-layer failure: DNS, refused, reset, half-close."""


class MmHttpStatusError(MmHttpError):
    """Server responded with a non-2xx status."""

    def __init__(self, status: int, message: str, url: str) -> None:
        super().__init__(f"HTTP {status} for {url}: {message}")
        self.status = status
        self.message = message
        self.url = url


_VALID_BACKENDS = ("aiohttp", "httpx")
_impl: ModuleType | None = None


def _resolve_backend() -> ModuleType:
    """Pick a backend module based on ``DYN_MM_HTTP_BACKEND``.

    Resolved once per process on first call; subsequent calls return the
    cached module. Raises ``ValueError`` if the env var is set to an
    unsupported value.
    """
    global _impl
    if _impl is not None:
        return _impl

    name = os.environ.get("DYN_MM_HTTP_BACKEND", "aiohttp").lower()
    if name == "aiohttp":
        from . import aiohttp_backend as backend
    elif name == "httpx":
        from . import httpx_backend as backend
    else:
        raise ValueError(
            f"DYN_MM_HTTP_BACKEND={name!r} is invalid; must be one of {_VALID_BACKENDS}"
        )

    _impl = backend
    logger.info("Multimodal HTTP backend resolved: %s", name)
    return backend


async def fetch_bytes(
    url: str,
    timeout: float,
    *,
    policy: UrlValidationPolicy | None = None,
) -> bytes:
    """Fetch ``url`` via the configured backend and return the response body.

    Single-shot: no retries. Raises one of the unified exception classes
    above; callers never see native httpx/aiohttp classes.

    ``policy=None``: use the backend's built-in redirect handling.

    ``policy`` set: follow redirects manually and revalidate each hop
    against the policy via :func:`url_validator.validate_url`. This is
    the SSRF-safe path; raises :class:`UrlValidationError` if any hop
    fails or the chain exceeds ``_MAX_REDIRECTS``.
    """
    backend = _resolve_backend()
    if policy is None:
        return await backend.fetch_bytes(url, timeout)
    return await _fetch_with_revalidation(backend, url, timeout, policy)


async def _fetch_with_revalidation(
    backend: ModuleType,
    url: str,
    timeout: float,
    policy: UrlValidationPolicy,
) -> bytes:
    """Manual redirect loop with per-hop SSRF validation (backend-neutral)."""
    current = url
    hops_remaining = _MAX_REDIRECTS
    visited: list[str] = []
    while True:
        await validate_url(current, policy)
        visited.append(current)

        body, redirect_to = await backend.fetch_body_or_redirect(current, timeout)

        if redirect_to is None:
            # Backend must return bytes on terminal (2xx or 3xx-without-Location).
            assert body is not None
            return body

        if hops_remaining <= 0:
            raise UrlValidationError(
                f"Too many redirects (max={_MAX_REDIRECTS}); chain={visited}"
            )
        hops_remaining -= 1
        current = redirect_to


async def close_http_client() -> None:
    """Close the active backend's singleton. Idempotent. Safe across resets.

    Clears the resolved backend so a fresh env-var reading happens on the
    next call (primarily useful in tests that vary ``DYN_MM_HTTP_BACKEND``).
    """
    global _impl
    if _impl is None:
        return
    await _impl.close()
    _impl = None


def get_http_client(timeout: float = 60.0) -> Any:
    """Back-compat: return the shared ``httpx.AsyncClient``.

    Always returns an ``httpx.AsyncClient`` regardless of
    ``DYN_MM_HTTP_BACKEND``. Kept for out-of-tree consumers that imported
    this symbol before the facade switch; no in-tree code calls it after
    ImageLoader migrated to :func:`fetch_bytes`.
    """
    from . import httpx_backend
    return httpx_backend.get_raw_client(timeout)


__all__ = [
    "MmHttpError",
    "MmHttpTimeout",
    "MmHttpConnectionError",
    "MmHttpStatusError",
    "fetch_bytes",
    "close_http_client",
    "get_http_client",
]
