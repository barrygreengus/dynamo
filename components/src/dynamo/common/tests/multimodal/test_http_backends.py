# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-resolution and exception-mapping tests for the multimodal HTTP facade."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import aiohttp
import httpx
import pytest

from dynamo.common.multimodal import http as mm_http

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture(autouse=True)
def _reset_backend_cache():
    """Every test resolves the backend from scratch."""
    mm_http._impl = None
    yield
    mm_http._impl = None


# --- Backend selection ---


async def test_default_backend_is_aiohttp(monkeypatch) -> None:
    monkeypatch.delenv("DYN_MM_HTTP_BACKEND", raising=False)
    from dynamo.common.multimodal.http import aiohttp_backend
    assert mm_http._resolve_backend() is aiohttp_backend


async def test_httpx_backend_selected(monkeypatch) -> None:
    monkeypatch.setenv("DYN_MM_HTTP_BACKEND", "httpx")
    from dynamo.common.multimodal.http import httpx_backend
    assert mm_http._resolve_backend() is httpx_backend


async def test_invalid_backend_raises(monkeypatch) -> None:
    monkeypatch.setenv("DYN_MM_HTTP_BACKEND", "requests")
    with pytest.raises(ValueError, match="DYN_MM_HTTP_BACKEND"):
        mm_http._resolve_backend()


# --- get_http_client always returns httpx (url_validator contract) ---


async def test_get_http_client_returns_httpx_under_httpx_backend(monkeypatch) -> None:
    monkeypatch.setenv("DYN_MM_HTTP_BACKEND", "httpx")
    client = mm_http.get_http_client(timeout=30.0)
    assert isinstance(client, httpx.AsyncClient)


async def test_get_http_client_returns_httpx_under_aiohttp_backend(monkeypatch) -> None:
    """``get_http_client`` must always return httpx — url_validator requires it."""
    monkeypatch.setenv("DYN_MM_HTTP_BACKEND", "aiohttp")
    client = mm_http.get_http_client(timeout=30.0)
    assert isinstance(client, httpx.AsyncClient)


# --- httpx backend exception mapping ---


async def test_httpx_timeout_mapped(monkeypatch) -> None:
    monkeypatch.setenv("DYN_MM_HTTP_BACKEND", "httpx")
    from dynamo.common.multimodal.http import httpx_backend

    async def _raise_timeout(url, **kwargs):
        raise httpx.ConnectTimeout("timeout")

    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.get = MagicMock(side_effect=_raise_timeout)
    with patch.object(httpx_backend, "_client", mock_client):
        with pytest.raises(mm_http.MmHttpTimeout) as exc:
            await httpx_backend.fetch_bytes("https://x", 30.0)
        assert isinstance(exc.value.__cause__, httpx.ConnectTimeout)


async def test_httpx_status_mapped(monkeypatch) -> None:
    monkeypatch.setenv("DYN_MM_HTTP_BACKEND", "httpx")
    from dynamo.common.multimodal.http import httpx_backend

    response = MagicMock(spec=httpx.Response)
    response.status_code = 404
    response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404 Not Found", request=MagicMock(), response=response
    )

    async def _return_response(url, **kwargs):
        return response

    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.get = MagicMock(side_effect=_return_response)
    with patch.object(httpx_backend, "_client", mock_client):
        with pytest.raises(mm_http.MmHttpStatusError) as exc:
            await httpx_backend.fetch_bytes("https://x", 30.0)
        assert exc.value.status == 404


# --- aiohttp backend exception mapping ---
# aiohttp's session.get(...) returns an async context manager, so the mock
# must return an object with __aenter__ / __aexit__, not a coroutine that
# raises. We construct a fake CM whose __aenter__ raises the target exception.


def _cm_raising(exc_factory):
    """Return a session.get() stand-in: a factory that yields an async CM
    whose ``__aenter__`` raises what ``exc_factory()`` returns."""

    class _RaisingCM:
        async def __aenter__(self):
            raise exc_factory()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def _get(url, **kwargs):
        return _RaisingCM()

    return _get


async def test_aiohttp_timeout_mapped(monkeypatch) -> None:
    monkeypatch.setenv("DYN_MM_HTTP_BACKEND", "aiohttp")
    from dynamo.common.multimodal.http import aiohttp_backend

    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False
    session.get = _cm_raising(lambda: asyncio.TimeoutError())
    with patch.object(aiohttp_backend, "_session", session):
        with pytest.raises(mm_http.MmHttpTimeout):
            await aiohttp_backend.fetch_bytes("https://x", 30.0)


async def test_aiohttp_status_mapped(monkeypatch) -> None:
    monkeypatch.setenv("DYN_MM_HTTP_BACKEND", "aiohttp")
    from dynamo.common.multimodal.http import aiohttp_backend

    def _mk_error():
        return aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=404, message="Not Found"
        )

    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False
    session.get = _cm_raising(_mk_error)
    with patch.object(aiohttp_backend, "_session", session):
        with pytest.raises(mm_http.MmHttpStatusError) as exc:
            await aiohttp_backend.fetch_bytes("https://x", 30.0)
        assert exc.value.status == 404


# --- Backend-neutral SSRF revalidation via fetch_bytes(policy=...) ---
#
# These replace the deleted httpx-only tests in test_url_validator.py
# (test_fetch_with_revalidation_*). They exercise the facade path through
# an injected `fetch_body_or_redirect` stub, so they cover the shared
# redirect-loop + validate-each-hop logic independent of the chosen
# backend. We then parametrize over both backends to confirm routing.


from dynamo.common.multimodal.url_validator import (  # noqa: E402
    UrlValidationError,
    UrlValidationPolicy,
)

_PERMISSIVE = UrlValidationPolicy(allow_http=True, allow_private_ips=True)


def _backend_mod(name: str):
    from dynamo.common.multimodal.http import aiohttp_backend, httpx_backend
    return {"aiohttp": aiohttp_backend, "httpx": httpx_backend}[name]


@pytest.mark.parametrize("backend_name", ["aiohttp", "httpx"])
async def test_fetch_with_policy_returns_first_response(
    monkeypatch, backend_name
) -> None:
    monkeypatch.setenv("DYN_MM_HTTP_BACKEND", backend_name)
    backend = _backend_mod(backend_name)

    call_count = {"n": 0}

    async def _fake(url, timeout):
        call_count["n"] += 1
        return b"body-bytes", None

    with patch.object(backend, "fetch_body_or_redirect", _fake):
        result = await mm_http.fetch_bytes(
            "https://example.com/x.png", 30.0, policy=_PERMISSIVE
        )
    assert result == b"body-bytes"
    assert call_count["n"] == 1


@pytest.mark.parametrize("backend_name", ["aiohttp", "httpx"])
async def test_fetch_with_policy_follows_safe_redirect(
    monkeypatch, backend_name
) -> None:
    monkeypatch.setenv("DYN_MM_HTTP_BACKEND", backend_name)
    backend = _backend_mod(backend_name)

    hops: list[str] = []

    async def _fake(url, timeout):
        hops.append(url)
        if url == "https://example.com/x.png":
            return None, "https://example.com/final.png"
        return b"final-bytes", None

    with patch.object(backend, "fetch_body_or_redirect", _fake):
        result = await mm_http.fetch_bytes(
            "https://example.com/x.png", 30.0, policy=_PERMISSIVE
        )
    assert result == b"final-bytes"
    assert hops == ["https://example.com/x.png", "https://example.com/final.png"]


@pytest.mark.parametrize("backend_name", ["aiohttp", "httpx"])
async def test_fetch_with_policy_blocks_redirect_to_private_ip(
    monkeypatch, backend_name
) -> None:
    monkeypatch.setenv("DYN_MM_HTTP_BACKEND", backend_name)
    backend = _backend_mod(backend_name)

    strict = UrlValidationPolicy(allow_private_ips=False)

    async def _fake(url, timeout):
        return None, "http://169.254.169.254/latest/meta-data/"

    with patch.object(backend, "fetch_body_or_redirect", _fake):
        with pytest.raises(UrlValidationError):
            await mm_http.fetch_bytes("https://8.8.8.8/x.png", 30.0, policy=strict)


@pytest.mark.parametrize("backend_name", ["aiohttp", "httpx"])
async def test_fetch_with_policy_enforces_redirect_limit(
    monkeypatch, backend_name
) -> None:
    monkeypatch.setenv("DYN_MM_HTTP_BACKEND", backend_name)
    backend = _backend_mod(backend_name)

    # _MAX_REDIRECTS=3 → 4 hops trip the cap.
    chain = {
        "https://example.com/a": "https://example.com/b",
        "https://example.com/b": "https://example.com/c",
        "https://example.com/c": "https://example.com/d",
        "https://example.com/d": "https://example.com/e",
    }

    async def _fake(url, timeout):
        return None, chain[url]

    with patch.object(backend, "fetch_body_or_redirect", _fake):
        with pytest.raises(UrlValidationError, match="Too many redirects"):
            await mm_http.fetch_bytes(
                "https://example.com/a", 30.0, policy=_PERMISSIVE
            )
