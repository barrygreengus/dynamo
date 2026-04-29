# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``aiohttp_backend`` exception mapping + redirect parsing.

Module-level singleton (``_session``) is replaced via ``patch.object``;
the autouse ``_close_shared_http_client`` fixture in
``conftest.py`` resets backend state between tests.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import aiohttp
import pytest
from yarl import URL

from dynamo.common.multimodal import http as mm_http
from dynamo.common.multimodal.http import aiohttp_backend

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _FakeResponse:
    """Minimal aiohttp response stand-in for ``async with session.get(...) as r``."""

    def __init__(self, *, status=200, headers=None, url=None, body=b"") -> None:
        self.status = status
        self.headers = headers or {}
        self.url = url
        self._body = body

    def raise_for_status(self) -> None:
        return None

    async def read(self) -> bytes:
        return self._body


def _cm_returning(response):
    """``session.get`` stand-in: async-CM whose ``__aenter__`` yields ``response``."""

    class _CM:
        async def __aenter__(self):
            return response

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def _get(url, **kwargs):
        return _CM()

    return _get


def _cm_raising(exc_factory):
    """``session.get`` stand-in: async-CM whose ``__aenter__`` raises."""

    class _CM:
        async def __aenter__(self):
            raise exc_factory()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def _get(url, **kwargs):
        return _CM()

    return _get


async def test_fetch_bytes_returns_body_on_200() -> None:
    response = _FakeResponse(status=200, body=b"hello")
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False
    session.get = _cm_returning(response)
    with patch.object(aiohttp_backend, "_session", session):
        result = await aiohttp_backend.fetch_bytes("https://h/x", 30.0)
    assert result == b"hello"


async def test_fetch_bytes_maps_timeout() -> None:
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False
    session.get = _cm_raising(lambda: asyncio.TimeoutError())
    with patch.object(aiohttp_backend, "_session", session):
        with pytest.raises(mm_http.MmHttpTimeout):
            await aiohttp_backend.fetch_bytes("https://h/x", 30.0)


async def test_fetch_bytes_maps_status() -> None:
    def _mk_error():
        return aiohttp.ClientResponseError(
            request_info=MagicMock(), history=(), status=404, message="Not Found"
        )

    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False
    session.get = _cm_raising(_mk_error)
    with patch.object(aiohttp_backend, "_session", session):
        with pytest.raises(mm_http.MmHttpStatusError) as exc:
            await aiohttp_backend.fetch_bytes("https://h/x", 30.0)
        assert exc.value.status == 404


async def test_fetch_bytes_maps_connection_error() -> None:
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False
    session.get = _cm_raising(lambda: aiohttp.ClientConnectionError("refused"))
    with patch.object(aiohttp_backend, "_session", session):
        with pytest.raises(mm_http.MmHttpConnectionError):
            await aiohttp_backend.fetch_bytes("https://h/x", 30.0)


async def test_fetch_body_or_redirect_returns_absolute_next_url_on_302() -> None:
    response = _FakeResponse(
        status=302,
        headers={"Location": "/next.png"},
        url=URL("https://h/x.png"),
    )
    session = MagicMock(spec=aiohttp.ClientSession)
    session.closed = False
    session.get = _cm_returning(response)
    with patch.object(aiohttp_backend, "_session", session):
        body, next_url = await aiohttp_backend.fetch_body_or_redirect(
            "https://h/x.png", 30.0
        )
    assert body is None
    assert next_url == "https://h/next.png"
