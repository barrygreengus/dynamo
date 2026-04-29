# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``httpx_backend`` exception mapping + redirect parsing.

Module-level singleton (``_client``) is replaced via ``patch.object``;
the autouse ``_close_shared_http_client`` fixture in
``conftest.py`` resets backend state between tests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from dynamo.common.multimodal import http as mm_http
from dynamo.common.multimodal.http import httpx_backend

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _async_returning(value):
    """Coroutine factory used as ``side_effect`` for an awaited async call."""

    async def _coro(*args, **kwargs):
        return value

    return _coro


def _async_raising(exc):
    async def _coro(*args, **kwargs):
        raise exc

    return _coro


async def test_fetch_bytes_returns_body_on_200() -> None:
    response = MagicMock(spec=httpx.Response)
    response.content = b"hello"
    response.raise_for_status = MagicMock(return_value=None)
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.get = MagicMock(side_effect=_async_returning(response))
    with patch.object(httpx_backend, "_client", mock_client):
        result = await httpx_backend.fetch_bytes("https://h/x", 30.0)
    assert result == b"hello"


async def test_fetch_bytes_maps_timeout() -> None:
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.get = MagicMock(
        side_effect=_async_raising(httpx.ConnectTimeout("timeout"))
    )
    with patch.object(httpx_backend, "_client", mock_client):
        with pytest.raises(mm_http.MmHttpTimeout) as exc:
            await httpx_backend.fetch_bytes("https://h/x", 30.0)
        assert isinstance(exc.value.__cause__, httpx.ConnectTimeout)


async def test_fetch_bytes_maps_status() -> None:
    response = MagicMock(spec=httpx.Response)
    response.status_code = 404
    response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "404 Not Found", request=MagicMock(), response=response
        )
    )
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.get = MagicMock(side_effect=_async_returning(response))
    with patch.object(httpx_backend, "_client", mock_client):
        with pytest.raises(mm_http.MmHttpStatusError) as exc:
            await httpx_backend.fetch_bytes("https://h/x", 30.0)
        assert exc.value.status == 404


async def test_fetch_bytes_maps_connection_error() -> None:
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.get = MagicMock(
        side_effect=_async_raising(httpx.ConnectError("refused"))
    )
    with patch.object(httpx_backend, "_client", mock_client):
        with pytest.raises(mm_http.MmHttpConnectionError):
            await httpx_backend.fetch_bytes("https://h/x", 30.0)


async def test_fetch_body_or_redirect_returns_absolute_next_url_on_302() -> None:
    response = MagicMock(spec=httpx.Response)
    response.is_redirect = True
    response.headers = {"location": "/next.png"}
    response.url = httpx.URL("https://h/x.png")
    response.aclose = AsyncMock(return_value=None)
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_client.is_closed = False
    mock_client.build_request = MagicMock(return_value=MagicMock())
    mock_client.send = MagicMock(side_effect=_async_returning(response))
    with patch.object(httpx_backend, "_client", mock_client):
        body, next_url = await httpx_backend.fetch_body_or_redirect(
            "https://h/x.png", 30.0
        )
    assert body is None
    assert next_url == "https://h/next.png"
