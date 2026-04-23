# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ImageLoader in-flight dedup, cancellation, and error contract."""

import asyncio
from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from dynamo.common.multimodal.http import MmHttpStatusError, MmHttpTimeout
from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.common.multimodal.url_validator import UrlValidationPolicy

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

_FETCH_BYTES_PATH = "dynamo.common.multimodal.image_loader.fetch_bytes"


def _make_png_bytes() -> bytes:
    """Create a minimal valid PNG in memory."""
    img = Image.new("RGB", (2, 2), color="red")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


PNG_BYTES = _make_png_bytes()


def _permissive_policy(
    allowed_local_path: str | None = None,
) -> UrlValidationPolicy:
    """Return a policy that permits the schemes used by tests without DNS hits."""
    return UrlValidationPolicy(
        allow_http=True,
        allow_private_ips=True,
        allowed_local_path=allowed_local_path,
    )


def _mock_fetch_bytes(
    content: bytes = PNG_BYTES,
    delay: float = 0.0,
    side_effect: Exception | None = None,
) -> AsyncMock:
    """Return an AsyncMock drop-in for ``fetch_bytes(url, timeout, policy=...)``.

    Args:
        content: Raw bytes returned as the fetch result.
        delay: Seconds to sleep before responding (simulates network latency).
        side_effect: If set, the mock raises this exception instead of returning.
    """

    async def _fetch(url, timeout, *, policy=None):
        if delay > 0:
            await asyncio.sleep(delay)
        if side_effect is not None:
            raise side_effect
        return content

    return AsyncMock(side_effect=_fetch)


@pytest.fixture(autouse=True)
def loader() -> ImageLoader:
    return ImageLoader(
        cache_size=4,
        http_timeout=30.0,
        url_policy=_permissive_policy(),
    )


# --- Concurrent same-URL dedup ---


async def test_concurrent_same_url_deduplicates(loader: ImageLoader) -> None:
    """Two concurrent load_image calls for the same URL should issue only one HTTP fetch."""
    mock_fetch = _mock_fetch_bytes(delay=0.05)
    with patch(_FETCH_BYTES_PATH, mock_fetch):
        results = await asyncio.gather(
            loader.load_image("https://example.com/img.png"),
            loader.load_image("https://example.com/img.png"),
        )

    assert len(results) == 2
    assert results[0].size == results[1].size
    assert mock_fetch.call_count == 1


async def test_concurrent_different_urls_fetch_independently(
    loader: ImageLoader,
) -> None:
    """Different URLs should each get their own fetch."""
    mock_fetch = _mock_fetch_bytes()
    with patch(_FETCH_BYTES_PATH, mock_fetch):
        await asyncio.gather(
            loader.load_image("https://example.com/a.png"),
            loader.load_image("https://example.com/b.png"),
        )

    assert mock_fetch.call_count == 2


# --- Waiter cancellation isolation ---


async def test_waiter_cancellation_does_not_cancel_shared_task(
    loader: ImageLoader,
) -> None:
    """Cancelling one waiter should not prevent the other from getting the image."""
    mock_fetch = _mock_fetch_bytes(delay=0.1)
    with patch(_FETCH_BYTES_PATH, mock_fetch):
        task_a = asyncio.create_task(loader.load_image("https://example.com/img.png"))
        task_b = asyncio.create_task(loader.load_image("https://example.com/img.png"))
        await asyncio.sleep(0.01)
        task_a.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task_a

        result_b = await task_b
        assert isinstance(result_b, Image.Image)


# --- Retry after failure ---


async def test_retry_after_failure(loader: ImageLoader) -> None:
    """After a fetch failure, the next caller should start a fresh fetch."""
    fail_fetch = _mock_fetch_bytes(side_effect=MmHttpTimeout("timeout"))
    ok_fetch = _mock_fetch_bytes()

    with patch(_FETCH_BYTES_PATH, fail_fetch):
        with pytest.raises(ValueError, match="Timeout"):
            await loader.load_image("https://example.com/img.png")

    # _inflight should be cleared after failure
    assert "https://example.com/img.png" not in loader._inflight

    with patch(_FETCH_BYTES_PATH, ok_fetch):
        result = await loader.load_image("https://example.com/img.png")
        assert isinstance(result, Image.Image)


# --- Error contract preserved for non-HTTP ---


async def test_file_url_is_rejected(loader: ImageLoader) -> None:
    """file:// inputs should be rejected before any local file read is attempted."""
    with pytest.raises(ValueError, match="Invalid image source scheme"):
        await loader.load_image("file:///nonexistent/path/img.png")


@pytest.mark.parametrize("url_factory", [lambda p: p.as_uri(), lambda p: str(p)])
async def test_local_file_inputs_are_rejected(
    loader: ImageLoader, tmp_path, url_factory
) -> None:
    """Local filesystem image inputs must be rejected for both file:// and bare paths."""
    image_path = tmp_path / "secret.png"
    Image.new("RGB", (1, 1), color="red").save(image_path, format="PNG")

    with pytest.raises(ValueError, match="Invalid image source scheme"):
        await loader.load_image(url_factory(image_path))


async def test_data_url_invalid_base64_normalized(loader: ImageLoader) -> None:
    """Malformed base64 data URL should raise ValueError."""
    with pytest.raises(ValueError, match="Invalid base64"):
        await loader.load_image("data:image/png;base64,NOT_VALID!!!")


async def test_data_url_non_image_rejected(loader: ImageLoader) -> None:
    """data: URL with non-image media type should raise ValueError."""
    with pytest.raises(ValueError, match="Data URL must be an image type"):
        await loader.load_image("data:text/plain;base64,aGVsbG8=")


# --- SSRF / scheme rejection ---


async def test_http_scheme_rejected_by_default(monkeypatch) -> None:
    """With default env policy, http:// URLs must be rejected before any fetch."""
    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL", raising=False)
    monkeypatch.delenv("DYN_MM_LOCAL_PATH", raising=False)

    default_loader = ImageLoader(cache_size=4, http_timeout=30.0)

    mock_fetch = _mock_fetch_bytes()
    with patch(_FETCH_BYTES_PATH, mock_fetch):
        with pytest.raises(ValueError, match="scheme|not allowed"):
            await default_loader.load_image("http://example.com/x.png")

    # Fetch must not be reached when the URL is rejected.
    assert mock_fetch.call_count == 0


async def test_blocked_private_ip_rejected(monkeypatch) -> None:
    """Cloud metadata / private IPs must be rejected even over https."""
    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL", raising=False)

    strict_loader = ImageLoader(
        cache_size=4,
        http_timeout=30.0,
        url_policy=UrlValidationPolicy(
            allow_http=True,
            allow_private_ips=False,
        ),
    )

    with pytest.raises(ValueError, match="blocked range"):
        await strict_loader.load_image("https://169.254.169.254/latest/meta-data/")


# --- HTTP error contract ---


async def test_http_timeout_raises_valueerror(loader: ImageLoader) -> None:
    """HTTP timeout should be normalized to ValueError."""
    mock_fetch = _mock_fetch_bytes(side_effect=MmHttpTimeout("timed out"))
    with patch(_FETCH_BYTES_PATH, mock_fetch):
        with pytest.raises(ValueError, match="Timeout loading image"):
            await loader.load_image("https://example.com/img.png")


async def test_http_status_error_propagated(loader: ImageLoader) -> None:
    """HTTP 4xx/5xx should propagate as MmHttpStatusError."""
    mock_fetch = _mock_fetch_bytes(
        side_effect=MmHttpStatusError(404, "Not Found", "https://example.com/img.png")
    )
    with patch(_FETCH_BYTES_PATH, mock_fetch):
        with pytest.raises(MmHttpStatusError) as exc_info:
            await loader.load_image("https://example.com/img.png")
        assert exc_info.value.status == 404


# --- Cache behavior ---


async def test_cache_hit_skips_fetch(loader: ImageLoader) -> None:
    """A cached image should be returned without making an HTTP request."""
    img = Image.new("RGB", (2, 2))
    loader._image_cache["https://example.com/img.png"] = img

    result = await loader.load_image("https://example.com/img.png")
    assert result is img


async def test_cache_is_lru_not_fifo(loader: ImageLoader) -> None:
    """Accessing a cached entry should protect it from eviction (LRU, not FIFO)."""
    loader._cache_size = 3
    mock_fetch = _mock_fetch_bytes()

    with patch(_FETCH_BYTES_PATH, mock_fetch):
        await loader.load_image("https://example.com/a.png")
        await loader.load_image("https://example.com/b.png")
        await loader.load_image("https://example.com/c.png")
        assert len(loader._image_cache) == 3

        # Touch "a" so it becomes most-recently-used
        await loader.load_image("https://example.com/a.png")

        # Insert "d" — should evict "b" (least recently used), not "a"
        await loader.load_image("https://example.com/d.png")

    assert "https://example.com/a.png" in loader._image_cache
    assert "https://example.com/b.png" not in loader._image_cache
    assert "https://example.com/c.png" in loader._image_cache
    assert "https://example.com/d.png" in loader._image_cache
