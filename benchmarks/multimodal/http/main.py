"""Pool-saturation benchmark: aiohttp vs httpx via dynamo.common.multimodal.http.

Fires --n concurrent GETs against four single-origin CDN seed URLs (each
cache-busted with a unique ?cb=<uuid>) through the project's
backend-neutral facade. Reports wall time + outcome class counts.

Backend is selected by DYN_MM_HTTP_BACKEND (aiohttp by default, or
httpx). Replace SEEDS below with image URLs reachable from your test
environment before running.

Usage:
  DYN_MM_HTTP_BACKEND=aiohttp python main.py --n 5000
  DYN_MM_HTTP_BACKEND=httpx   python main.py --n 5000
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid
from collections import Counter

from dynamo.common.multimodal.http import close_http_client, fetch_bytes

SEEDS = [
    "https://example.com/image-1.jpg",
    "https://example.com/image-2.jpg",
    "https://example.com/image-3.jpg",
    "https://example.com/image-4.jpg",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pool-saturation benchmark via dynamo.common.multimodal.http.",
    )
    p.add_argument("--n", type=int, default=5000,
                   help="number of concurrent fetches fired via asyncio.gather")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="per-request timeout in seconds (default: 30)")
    return p.parse_args()


def gen_urls(n: int) -> list[str]:
    return [f"{SEEDS[i % len(SEEDS)]}?cb={uuid.uuid4().hex}" for i in range(n)]


async def main() -> None:
    args = parse_args()
    backend = os.environ.get("DYN_MM_HTTP_BACKEND", "aiohttp").lower()
    print(f"[bench] backend={backend} n={args.n} timeout={args.timeout}s")

    urls = gen_urls(args.n)
    t0 = time.perf_counter()
    results = await asyncio.gather(
        *(fetch_bytes(u, timeout=args.timeout) for u in urls),
        return_exceptions=True,
    )
    wall = time.perf_counter() - t0
    await close_http_client()

    counter: Counter[str] = Counter()
    for r in results:
        if isinstance(r, BaseException):
            counter[type(r).__name__] += 1
        else:
            counter["success"] += 1

    total = len(results)
    print(f"[bench] backend={backend} n={total} wall={wall:.1f}s")
    for k, v in counter.most_common():
        print(f"  {k:<28} {v:>6}  {100.0 * v / total:5.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
