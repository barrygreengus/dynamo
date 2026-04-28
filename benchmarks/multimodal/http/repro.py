"""Minimal pool-saturation reproducer: Dynamo (httpx) vs vLLM (aiohttp).

Fires --n concurrent GETs against 4 single-origin CDN seed URLs (each
with a unique ?cb=<uuid> cache-buster so the CDN treats them as
distinct). Reports wall time and outcome class counts.

Replace the SEEDS list below with image URLs reachable from your test
environment before running.

Usage:
  # as-shipped defaults (dynamo=30s, vllm=5s)
  python repro.py --client dynamo --n 5000
  python repro.py --client vllm   --n 5000

  # mechanism-isolating: same 30s budget for both
  python repro.py --client dynamo --n 5000 --timeout 30
  python repro.py --client vllm   --n 5000 --timeout 30
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import Counter

import dynamo_client
import vllm_client
from args import parse_args

SEEDS = [
    "https://example.com/image-1.jpg",
    "https://example.com/image-2.jpg",
    "https://example.com/image-3.jpg",
    "https://example.com/image-4.jpg",
]


def gen_urls(n: int) -> list[str]:
    return [f"{SEEDS[i % len(SEEDS)]}?cb={uuid.uuid4().hex}" for i in range(n)]


async def main() -> None:
    args = parse_args()
    mod = dynamo_client if args.client == "dynamo" else vllm_client
    await mod.reset()

    urls = gen_urls(args.n)
    t0 = time.perf_counter()
    results = await asyncio.gather(
        *(mod.fetch(u, timeout=args.timeout) for u in urls), return_exceptions=True
    )
    wall = time.perf_counter() - t0
    await mod.reset()

    counter: Counter[str] = Counter()
    for r in results:
        if isinstance(r, BaseException):
            counter[type(r).__name__] += 1
        else:
            counter["success"] += 1

    total = len(results)
    print(f"[{args.client}] n={total} wall={wall:.1f}s")
    for k, v in counter.most_common():
        print(f"  {k:<28} {v:>6}  {100.0 * v / total:5.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
