"""CLI argument parsing for the minimal reproducer."""

from __future__ import annotations

import argparse

import dynamo_args


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Minimal pool-saturation reproducer: Dynamo (httpx) vs vLLM (aiohttp).",
    )
    p.add_argument("--client", choices=["dynamo", "vllm"], required=True,
                   help="which client mirror to exercise")
    p.add_argument("--n", type=int, default=5000,
                   help="number of concurrent fetches fired via asyncio.gather")
    p.add_argument("--timeout", type=float, default=None,
                   help="per-request timeout override (default: each client's shipped default, "
                        "30s for dynamo / 5s for vllm)")
    dynamo_args.add_arguments(p)
    args = p.parse_args()
    if args.client == "dynamo":
        dynamo_args.apply(args)
    return args
