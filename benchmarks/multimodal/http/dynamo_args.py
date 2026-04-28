"""CLI knobs for the Dynamo (httpx) client mirror.

Lets users override the module-level constants in dynamo_client so the
singleton AsyncClient is built with their values.
"""

from __future__ import annotations

import argparse

import dynamo_client


def add_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("dynamo client (httpx)")
    g.add_argument(
        "--max-connections", type=int, default=dynamo_client.MAX_CONNECTIONS,
        help=f"httpx Limits.max_connections (default: {dynamo_client.MAX_CONNECTIONS})",
    )
    g.add_argument(
        "--max-keepalive", type=int, default=dynamo_client.MAX_KEEPALIVE,
        help=f"httpx Limits.max_keepalive_connections "
             f"(default: {dynamo_client.MAX_KEEPALIVE}). Raising this to match "
             "--max-connections eliminates TLS re-handshake churn — the single "
             "flag that turned Dynamo from 1.3%% success into 99.9%% in our "
             "COCO test.",
    )


def apply(args: argparse.Namespace) -> None:
    """Mutate dynamo_client module constants so get_client() picks up overrides."""
    dynamo_client.MAX_CONNECTIONS = args.max_connections
    dynamo_client.MAX_KEEPALIVE = args.max_keepalive
