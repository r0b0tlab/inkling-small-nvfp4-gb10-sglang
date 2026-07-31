#!/usr/bin/env python3
"""Render a serving benchmark plan; this wrapper never calls a live endpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inkling_release.bench import render_benchmark_plan
from inkling_release.profiles import load_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--dist-init-addr", default="127.0.0.1:5000")
    parser.add_argument("--execute", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute:
        parser.error("live benchmark execution is not part of the non-live foundation")
    print(json.dumps(render_benchmark_plan(load_profile(args.profile), node_rank=args.node_rank, dist_init_addr=args.dist_init_addr), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
