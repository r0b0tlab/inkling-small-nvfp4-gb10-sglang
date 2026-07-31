#!/usr/bin/env python3
"""Render bounded tuning candidates without executing them."""
from __future__ import annotations

import argparse
import json

from inkling_release.tuning import plan_tuning


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mem-fraction-static", type=float, action="append")
    parser.add_argument("--context-length", type=int, action="append")
    parser.add_argument("--chunk-size", type=int, action="append")
    parser.add_argument("--concurrency", type=int, action="append")
    args = parser.parse_args()
    plans = plan_tuning(
        mem_fraction_static=args.mem_fraction_static,
        context_lengths=args.context_length,
        chunk_sizes=args.chunk_size,
        concurrency=args.concurrency,
    )
    print(json.dumps({"schema_version": 1, "status": "NOT_RUN", "plans": plans}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
