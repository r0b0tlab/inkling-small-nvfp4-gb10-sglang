#!/usr/bin/env python3
"""Render a launch command without executing it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inkling_release.profiles import load_profile
from inkling_release.render import render_command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--dist-init-addr", required=True)
    parser.add_argument("--model-root", default="/models/Inkling-Small-NVFP4")
    parser.add_argument("--image-ref")
    args = parser.parse_args()
    result = render_command(
        load_profile(args.profile),
        node_rank=args.node_rank,
        dist_init_addr=args.dist_init_addr,
        model_root=args.model_root,
        image_ref=args.image_ref,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
