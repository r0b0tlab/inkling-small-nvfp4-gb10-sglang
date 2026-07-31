#!/usr/bin/env python3
"""Build an owned launch spec; execution is a separate explicit opt-in boundary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inkling_release.launch import build_launch_spec, launch


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("runtime-manifest.json"))
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True, help="private owner-matched executable cache directory")
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--dist-init-addr", required=True)
    parser.add_argument("--image-ref", help="explicit frozen derivative image reference (repo@sha256:...)")
    parser.add_argument("--pass-env", action="append", default=[], metavar="NAME", help="pass one allowlisted runtime environment name")
    parser.add_argument("--readiness-url", help="optional loopback readiness URL for the live boundary")
    parser.add_argument("--readiness-timeout", type=float, default=30.0)
    parser.add_argument("--receipt", type=Path, default=Path("receipts/inkling-sglang.json"))
    parser.add_argument("--execute", action="store_true", help="explicitly start the owned derivative Docker container")
    args = parser.parse_args()
    if args.execute and not args.image_ref:
        parser.error("--execute requires an explicit frozen derivative --image-ref")
    try:
        spec = build_launch_spec(
            profile_path=args.profile,
            manifest_path=args.manifest,
            admission_path=args.admission,
            model_root=args.model_root,
            cache_root=args.cache_root,
            node_rank=args.node_rank,
            dist_init_addr=args.dist_init_addr,
            image_ref=args.image_ref,
            env_passthrough=args.pass_env,
        )
        result = launch(
            spec,
            receipt_path=args.receipt,
            execute=args.execute,
            readiness_url=args.readiness_url,
            readiness_timeout=args.readiness_timeout,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"schema_version": 1, "status": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
