#!/usr/bin/env python3
"""Bind an existing PHASE0 admission artifact to the frozen runtime manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inkling_release.manifest import load_runtime_manifest, manifest_sha256
from inkling_release.preflight import PreflightError, bind_admission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("runtime-manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path)
    args = parser.parse_args()
    try:
        digest = manifest_sha256(load_runtime_manifest(args.manifest))
        result = bind_admission(
            args.admission,
            args.output,
            runtime_manifest_sha256=digest,
            model_root=args.model_root,
        )
    except (OSError, ValueError, PreflightError) as exc:
        print(json.dumps({"schema_version": 1, "status": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps({"schema_version": 1, "status": "BOUND", "output": str(args.output), "runtime_manifest_sha256": result["runtime_manifest_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
