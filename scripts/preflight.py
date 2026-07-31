#!/usr/bin/env python3
"""Run descriptor-rooted launch admission checks only; never starts a runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inkling_release.preflight import PreflightError, run_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--runtime-manifest-sha256")
    parser.add_argument("--require-runtime-binding", action="store_true")
    args = parser.parse_args()
    try:
        result = run_preflight(
            args.admission,
            args.model_root,
            expected_runtime_manifest_sha256=args.runtime_manifest_sha256,
            require_runtime_binding=args.require_runtime_binding,
        )
    except (OSError, ValueError, PreflightError) as exc:
        print(json.dumps({"schema_version": 1, "status": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
