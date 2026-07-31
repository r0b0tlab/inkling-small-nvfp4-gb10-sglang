#!/usr/bin/env python3
"""Evaluate local, non-live gate inputs and emit PHASE0_PASS or NO_VERDICT."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inkling_release.gates import evaluate_gates
from inkling_release.manifest import load_runtime_manifest, manifest_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("runtime-manifest.json"))
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--evidence-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        gates = json.loads(args.gates.read_text(encoding="utf-8"))
        if not isinstance(gates, dict):
            raise ValueError("gates must be a JSON object")
        result = evaluate_gates(gates, runtime_manifest_sha256=manifest_sha256(load_runtime_manifest(args.manifest)), evidence_manifest_sha256=args.evidence_manifest_sha256)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result = {"schema_version": 1, "status": "NO_VERDICT", "reason": str(exc)}
    raw = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(raw, encoding="utf-8")
    print(raw, end="")
    return 0 if result["status"] == "PHASE0_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
