#!/usr/bin/env python3
"""Validate an evidence artifact without upgrading its verdict."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inkling_release.evidence import EvidenceError, accept_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    try:
        value = json.loads(args.evidence.read_text(encoding="utf-8"))
        result = accept_evidence(value)
    except (OSError, ValueError, json.JSONDecodeError, EvidenceError) as exc:
        print(json.dumps({"schema_version": 1, "status": "REJECTED", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
