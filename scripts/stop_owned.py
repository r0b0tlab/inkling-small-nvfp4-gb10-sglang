#!/usr/bin/env python3
"""Stop one explicitly named owned container; never perform broad cleanup."""
from __future__ import annotations

import argparse
import json
import subprocess

from inkling_release.stop import build_stop_argv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", required=True)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        argv = build_stop_argv(args.container, timeout_seconds=args.timeout)
        result = {"schema_version": 1, "status": "DRY_RUN", "argv": argv, "execute": False}
        if args.execute:
            completed = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=args.timeout + 10)
            result.update({"status": "STOPPED" if completed.returncode == 0 else "FAIL", "returncode": completed.returncode, "execute": True})
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"schema_version": 1, "status": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
