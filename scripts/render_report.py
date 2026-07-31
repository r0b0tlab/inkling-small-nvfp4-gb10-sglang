#!/usr/bin/env python3
"""Render a safe evidence summary without raw logs or runtime claims."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inkling_release.report import render_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    value = json.loads(args.evidence.read_text(encoding="utf-8"))
    result = render_report(value)
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
