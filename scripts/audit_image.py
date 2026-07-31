#!/usr/bin/env python3
"""Audit locally supplied immutable image metadata; no registry or Docker access."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from inkling_release.image import audit_image_metadata
from inkling_release.manifest import IMAGE_ARM64_MANIFEST, IMAGE_CONFIG, SOURCE_COMMIT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    try:
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        result = audit_image_metadata(
            metadata,
            expected_arm64_manifest=IMAGE_ARM64_MANIFEST,
            expected_config=IMAGE_CONFIG,
            expected_source_commit=SOURCE_COMMIT,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAIL", "reason": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
