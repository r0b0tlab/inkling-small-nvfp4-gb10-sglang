#!/usr/bin/env python3
"""Create a deterministic, descriptor-bound SHA-256 manifest for an evidence tree."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

try:
    from secure_tree import (
        hash_relative_file,
        inventory_regular_files,
        open_root,
        revalidate_root,
    )
except ImportError:
    from scripts.secure_tree import (  # type: ignore[no-redef]
        hash_relative_file,
        inventory_regular_files,
        open_root,
        revalidate_root,
    )


def build_manifest(
    root: Path,
    *,
    before_leaf_open: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    root_fd, root_identity = open_root(root)
    try:
        observed_before, rejected_before = inventory_regular_files(root_fd)
        if rejected_before:
            raise ValueError(
                "evidence tree contains a symlink, linked file, or special entry: "
                + ", ".join(rejected_before)
            )
        files: list[dict[str, Any]] = []
        total_bytes = 0
        for relative in observed_before:
            result = hash_relative_file(
                root_fd,
                relative,
                "sha256",
                before_leaf_open=before_leaf_open,
            )
            files.append(
                {
                    "path": relative,
                    "size": result.size,
                    "sha256": result.digest,
                }
            )
            total_bytes += result.size
        observed_after, rejected_after = inventory_regular_files(root_fd)
        if observed_after != observed_before or rejected_after != rejected_before:
            raise ValueError("evidence tree changed while hashing")
        revalidate_root(root, root_fd, root_identity)
        return {
            "schema_version": 2,
            "file_count": len(files),
            "total_bytes": total_bytes,
            "files": files,
        }
    finally:
        os.close(root_fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root
    payload = json.dumps(build_manifest(root), indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
