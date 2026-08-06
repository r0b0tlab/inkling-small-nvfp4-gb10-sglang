#!/usr/bin/env python3
"""Verify an embedded AOT cache against its content-addressed manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


class CacheVerificationError(ValueError):
    """Raised when the cache does not match its manifest contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative_path(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise CacheVerificationError("manifest path is not a non-empty string")
    posix = PurePosixPath(value)
    if posix.is_absolute() or ".." in posix.parts:
        raise CacheVerificationError(f"unsafe manifest path: {value!r}")
    path = Path(*posix.parts)
    if path == Path("."):
        raise CacheVerificationError("manifest path names the cache root")
    return path


def verify_cache(root: Path, manifest_path: Path) -> dict[str, int]:
    root = root.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise CacheVerificationError("manifest files must be a list")

    expected: dict[Path, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise CacheVerificationError("manifest entry is not an object")
        relative = _safe_relative_path(entry.get("path"))
        if relative in expected:
            raise CacheVerificationError(f"duplicate manifest path: {relative}")
        expected[relative] = entry

    actual: list[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise CacheVerificationError(f"cache contains a symlink: {candidate}")
        if candidate.is_file():
            actual.append(candidate.relative_to(root))
        elif not candidate.is_dir():
            raise CacheVerificationError(f"cache contains a non-file/non-directory: {candidate}")

    actual_set = set(actual)
    expected_set = set(expected)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing or extra:
        raise CacheVerificationError(f"cache inventory mismatch: missing={missing[:3]} extra={extra[:3]}")

    total_bytes = 0
    for relative, entry in expected.items():
        path = root / relative
        mode = path.stat().st_mode
        if mode & 0o222:
            raise CacheVerificationError(f"cache file is writable: {relative}")
        size = path.stat().st_size
        digest = _sha256(path)
        if entry.get("bytes") != size or entry.get("sha256") != digest:
            raise CacheVerificationError(f"cache file hash/size mismatch: {relative}")
        total_bytes += size

    if manifest.get("file_count") != len(expected):
        raise CacheVerificationError("manifest file_count mismatch")
    if manifest.get("total_bytes") != total_bytes:
        raise CacheVerificationError("manifest total_bytes mismatch")
    return {"file_count": len(expected), "total_bytes": total_bytes}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    result = verify_cache(args.root, args.manifest)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
