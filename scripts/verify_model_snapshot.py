#!/usr/bin/env python3
"""Verify the frozen Inkling snapshot through descriptor-rooted no-follow reads."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable

try:
    from secure_tree import (
        hash_relative_file,
        inventory_regular_files,
        open_root,
        revalidate_root,
        validate_relative_path,
    )
except ImportError:
    from scripts.secure_tree import (  # type: ignore[no-redef]
        hash_relative_file,
        inventory_regular_files,
        open_root,
        revalidate_root,
        validate_relative_path,
    )

MODEL_ID = "thinkingmachines/Inkling-Small-NVFP4"
MODEL_REVISION = "b6a99534467840620d411e4cd4ad5819b2610d9c"
MODEL_FILE_COUNT = 21
MODEL_SNAPSHOT_BYTES = 170_764_923_366
MODEL_MANIFEST_SHA256 = "8b46f4b3c1d47a31341acee60b42d3340d5937744b90752605d177481a9470e4"
FROZEN_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "manifests/inkling-small-nvfp4-b6a99534.json"
)
MARKER_NAME = ".r0b0tlab-snapshot.json"
IGNORED_TOP_LEVEL = frozenset({".cache", MARKER_NAME})
_HEX = frozenset("0123456789abcdef")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(raw: bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    return json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_pairs,
        parse_constant=reject_constant,
    )


def _read_regular_nofollow(path: Path, *, maximum_bytes: int) -> bytes:
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError(f"manifest is not an immutable regular file: {path}")
        if info.st_size > maximum_bytes:
            raise ValueError(f"manifest exceeds {maximum_bytes} bytes")
        chunks: list[bytes] = []
        remaining = info.st_size + 1
        while remaining:
            block = os.read(fd, min(remaining, 1024 * 1024))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        after = os.fstat(fd)
        named = os.stat(path, follow_symlinks=False)
        identity = (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size)
        if identity != (after.st_dev, after.st_ino, after.st_mode, after.st_nlink, after.st_size):
            raise ValueError("manifest changed while reading")
        if identity != (named.st_dev, named.st_ino, named.st_mode, named.st_nlink, named.st_size):
            raise ValueError("manifest path was rebound while reading")
        return raw
    finally:
        os.close(fd)


def validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    required = {
        "schema_version",
        "repo_id",
        "revision",
        "source",
        "file_count",
        "total_bytes",
        "files",
    }
    if set(manifest) != required:
        raise ValueError("manifest keys do not match the frozen schema")
    if manifest["schema_version"] != 1 or type(manifest["schema_version"]) is not int:
        raise ValueError("manifest schema_version must be integer 1")
    if not isinstance(manifest["repo_id"], str) or not isinstance(manifest["revision"], str):
        raise ValueError("manifest identity fields must be strings")
    if not isinstance(manifest["source"], str):
        raise ValueError("manifest source must be a string")
    if type(manifest["file_count"]) is not int or manifest["file_count"] < 0:
        raise ValueError("manifest file_count must be a non-negative integer")
    if type(manifest["total_bytes"]) is not int or manifest["total_bytes"] < 0:
        raise ValueError("manifest total_bytes must be a non-negative integer")
    files = manifest["files"]
    if not isinstance(files, list):
        raise ValueError("manifest files must be an array")
    paths: set[str] = set()
    total_bytes = 0
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(files):
        if not isinstance(item, dict) or set(item) != {
            "path",
            "size",
            "digest_algorithm",
            "digest",
        }:
            raise ValueError(f"manifest file {index} has an invalid schema")
        path = item["path"]
        size = item["size"]
        algorithm = item["digest_algorithm"]
        digest = item["digest"]
        if not isinstance(path, str):
            raise ValueError(f"manifest file {index} path is not a string")
        validate_relative_path(path)
        if path in paths:
            raise ValueError(f"duplicate manifest path: {path}")
        paths.add(path)
        if type(size) is not int or size < 0:
            raise ValueError(f"manifest size is invalid: {path}")
        if algorithm not in {"sha256", "git-sha1"}:
            raise ValueError(f"manifest algorithm is invalid: {path}")
        expected_length = 64 if algorithm == "sha256" else 40
        if (
            not isinstance(digest, str)
            or len(digest) != expected_length
            or any(character not in _HEX for character in digest)
        ):
            raise ValueError(f"manifest digest is invalid: {path}")
        total_bytes += size
        normalized.append(dict(item))
    if len(files) != manifest["file_count"] or total_bytes != manifest["total_bytes"]:
        raise ValueError("manifest aggregate counts do not match its files")
    return {**manifest, "files": normalized}


def load_frozen_manifest(path: Path = FROZEN_MANIFEST_PATH) -> dict[str, Any]:
    raw = _read_regular_nofollow(path, maximum_bytes=1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != MODEL_MANIFEST_SHA256:
        raise ValueError("frozen manifest SHA-256 mismatch")
    manifest = validate_manifest(_strict_json(raw))
    if (
        manifest["repo_id"] != MODEL_ID
        or manifest["revision"] != MODEL_REVISION
        or manifest["file_count"] != MODEL_FILE_COUNT
        or manifest["total_bytes"] != MODEL_SNAPSHOT_BYTES
    ):
        raise ValueError("frozen manifest identity mismatch")
    return manifest


def verify_snapshot_entries(
    root: Path,
    manifest: dict[str, Any],
    *,
    before_leaf_open: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    checked_manifest = validate_manifest(manifest)
    expected_files = [item["path"] for item in checked_manifest["files"]]
    expected_set = set(expected_files)
    mismatches: list[dict[str, Any]] = []
    verified_files = 0
    verified_bytes = 0
    root_fd, root_identity = open_root(root)
    try:
        observed_before, rejected_before = inventory_regular_files(
            root_fd, ignored_top_level=IGNORED_TOP_LEVEL
        )
        for rejected in rejected_before:
            mismatches.append({"path": rejected, "reason": "non-regular-or-linked-entry"})
        for extra in sorted(set(observed_before) - expected_set):
            mismatches.append({"path": extra, "reason": "unexpected-file"})
        for missing in sorted(expected_set - set(observed_before)):
            mismatches.append({"path": missing, "reason": "missing-file"})
        for item in checked_manifest["files"]:
            relative = item["path"]
            if relative not in observed_before:
                continue
            try:
                result = hash_relative_file(
                    root_fd,
                    relative,
                    item["digest_algorithm"],
                    expected_size=item["size"],
                    before_leaf_open=before_leaf_open,
                )
            except (OSError, ValueError) as exc:
                mismatches.append({"path": relative, "reason": str(exc)})
                continue
            if result.digest != item["digest"]:
                mismatches.append(
                    {
                        "path": relative,
                        "reason": "digest-mismatch",
                        "expected": item["digest"],
                        "actual": result.digest,
                    }
                )
                continue
            verified_files += 1
            verified_bytes += result.size
        observed_after, rejected_after = inventory_regular_files(
            root_fd, ignored_top_level=IGNORED_TOP_LEVEL
        )
        if observed_after != observed_before or rejected_after != rejected_before:
            mismatches.append({"path": ".", "reason": "snapshot-tree-changed-during-verification"})
        revalidate_root(root, root_fd, root_identity)
        return {
            "schema_version": 2,
            "status": "PASS" if not mismatches else "FAIL",
            "repo_id": checked_manifest["repo_id"],
            "revision": checked_manifest["revision"],
            "expected_files": checked_manifest["file_count"],
            "expected_bytes": checked_manifest["total_bytes"],
            "verified_files": verified_files,
            "verified_bytes": verified_bytes,
            "root_device": root_identity.st_dev,
            "root_inode": root_identity.st_ino,
            "mismatches": mismatches,
        }
    finally:
        os.close(root_fd)


def verify_snapshot(
    root: Path,
    manifest: dict[str, Any] | None = None,
    *,
    before_leaf_open: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    frozen = load_frozen_manifest()
    if manifest is not None and validate_manifest(manifest) != frozen:
        raise ValueError("caller manifest does not match the frozen manifest")
    return verify_snapshot_entries(root, frozen, before_leaf_open=before_leaf_open)


def _link_tmpfile(file_fd: int, root_fd: int, destination: str) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    linkat = libc.linkat
    linkat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    linkat.restype = ctypes.c_int
    if linkat(file_fd, b"", root_fd, destination.encode("utf-8"), 0x1000) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, os.strerror(error), destination)
        raise OSError(error, os.strerror(error), destination)


def write_completion_marker(
    root: Path,
    manifest_path: Path = FROZEN_MANIFEST_PATH,
) -> tuple[Path, dict[str, Any]]:
    """Re-verify the frozen tree and atomically install a point-in-time evidence marker."""
    raw_manifest = _read_regular_nofollow(manifest_path, maximum_bytes=1024 * 1024)
    if hashlib.sha256(raw_manifest).hexdigest() != MODEL_MANIFEST_SHA256:
        raise ValueError("completion marker requires the frozen manifest bytes")
    load_frozen_manifest(manifest_path)
    report = verify_snapshot(root)
    if report.get("status") != "PASS":
        raise ValueError("snapshot verification failed; completion marker was not written")
    root_fd, root_identity = open_root(root)
    try:
        if report.get("root_device") != root_identity.st_dev or report.get("root_inode") != root_identity.st_ino:
            raise ValueError("verification report is not bound to the current snapshot root")
        payload = {
            "schema_version": 2,
            "status": "COMPLETE",
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "manifest_sha256": MODEL_MANIFEST_SHA256,
            "verified_files": MODEL_FILE_COUNT,
            "verified_bytes": MODEL_SNAPSHOT_BYTES,
            "root_device": root_identity.st_dev,
            "root_inode": root_identity.st_ino,
        }
        data = (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        flags = os.O_WRONLY | os.O_CLOEXEC | getattr(os, "O_TMPFILE", 0)
        marker_fd = os.open(".", flags, 0o400, dir_fd=root_fd)
        marker_linked = False
        try:
            os.fchmod(marker_fd, 0o400)
            offset = 0
            while offset < len(data):
                written = os.write(marker_fd, data[offset:])
                if written <= 0:
                    raise OSError("short write while creating completion marker")
                offset += written
            os.fsync(marker_fd)
            _link_tmpfile(marker_fd, root_fd, MARKER_NAME)
            marker_linked = True
            os.fsync(root_fd)
            installed = os.stat(MARKER_NAME, dir_fd=root_fd, follow_symlinks=False)
            held = os.fstat(marker_fd)
            if (
                not stat.S_ISREG(installed.st_mode)
                or stat.S_IMODE(installed.st_mode) != 0o400
                or installed.st_nlink != 1
                or installed.st_size != len(data)
                or (installed.st_dev, installed.st_ino) != (held.st_dev, held.st_ino)
            ):
                raise ValueError("installed completion marker has invalid metadata")
            revalidate_root(root, root_fd, root_identity)
        except Exception:
            if marker_linked:
                try:
                    named = os.stat(MARKER_NAME, dir_fd=root_fd, follow_symlinks=False)
                    held = os.fstat(marker_fd)
                    if (named.st_dev, named.st_ino) == (held.st_dev, held.st_ino):
                        os.unlink(MARKER_NAME, dir_fd=root_fd)
                        os.fsync(root_fd)
                except OSError:
                    pass
            raise
        finally:
            os.close(marker_fd)
        return root / MARKER_NAME, report
    finally:
        os.close(root_fd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--write-marker", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.write_marker:
            marker, report = write_completion_marker(args.root)
            report["completion_marker"] = str(marker)
        else:
            report = verify_snapshot(args.root)
    except (OSError, ValueError) as exc:
        report = {
            "schema_version": 2,
            "status": "FAIL",
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "verified_files": 0,
            "verified_bytes": 0,
            "mismatches": [{"path": ".", "reason": str(exc)}],
        }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
