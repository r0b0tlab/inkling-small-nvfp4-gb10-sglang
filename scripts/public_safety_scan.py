#!/usr/bin/env python3
"""Fail-closed scan of every tracked public artifact root supplied."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
from typing import Iterable

try:
    from secure_tree import open_root
except ImportError:
    from scripts.secure_tree import open_root  # type: ignore[no-redef]

PATTERNS = (
    ("home-path", re.compile(r"/home/[A-Za-z0-9_.-]+")),
    ("github-token", re.compile(r"(?:ghp|github_pat)_[A-Za-z0-9_\-]{20,}")),
    ("hf-token", re.compile(r"\bhf_[A-Za-z0-9]{24,}\b")),
    ("api-key", re.compile(r"\b(?:sk|xai)-[A-Za-z0-9_\-]{20,}\b")),
    ("email-address", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("private-ip", re.compile(r"\b(?:10|192\.168|172\.(?:1[6-9]|2[0-9]|3[01]))\.\d+\.\d+\b")),
    (
        "secret-assignment",
        re.compile(r"(?:GITHUB_TOKEN|HF_TOKEN|OPENAI_API_KEY|XAI_API_KEY)\s*=\s*[^\s\"']+"),
    ),
)
MEDIA_SUFFIXES = {
    ".gif",
    ".jpeg",
    ".jpg",
    ".mp3",
    ".mp4",
    ".pdf",
    ".png",
    ".wav",
    ".webm",
    ".webp",
}
MAX_SCAN_BYTES = 128 * 1024 * 1024


def validated_root(root: Path) -> Path:
    root_fd = -1
    try:
        root_fd, _ = open_root(root)
    finally:
        if root_fd >= 0:
            os.close(root_fd)
    return Path(os.path.abspath(os.fspath(root)))


def tracked_or_exported_files(root: Path) -> list[Path]:
    root = validated_root(root)
    top = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if top.returncode == 0 and Path(top.stdout.strip()) == root:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError("git ls-files failed for the public repository root")
        return sorted(root / item.decode() for item in result.stdout.split(b"\0") if item)
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_symlink()
        or (
            path.is_file()
            and ".git" not in path.relative_to(root).parts
            and "__pycache__" not in path.relative_to(root).parts
        )
    )


def scan_files(paths: Iterable[Path], allowed: set[Path] | None = None) -> list[str]:
    allowed = {path.resolve() for path in (allowed or set())}
    failures: list[str] = []
    for path in paths:
        if path.is_symlink():
            failures.append(f"{path}: tracked symlink is forbidden")
            continue
        if path.resolve() in allowed:
            continue
        if not path.is_file():
            failures.append(f"{path}: tracked artifact is missing or non-regular")
            continue
        size = path.stat().st_size
        if size > MAX_SCAN_BYTES:
            failures.append(
                f"{path}: artifact exceeds fail-closed scan limit {MAX_SCAN_BYTES}"
            )
            continue
        raw = path.read_bytes()
        pattern_text = raw.decode("latin-1")
        for name, regex in PATTERNS:
            for match in regex.finditer(pattern_text):
                failures.append(f"{path}: {name}: {match.group(0)[:120]}")
        if path.suffix.lower() in MEDIA_SUFFIXES:
            continue
        if b"\0" in raw:
            failures.append(f"{path}: NUL byte is forbidden in a public text artifact")
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            failures.append(f"{path}: invalid UTF-8 in a public text artifact")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    for root in args.roots:
        try:
            files = tracked_or_exported_files(root)
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append(
                f"{root}: invalid public root (symlink/nofollow check): {exc}"
            )
            continue
        failures.extend(scan_files(files))
    if failures:
        print("PUBLIC_SAFETY_SCAN=FAIL")
        for failure in failures[:200]:
            print(failure)
        return 2
    print(f"PUBLIC_SAFETY_SCAN=PASS roots={len(args.roots)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
