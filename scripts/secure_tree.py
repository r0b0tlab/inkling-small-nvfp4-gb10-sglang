#!/usr/bin/env python3
"""Descriptor-rooted, no-follow helpers for point-in-time release-tree verification.

All path components are opened relative to descriptors without following symlinks. These
checks detect mutation during a read, but they do not defend against an uncooperative
same-UID process that changes a pathname after verification returns.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Callable

CHUNK_SIZE = 8 * 1024 * 1024
BeforeLeafOpen = Callable[[str], None]


@dataclass(frozen=True)
class FileDigest:
    size: int
    digest: str
    device: int
    inode: int
    mode: int
    link_count: int


def validate_relative_path(value: str) -> tuple[str, ...]:
    path = PurePosixPath(value)
    parts = path.parts
    if (
        not value
        or path.is_absolute()
        or value != path.as_posix()
        or any(part in ("", ".", "..") for part in parts)
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"unsafe relative path: {value!r}")
    return parts


def _absolute_components(path: Path) -> tuple[Path, tuple[str, ...]]:
    raw = os.fspath(path)
    if not raw or "\x00" in raw:
        raise ValueError(f"invalid root path: {path!s}")
    supplied = PurePosixPath(raw)
    if any(part in (".", "..") for part in supplied.parts):
        raise ValueError(f"root path cannot contain dot components: {path}")
    if not supplied.is_absolute():
        supplied = PurePosixPath(os.getcwd()) / supplied
    absolute = Path(supplied.as_posix())
    return absolute, tuple(part for part in supplied.parts if part != "/")


def open_root(path: Path) -> tuple[int, os.stat_result]:
    absolute, components = _absolute_components(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open("/", flags)
    try:
        for component in components:
            try:
                next_fd = os.open(component, flags, dir_fd=fd)
            except OSError as exc:
                raise ValueError(
                    f"root ancestor is unavailable, moved, or rejected by nofollow: "
                    f"{absolute}: {exc}"
                ) from exc
            os.close(fd)
            fd = next_fd
        identity = os.fstat(fd)
        if not stat.S_ISDIR(identity.st_mode):
            raise ValueError(f"root is not a directory: {absolute}")
        return fd, identity
    except Exception:
        os.close(fd)
        raise


def _open_parent(root_fd: int, parts: tuple[str, ...]) -> tuple[int, str]:
    parent_fd = os.dup(root_fd)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        for component in parts[:-1]:
            next_fd = os.open(component, flags, dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        return parent_fd, parts[-1]
    except Exception:
        os.close(parent_fd)
        raise


def hash_relative_file(
    root_fd: int,
    relative: str,
    algorithm: str,
    *,
    expected_size: int | None = None,
    before_leaf_open: BeforeLeafOpen | None = None,
) -> FileDigest:
    parts = validate_relative_path(relative)
    parent_fd, leaf = _open_parent(root_fd, parts)
    file_fd = -1
    try:
        if before_leaf_open is not None:
            before_leaf_open(relative)
        flags = os.O_RDONLY | os.O_CLOEXEC
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            file_fd = os.open(leaf, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise ValueError(f"nofollow open rejected {relative}: {exc}") from exc
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular file: {relative}")
        if before.st_nlink != 1:
            raise ValueError(f"link count is not one: {relative}")
        if expected_size is not None and before.st_size != expected_size:
            raise ValueError(
                f"size mismatch for {relative}: expected {expected_size}, got {before.st_size}"
            )
        if algorithm == "sha256":
            digest = hashlib.sha256()
        elif algorithm == "git-sha1":
            digest = hashlib.sha1()
            digest.update(f"blob {before.st_size}\0".encode("ascii"))
        else:
            raise ValueError(f"unsupported digest algorithm: {algorithm}")
        while True:
            block = os.read(file_fd, CHUNK_SIZE)
            if not block:
                break
            digest.update(block)
        after = os.fstat(file_fd)
        named = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        immutable_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        immutable_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        named_identity = (
            named.st_dev,
            named.st_ino,
            named.st_mode,
            named.st_nlink,
            named.st_size,
            named.st_mtime_ns,
            named.st_ctime_ns,
        )
        if immutable_before != immutable_after or immutable_after != named_identity:
            raise ValueError(f"file changed while hashing: {relative}")
        return FileDigest(
            size=after.st_size,
            digest=digest.hexdigest(),
            device=after.st_dev,
            inode=after.st_ino,
            mode=stat.S_IMODE(after.st_mode),
            link_count=after.st_nlink,
        )
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)


def inventory_regular_files(
    root_fd: int,
    *,
    ignored_top_level: frozenset[str] = frozenset(),
) -> tuple[list[str], list[str]]:
    regular: list[str] = []
    rejected: list[str] = []

    def walk(directory_fd: int, prefix: tuple[str, ...]) -> None:
        for name in sorted(os.listdir(directory_fd)):
            if not prefix and name in ignored_top_level:
                continue
            relative = "/".join((*prefix, name))
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError:
                rejected.append(relative)
                continue
            if stat.S_ISDIR(info.st_mode):
                flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
                flags |= getattr(os, "O_NOFOLLOW", 0)
                try:
                    child_fd = os.open(name, flags, dir_fd=directory_fd)
                except OSError:
                    rejected.append(relative)
                    continue
                try:
                    walk(child_fd, (*prefix, name))
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                regular.append(relative)
            else:
                rejected.append(relative)

    walk(root_fd, ())
    return sorted(regular), sorted(rejected)


def revalidate_root(path: Path, root_fd: int, initial: os.stat_result) -> None:
    held = os.fstat(root_fd)
    rebound_fd = -1
    try:
        rebound_fd, named = open_root(path)
    except ValueError as exc:
        raise ValueError(f"root path disappeared, changed, or became symlinked: {path}: {exc}") from exc
    finally:
        if rebound_fd >= 0:
            os.close(rebound_fd)
    expected = (initial.st_dev, initial.st_ino, initial.st_mode)
    if (held.st_dev, held.st_ino, held.st_mode) != expected:
        raise ValueError("held root identity changed")
    if (named.st_dev, named.st_ino, named.st_mode) != expected:
        raise ValueError("root path no longer names the verified directory")
