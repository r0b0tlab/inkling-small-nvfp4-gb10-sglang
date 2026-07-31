from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any


class PreflightError(ValueError):
    """A launch admission precondition failed."""


_MAX_ADMISSION_BYTES = 256 * 1024
_HEX = frozenset("0123456789abcdef")


def _read_json(path: Path, limit: int = _MAX_ADMISSION_BYTES) -> dict[str, Any]:
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise PreflightError(f"cannot stat admission file: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise PreflightError("admission file must be a regular non-symlink with one link")
    if info.st_size > limit:
        raise PreflightError("admission file exceeds size limit")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PreflightError(f"cannot read admission file: {exc}") from exc

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise PreflightError(f"duplicate admission key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise PreflightError(f"non-standard admission JSON constant: {value}")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError, PreflightError) as exc:
        raise PreflightError("admission file is not strict JSON") from exc
    if not isinstance(value, dict):
        raise PreflightError("admission must be a JSON object")
    return value


def _validate_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _HEX for char in value):
        raise PreflightError(f"{field} must be lowercase SHA-256")
    return value


def validate_admission(value: Any, *, require_runtime_binding: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreflightError("admission must be an object")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise PreflightError("unsupported admission schema")
    if value.get("status") != "PHASE0_PASS":
        raise PreflightError("live launch requires admission status PHASE0_PASS")
    for key in ("descriptor_rooted", "non_symlink", "read_only"):
        if value.get(key) is not True:
            raise PreflightError(f"admission must assert {key}=true")
    if "read_only_mount" in value and value["read_only_mount"] != "ro":
        raise PreflightError("admission read_only_mount must be ro")
    if "runtime_manifest_sha256" in value:
        _validate_sha(value["runtime_manifest_sha256"], "runtime_manifest_sha256")
    elif require_runtime_binding:
        raise PreflightError(
            "admission is unbound: bind it to the runtime manifest with scripts/bind_admission.py"
        )
    if "model_root" in value and (
        not isinstance(value["model_root"], str) or not value["model_root"]
    ):
        raise PreflightError("admission model_root must be an absolute path string")
    return value


def _absolute_components(path: Path) -> tuple[str, tuple[str, ...]]:
    raw = os.fspath(path)
    if not raw or "\x00" in raw:
        raise PreflightError("model root contains an invalid NUL byte")
    supplied = PurePosixPath(raw)
    if not supplied.is_absolute() or any(part in ("", ".", "..") for part in supplied.parts):
        raise PreflightError("model root must be an absolute normalized path")
    return supplied.as_posix(), tuple(part for part in supplied.parts if part != "/")


def _open_directory_descriptor(path: Path) -> tuple[int, os.stat_result, str]:
    normalized, components = _absolute_components(path)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open("/", flags)
    try:
        for component in components:
            next_fd = os.open(component, flags, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        info = os.fstat(fd)
        if not stat.S_ISDIR(info.st_mode):
            raise PreflightError("model root is not a directory")
        if info.st_nlink < 1:
            raise PreflightError("model root has an invalid link count")
        return fd, info, normalized
    except Exception:
        os.close(fd)
        raise


def _check_read_only(path: Path, info: os.stat_result) -> tuple[bool, bool]:
    mode_read_only = not bool(info.st_mode & 0o222)
    try:
        flags = os.statvfs(path).f_flag
        fs_read_only = bool(flags & getattr(os, "ST_RDONLY", 1))
    except OSError as exc:
        raise PreflightError(f"cannot inspect model mount mode: {exc}") from exc
    if not (mode_read_only or fs_read_only):
        raise PreflightError("model root is not read-only by mode or mounted filesystem")
    return mode_read_only, fs_read_only


def _descriptor_rooted_directory(path: Path) -> dict[str, Any]:
    """Open every ancestor with O_NOFOLLOW; ownership is deliberately not required.

    Rank 1 may consume an admitted read-only NFS export owned by a service account.
    The admission and descriptor checks, rather than local UID equality, authorize it.
    """
    fd, info, normalized = _open_directory_descriptor(path)
    try:
        mode_read_only, fs_read_only = _check_read_only(path, info)
        rebound = os.fstat(fd)
        if (rebound.st_dev, rebound.st_ino, rebound.st_mode) != (info.st_dev, info.st_ino, info.st_mode):
            raise PreflightError("model root changed while being admitted")
        return {
            "model_root": normalized,
            "device": info.st_dev,
            "inode": info.st_ino,
            "uid": info.st_uid,
            "mode_read_only": mode_read_only,
            "filesystem_read_only": fs_read_only,
        }
    finally:
        os.close(fd)


def bind_admission(
    admission_path: Path,
    output_path: Path,
    *,
    runtime_manifest_sha256: str,
    model_root: Path | None = None,
) -> dict[str, Any]:
    """Create an explicit runtime-manifest binding for older PHASE0 evidence."""
    digest = _validate_sha(runtime_manifest_sha256, "runtime_manifest_sha256")
    admission = validate_admission(_read_json(admission_path), require_runtime_binding=False)
    if model_root is not None:
        checked = _descriptor_rooted_directory(model_root)
        admission["model_root"] = checked["model_root"]
    admission["runtime_manifest_sha256"] = digest
    admission["binding"] = "runtime-manifest-sha256"
    if output_path.is_symlink():
        raise PreflightError("bound admission output may not be a symlink")
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(admission, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(output_path)
    return admission


def run_preflight(
    admission_path: Path,
    model_root: Path,
    *,
    expected_runtime_manifest_sha256: str | None = None,
    require_runtime_binding: bool = False,
) -> dict[str, Any]:
    admission = validate_admission(
        _read_json(admission_path),
        require_runtime_binding=require_runtime_binding or expected_runtime_manifest_sha256 is not None,
    )
    if expected_runtime_manifest_sha256 is not None:
        expected = _validate_sha(expected_runtime_manifest_sha256, "expected_runtime_manifest_sha256")
        actual = admission.get("runtime_manifest_sha256")
        if actual is None:
            raise PreflightError(
                "admission is unbound: bind it to the runtime manifest with scripts/bind_admission.py"
            )
        if actual != expected:
            raise PreflightError("admission runtime manifest digest mismatch")
    checked = _descriptor_rooted_directory(model_root)
    admitted_root = admission.get("model_root")
    if admitted_root is not None and admitted_root != checked["model_root"]:
        raise PreflightError("admission model_root does not match the descriptor-rooted path")
    return {
        "schema_version": 1,
        "status": "PASS",
        "admission_status": admission["status"],
        "model_root": checked["model_root"],
        "model_root_uid": checked["uid"],
        "model_root_device": checked["device"],
        "model_root_inode": checked["inode"],
        "descriptor_rooted": True,
        "non_symlink": True,
        "read_only_mount_required": True,
        "read_only_mode": checked["mode_read_only"],
        "read_only_filesystem": checked["filesystem_read_only"],
        "runtime_manifest_bound": "runtime_manifest_sha256" in admission,
    }
