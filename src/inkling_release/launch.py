from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
from typing import Any, Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

from .image import validate_image_ref
from .manifest import (
    BASE_IMAGE_REF,
    IMAGE_ARM64_MANIFEST,
    IMAGE_CONFIG,
    RUNTIME_GID,
    RUNTIME_UID,
    load_runtime_manifest,
    manifest_sha256,
)
from .preflight import run_preflight
from .profiles import load_profile
from .render import render_command

OWNED_CONTAINER_RE = re.compile(r"^inkling-sglang-tp2-rank[01]$")
_ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_SAFE_PASSTHROUGH_PREFIXES = ("CUDA_", "NCCL_", "GLOO_", "HELION_", "SGLANG_", "TORCH_", "TRITON_", "HF_")
_TMPFS_MOUNTS = (
    "type=tmpfs,destination=/tmp,tmpfs-mode=1777,tmpfs-size=4294967296",
)
_RDMA_DEVICES = (
    "/dev/infiniband/rdma_cm",
    "/dev/infiniband/uverbs0",
    "/dev/infiniband/uverbs1",
    "/dev/infiniband/uverbs2",
    "/dev/infiniband/uverbs3",
)


def owned_container_name(node_rank: int) -> str:
    if type(node_rank) is not int or node_rank not in (0, 1):
        raise ValueError("node_rank must be 0 or 1")
    return f"inkling-sglang-tp2-rank{node_rank}"


def _safe_source_path(path: Path) -> str:
    raw = os.fspath(path)
    supplied = PurePosixPath(raw)
    if not supplied.is_absolute() or any(part in ("", ".", "..") for part in supplied.parts):
        raise ValueError("model root must be an absolute normalized path")
    if "\x00" in raw or "," in raw:
        raise ValueError("model root contains an unsafe mount character")
    info = os.lstat(path)
    if not stat_is_directory(info.st_mode) or stat_is_symlink(info.st_mode) or info.st_nlink < 1:
        raise ValueError("model root must be a non-symlink directory")
    return supplied.as_posix()


def _validate_runtime_identity(runtime_uid: int, runtime_gid: int) -> None:
    if type(runtime_uid) is not int or not 0 <= runtime_uid <= 65535:
        raise ValueError("runtime_uid must be an integer in the numeric UID range")
    if type(runtime_gid) is not int or not 0 <= runtime_gid <= 65535:
        raise ValueError("runtime_gid must be an integer in the numeric GID range")


def _safe_cache_path(path: Path, *, runtime_uid: int, runtime_gid: int) -> str:
    raw = os.fspath(path)
    supplied = PurePosixPath(raw)
    if not supplied.is_absolute() or any(part in ("", ".", "..") for part in supplied.parts):
        raise ValueError("cache root must be an absolute normalized path")
    if "\x00" in raw or "," in raw:
        raise ValueError("cache root contains an unsafe mount character")
    info = os.lstat(path)
    if not stat_is_directory(info.st_mode) or stat_is_symlink(info.st_mode) or info.st_nlink < 1:
        raise ValueError("cache root must be a non-symlink directory")
    _validate_runtime_identity(runtime_uid, runtime_gid)
    if info.st_uid != runtime_uid or info.st_gid != runtime_gid:
        raise ValueError("cache root must be owned by the declared runtime UID/GID")
    if info.st_mode & 0o700 != 0o700:
        raise ValueError("cache root must be readable, writable, and searchable by the runtime user")
    if info.st_mode & 0o022:
        raise ValueError("cache root must not be group/world writable")
    flags = os.statvfs(path).f_flag
    if flags & getattr(os, "ST_NOEXEC", 8):
        raise ValueError("cache root filesystem must permit native/JIT execution")
    return supplied.as_posix()


def stat_is_directory(mode: int) -> bool:
    return (mode & 0o170000) == 0o040000


def stat_is_symlink(mode: int) -> bool:
    return (mode & 0o170000) == 0o120000


def _validate_pass_through(env_passthrough: Mapping[str, str] | Sequence[str] | None) -> list[str]:
    if env_passthrough is None:
        return []
    if isinstance(env_passthrough, Mapping):
        items: list[tuple[str, str | None]] = [(key, value) for key, value in env_passthrough.items()]
    else:
        items = [(key, None) for key in env_passthrough]
    rendered: list[str] = []
    for key, value in sorted(items):
        if not isinstance(key, str) or not _ENV_NAME_RE.fullmatch(key):
            raise ValueError("environment pass-through name is invalid")
        if not key.startswith(_SAFE_PASSTHROUGH_PREFIXES):
            raise ValueError("environment pass-through is restricted to runtime-safe prefixes")
        if any(secret in key for secret in ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL")):
            raise ValueError("credential-bearing environment pass-through is forbidden")
        if value is None:
            rendered.append(key)
        else:
            if not isinstance(value, str) or "\x00" in value or "\n" in value:
                raise ValueError("environment pass-through value is invalid")
            rendered.append(f"{key}={value}")
    return rendered


def build_docker_argv(
    *,
    image_ref: str,
    rendered: dict[str, Any],
    model_root: Path,
    cache_root: Path,
    node_rank: int,
    manifest_sha256_value: str | None = None,
    env_passthrough: Mapping[str, str] | Sequence[str] | None = None,
    runtime_uid: int = RUNTIME_UID,
    runtime_gid: int = RUNTIME_GID,
) -> list[str]:
    validate_image_ref(image_ref)
    _validate_runtime_identity(runtime_uid, runtime_gid)
    name = owned_container_name(node_rank)
    source = _safe_source_path(model_root)
    cache = _safe_cache_path(
        cache_root,
        runtime_uid=runtime_uid,
        runtime_gid=runtime_gid,
    )
    if not isinstance(rendered, dict) or rendered.get("executes") is not False:
        raise ValueError("launcher accepts only a render-only command description")
    argv = [
        "docker", "run", "--rm", "--pull", "never", "--init", "--name", name,
        "--label", "inkling.release=sglang",
        "--label", f"inkling.rank={node_rank}",
        "--user", f"{runtime_uid}:{runtime_gid}",
        "--network", "host",
        "--ipc", "host",
        "--gpus", "all",
        "--cap-drop", "ALL",
        "--cap-add", "IPC_LOCK",
        "--security-opt", "no-new-privileges:true",
        "--ulimit", "memlock=-1:-1",
        "--read-only",
        "--mount", f"type=bind,source={source},target=/models/Inkling-Small-NVFP4,readonly",
    ]
    for device in _RDMA_DEVICES:
        argv.extend(("--device", device))
    for mount in _TMPFS_MOUNTS:
        argv.extend(("--mount", mount))
    if manifest_sha256_value is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256_value):
            raise ValueError("manifest_sha256_value must be lowercase SHA-256")
        argv.extend(("--label", f"inkling.manifest_sha256={manifest_sha256_value}"))
    env = dict(rendered.get("env", {}))
    env.update({
        "TMPDIR": "/tmp",
        "HOME": "/cache/user-cache",
        "USER": "inkling",
        "LOGNAME": "inkling",
        "HF_HOME": "/cache/hf",
        "TORCH_HOME": "/cache/torch",
        "TORCHINDUCTOR_CACHE_DIR": "/cache/torchinductor",
        "TRITON_CACHE_DIR": "/cache/triton",
        "TVM_FFI_CACHE_DIR": "/cache/user-cache/.cache/tvm-ffi",
        "CUDA_CACHE_PATH": "/cache/cuda",
    })
    for key, value in sorted(env.items()):
        if not isinstance(key, str) or not _ENV_NAME_RE.fullmatch(key) or not isinstance(value, str) or "\x00" in value:
            raise ValueError("rendered environment contains an invalid entry")
        argv.extend(("--env", f"{key}={value}"))
    for item in _validate_pass_through(env_passthrough):
        argv.extend(("--env", item))
    argv.extend((image_ref, *rendered["argv"]))
    if "--model" in argv:
        raise ValueError("launcher must not emit --model")
    return argv


def build_launch_spec(
    *,
    profile_path: Path,
    manifest_path: Path,
    admission_path: Path,
    model_root: Path,
    cache_root: Path,
    node_rank: int,
    dist_init_addr: str,
    image_ref: str | None = None,
    env_passthrough: Mapping[str, str] | Sequence[str] | None = None,
) -> dict[str, Any]:
    manifest = load_runtime_manifest(manifest_path)
    manifest_digest = manifest_sha256(manifest)
    preflight = run_preflight(
        admission_path,
        model_root,
        expected_runtime_manifest_sha256=manifest_digest,
        require_runtime_binding=True,
    )
    profile = load_profile(profile_path)
    selected_image = image_ref or BASE_IMAGE_REF
    validate_image_ref(selected_image)
    runtime_uid = manifest["recipe"]["runtime_uid"]
    runtime_gid = manifest["recipe"]["runtime_gid"]
    rendered = render_command(profile, node_rank=node_rank, dist_init_addr=dist_init_addr, image_ref=selected_image)
    docker_argv = build_docker_argv(
        image_ref=selected_image,
        rendered=rendered,
        model_root=model_root,
        cache_root=cache_root,
        node_rank=node_rank,
        manifest_sha256_value=manifest_digest,
        env_passthrough=env_passthrough,
        runtime_uid=runtime_uid,
        runtime_gid=runtime_gid,
    )
    image_kind = "base" if selected_image == BASE_IMAGE_REF else "derivative"
    return {
        "schema_version": 1,
        "status": "READY",
        "manifest_sha256": manifest_digest,
        "source_commit": manifest["source"]["commit"],
        "image_ref": selected_image,
        "image_kind": image_kind,
        "image_manifest": IMAGE_ARM64_MANIFEST,
        "image_config": IMAGE_CONFIG,
        "preflight": preflight,
        "container_name": owned_container_name(node_rank),
        "argv": docker_argv,
        "node_rank": node_rank,
        "runtime_uid": runtime_uid,
        "runtime_gid": runtime_gid,
        "execute": False,
    }


def _safe_log_path(path: Path) -> None:
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise ValueError(f"log path is not a regular file: {path}")


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists() and path.is_symlink():
        raise ValueError("receipt path may not be a symlink")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(path)


def write_receipt(path: Path, spec: dict[str, Any], *, pid: int, container_name: str, status: str = "STARTED", **extra: Any) -> None:
    if not OWNED_CONTAINER_RE.fullmatch(container_name):
        raise ValueError("receipt container is not owned")
    receipt = {
        "schema_version": 1,
        "status": status,
        "pid": pid,
        "container_name": container_name,
        "manifest_sha256": spec["manifest_sha256"],
        "source_commit": spec["source_commit"],
        "image_ref": spec.get("image_ref"),
        "argv": spec["argv"],
    }
    receipt.update(extra)
    _atomic_write_json(path, receipt)


def _tail(path: Path, maximum: int = 8192) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-maximum:].decode("utf-8", errors="replace")


def _validate_readiness_url(url: str) -> str:
    if not isinstance(url, str) or not (url.startswith("http://127.0.0.1:") or url.startswith("http://[::1]:")):
        raise ValueError("readiness URL must use a loopback address")
    if any(char in url for char in ("@", "?", "#")):
        raise ValueError("readiness URL may not contain credentials, query, or fragment")
    return url


def _probe_readiness(url: str, timeout_seconds: float) -> bool:
    request = Request(url, method="GET")
    try:
        with urlopen(request, timeout=min(1.0, timeout_seconds)) as response:
            return 200 <= response.status < 300
    except (OSError, URLError):
        return False


def launch(
    spec: dict[str, Any],
    *,
    receipt_path: Path,
    execute: bool = False,
    readiness_url: str | None = None,
    readiness_timeout: float = 30.0,
) -> dict[str, Any]:
    """Start only an explicitly admitted derivative image and retain both rank logs."""
    if not execute:
        return {**spec, "execute": False, "status": "DRY_RUN"}
    if spec.get("image_kind") != "derivative" or not spec.get("image_ref"):
        raise ValueError("live launch requires an explicitly supplied frozen derivative image digest")
    argv = spec.get("argv")
    if not isinstance(argv, list) or argv[:2] != ["docker", "run"]:
        raise ValueError("invalid docker launch specification")
    if not OWNED_CONTAINER_RE.fullmatch(str(spec.get("container_name", ""))):
        raise ValueError("refusing to launch an unowned container name")
    if readiness_timeout <= 0 or readiness_timeout > 300:
        raise ValueError("readiness timeout is outside the bounded range")
    readiness_url = _validate_readiness_url(readiness_url) if readiness_url is not None else None
    rank = spec.get("node_rank", 0)
    stem = receipt_path.stem
    stdout_path = receipt_path.with_name(f"{stem}.rank{rank}.stdout.log")
    stderr_path = receipt_path.with_name(f"{stem}.rank{rank}.stderr.log")
    _safe_log_path(stdout_path)
    _safe_log_path(stderr_path)
    stdout_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        try:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                close_fds=True,
                start_new_session=True,
            )
        except OSError as exc:
            write_receipt(
                receipt_path, spec, pid=-1, container_name=spec["container_name"], status="FAILED",
                error=str(exc), stdout_path=str(stdout_path), stderr_path=str(stderr_path),
            )
            raise
        write_receipt(
            receipt_path, spec, pid=process.pid, container_name=spec["container_name"], status="STARTED",
            stdout_path=str(stdout_path), stderr_path=str(stderr_path), readiness_url=readiness_url,
        )
    if readiness_url is None:
        return {**spec, "execute": True, "status": "STARTED", "pid": process.pid, "stdout_path": str(stdout_path), "stderr_path": str(stderr_path)}
    deadline = time.monotonic() + readiness_timeout
    ready = False
    while time.monotonic() < deadline:
        if process.poll() is not None:
            error = _tail(stderr_path) or _tail(stdout_path)
            write_receipt(
                receipt_path, spec, pid=process.pid, container_name=spec["container_name"], status="FAILED",
                returncode=process.returncode, error=error, stdout_path=str(stdout_path), stderr_path=str(stderr_path),
                readiness_url=readiness_url,
            )
            return {**spec, "execute": True, "status": "FAILED", "pid": process.pid, "returncode": process.returncode, "error": error}
        if _probe_readiness(readiness_url, deadline - time.monotonic()):
            ready = True
            break
        time.sleep(0.2)
    status = "READY" if ready else "STARTED"
    write_receipt(
        receipt_path, spec, pid=process.pid, container_name=spec["container_name"], status=status,
        stdout_path=str(stdout_path), stderr_path=str(stderr_path), readiness_url=readiness_url,
        readiness="PASS" if ready else "TIMEOUT", error=None if ready else _tail(stderr_path),
    )
    return {**spec, "execute": True, "status": status, "pid": process.pid, "stdout_path": str(stdout_path), "stderr_path": str(stderr_path), "readiness": "PASS" if ready else "TIMEOUT"}
