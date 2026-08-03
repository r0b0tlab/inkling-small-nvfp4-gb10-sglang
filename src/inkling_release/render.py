from __future__ import annotations

import ipaddress
import re
from pathlib import PurePosixPath
from typing import Any, Iterable

_MODEL_TARGET = "/models/Inkling-Small-NVFP4"
_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,252}$")
_DIGEST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,254}@sha256:[0-9a-f]{64}$")


def _safe_model_root(value: str) -> str:
    if not isinstance(value, str) or value != _MODEL_TARGET:
        raise ValueError("model root must be the exact offline mount path")
    path = PurePosixPath(value)
    if not path.is_absolute() or any(part in (".", "..") for part in path.parts):
        raise ValueError("model root must be absolute and normalized")
    return value


def _safe_addr(value: str) -> str:
    if not isinstance(value, str) or ":" not in value or "/" in value or "@" in value or any(c.isspace() for c in value):
        raise ValueError("dist-init-addr must be host:port without URL or userinfo")
    host, port_text = value.rsplit(":", 1)
    if not host or not port_text.isdigit() or not 1 <= int(port_text) <= 65535:
        raise ValueError("dist-init-addr has invalid port")
    if host.startswith("[") and host.endswith("]"):
        try:
            ipaddress.IPv6Address(host[1:-1])
        except ValueError as exc:
            raise ValueError("dist-init-addr IPv6 host is invalid") from exc
    elif not _HOST_RE.fullmatch(host):
        try:
            ipaddress.ip_address(host)
        except ValueError as exc:
            raise ValueError("dist-init-addr host is invalid") from exc
    return value


def _validate_argv_tokens(tokens: Iterable[Any], *, label: str) -> list[str]:
    if not isinstance(tokens, list):
        raise ValueError(f"{label} must be a string array")
    checked: list[str] = []
    for token in tokens:
        if not isinstance(token, str) or not token or "\x00" in token:
            raise ValueError(f"{label} contains an invalid token")
        checked.append(token)
    return checked


def _replace_model_path(argv: list[str], model_root: str) -> list[str]:
    """Replace the value paired with --model-path, never the flag itself."""
    result: list[str] = []
    occurrences = 0
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--model":
            raise ValueError("--model is forbidden; official cookbook uses --model-path")
        if token == "--model-path":
            occurrences += 1
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                raise ValueError("--model-path must be followed by a value")
            result.extend((token, model_root))
            index += 2
            continue
        if token.startswith("--model="):
            raise ValueError("--model is forbidden; official cookbook uses --model-path")
        if token.startswith("--model-path="):
            occurrences += 1
            result.extend(("--model-path", model_root))
            index += 1
            continue
        result.append(token)
        index += 1
    if occurrences > 1:
        raise ValueError("profile contains duplicate --model-path options")
    if occurrences == 0:
        result.extend(("--model-path", model_root))
    return result


def _add_pair(argv: list[str], flag: str, value: Any) -> None:
    argv.extend([flag, str(value)])


def render_command(
    profile: dict[str, Any],
    *,
    node_rank: int,
    dist_init_addr: str,
    model_root: str = _MODEL_TARGET,
    image_ref: str | None = None,
) -> dict[str, Any]:
    """Render an argv/env description only; this function never executes anything."""
    if type(node_rank) is not int or node_rank not in (0, 1):
        raise ValueError("node_rank must be 0 or 1")
    model_root = _safe_model_root(model_root)
    dist_init_addr = _safe_addr(dist_init_addr)
    required = {"name", "server_executable", "base_args", "model_path", "defaults", "mtp"}
    if not isinstance(profile, dict) or not required <= set(profile):
        raise ValueError("profile is incomplete")
    if profile["model_path"] != _MODEL_TARGET:
        raise ValueError("profile model path is frozen")
    executable = _validate_argv_tokens(profile["server_executable"], label="server_executable")
    base_args = _validate_argv_tokens(profile["base_args"], label="base_args")
    argv = _replace_model_path(executable + base_args, model_root)
    _add_pair(argv, "--node-rank", node_rank)
    _add_pair(argv, "--dist-init-addr", dist_init_addr)
    defaults = profile["defaults"]
    if not isinstance(defaults, dict):
        raise ValueError("profile defaults are invalid")
    for key in ("context_length", "concurrency", "chunk_size"):
        if key not in defaults:
            raise ValueError(f"profile default is missing: {key}")
    _add_pair(argv, "--context-length", defaults["context_length"])
    _add_pair(argv, "--max-running-requests", defaults["concurrency"])
    _add_pair(argv, "--chunked-prefill-size", defaults["chunk_size"])
    kv_dtype = defaults.get("kv_dtype")
    if kv_dtype == "bf16":
        kv_dtype = "bfloat16"
    if kv_dtype not in {"bfloat16", "mxfp8"}:
        raise ValueError("profile KV dtype is not an admitted candidate")
    if "--kv-cache-dtype" not in argv:
        _add_pair(argv, "--kv-cache-dtype", kv_dtype)
    _add_pair(argv, "--cuda-graph-max-bs-decode", defaults.get("graph_max_decode_batch"))
    _add_pair(argv, "--mem-fraction-static", profile.get("mem_fraction_static", 0.85))
    env = {
        "FLASHINFER_NVCC_THREADS": "2",
        "FLASHINFER_DISABLE_JIT": "1",
        "FLASHINFER_WORKSPACE_BASE": "/tmp/flashinfer",
        "HF_HUB_OFFLINE": "1",
        "MAX_JOBS": "6",
        "NVCC_THREADS": "2",
        "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
        "SGLANG_MODEL_ROOT": model_root,
        "SGLANG_OPT_USE_INKLING_SHEARED_BIAS": "0",
        "TRANSFORMERS_OFFLINE": "1",
    }
    if node_rank == 0:
        env.update({
            "NCCL_SOCKET_IFNAME": "=enp1s0f0np0",
            "NCCL_IB_HCA": "=rocep1s0f0",
        })
    else:
        env.update({
            "NCCL_SOCKET_IFNAME": "=enp1s0f1np1",
            "NCCL_IB_HCA": "=rocep1s0f1",
        })
    env.update({
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_CROSS_NIC": "0",
        "NCCL_NET": "IB",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,NET",
    })
    result: dict[str, Any] = {
        "argv": argv,
        "env": env,
        "node_rank": node_rank,
        "dist_init_addr": dist_init_addr,
        "model_root": model_root,
        "profile": profile["name"],
        "executes": False,
    }
    if image_ref is not None:
        if not isinstance(image_ref, str) or not _DIGEST_RE.fullmatch(image_ref):
            raise ValueError("image_ref must be a repository reference pinned by sha256")
        result["image"] = image_ref
    return result
