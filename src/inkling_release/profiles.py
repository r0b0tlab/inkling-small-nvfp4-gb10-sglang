from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REQUIRED = {
    "schema_version", "name", "description", "model_path", "server_executable", "base_args",
    "defaults", "mtp", "mem_fraction_static", "runtime_mode",
}


def load_profile(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) > 256 * 1024:
        raise ValueError("profile exceeds size limit")
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != _REQUIRED:
        raise ValueError("profile keys do not match schema")
    if not isinstance(value["name"], str) or not value["name"]:
        raise ValueError("profile name is required")
    if value["runtime_mode"] != "native-cuda13-arm64-sm121":
        raise ValueError("profile must target native CUDA 13 arm64 SM121")
    if value["model_path"] != "/models/Inkling-Small-NVFP4":
        raise ValueError("profile model path is frozen")
    if value["server_executable"] != ["sglang", "serve"]:
        raise ValueError("profile executable must match the inspected sglang serve CLI")
    if not isinstance(value["base_args"], list) or any(not isinstance(x, str) for x in value["base_args"]):
        raise ValueError("base_args must be a string array")
    critical_pairs = {
        "--model-path": "/models/Inkling-Small-NVFP4",
        "--model-type": "llm",
        "--model-impl": "sglang",
        "--tp": "2",
        "--nnodes": "2",
        "--quantization": "modelopt_fp4",
        "--attention-backend": "triton",
        "--page-size": "128",
        "--fp4-gemm-backend": "flashinfer_trtllm",
        "--moe-runner-backend": "flashinfer_trtllm_routed",
        "--random-seed": "0",
        "--cuda-graph-backend-decode": "disabled",
        "--cuda-graph-backend-prefill": "disabled",
    }
    expected_kv_flag = "mxfp8" if value["name"].endswith("mxfp8-candidate") else "bfloat16"
    critical_pairs["--kv-cache-dtype"] = expected_kv_flag
    for flag, expected in critical_pairs.items():
        if value["base_args"].count(flag) != 1:
            raise ValueError(f"critical option must occur exactly once: {flag}")
        index = value["base_args"].index(flag)
        if index + 1 >= len(value["base_args"]) or value["base_args"][index + 1] != expected:
            raise ValueError(f"critical option has unexpected value: {flag}")
    if "--enable-torch-symm-mem" in value["base_args"]:
        raise ValueError("torch symmetric memory is unsupported on SM121")
    defaults = value["defaults"]
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be an object")
    if defaults.get("context_length") != 32768 or defaults.get("concurrency") != 1:
        raise ValueError("conservative default profile is required")
    expected_kv = "mxfp8" if value["name"].endswith("mxfp8-candidate") else "bf16"
    if defaults.get("kv_dtype") != expected_kv or defaults.get("chunk_size") != 1024:
        raise ValueError("conservative KV/chunk defaults are required")
    if defaults.get("graph_max_decode_batch") != 1:
        raise ValueError("conservative graph batch default is required")
    mtp = value["mtp"]
    if not isinstance(mtp, dict) or set(mtp) != {"layers", "draft_layers", "total_layers"}:
        raise ValueError("mtp must contain the exact layer contract")
    is_mtp = value["name"].endswith("mtp")
    if is_mtp and mtp != {"layers": 8, "draft_layers": 1, "total_layers": 9}:
        raise ValueError("MTP profile must be exact 8-1-9")
    if is_mtp is not ("--enable-multi-layer-eagle" in value["base_args"]):
        raise ValueError("multi-layer EAGLE flag must match the MTP profile")
    if "--model" in value["base_args"]:
        raise ValueError("legacy --model option is forbidden; use --model-path")
    return value


def list_profiles(directory: Path) -> list[dict[str, Any]]:
    profiles = [load_profile(path) for path in sorted(directory.glob("*.json"))]
    if not profiles:
        raise ValueError("no profiles found")
    names = [item["name"] for item in profiles]
    if len(names) != len(set(names)):
        raise ValueError("duplicate profile names")
    return profiles
