from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MODEL_ID = "thinkingmachines/Inkling-Small-NVFP4"
MODEL_REVISION = "b6a99534467840620d411e4cd4ad5819b2610d9c"
SOURCE_REPOSITORY = "sgl-project/sglang"
SOURCE_COMMIT = "b7252cc6b0c78b25ecea7ee5efa91a6ae37d0f19"
SOURCE_TREE = "b8c966b81dcce80824261ccae7aa9d33441935a7"
IMAGE_REPOSITORY = "lmsysorg/sglang"
IMAGE_INDEX = "sha256:fbea1a4e25b26660dbc2384a27ead8817e9b7670f257b5c3143e0450d14524d7"
IMAGE_ARM64_MANIFEST = "sha256:c60f221f8f42929469bedf74716b4314a1951ff97556dd9e17d9e11040512ac6"
IMAGE_CONFIG = "sha256:a2364fcb06508b66f464ecd16921144619bdac9aa883391bcdec95be2b632293"
BASE_IMAGE_REF = f"{IMAGE_REPOSITORY}@{IMAGE_ARM64_MANIFEST}"
TORCHCODEC_VERSION = "0.12.0+cu130"
TORCHCODEC_WHEEL_URL = "https://download.pytorch.org/whl/cu130/torchcodec-0.12.0%2Bcu130-cp312-cp312-manylinux_2_28_aarch64.whl"
TORCHCODEC_WHEEL_SHA256 = "7293d3bbf3e27621f7d5888cc10e233d828faa5896f76f4d3f3ab1457e9c8c9e"
RUNTIME_UID = 1001
RUNTIME_GID = 1001

_REQUIRED_TOP = {"schema_version", "model", "source", "image", "recipe", "policies"}
_HEX = frozenset("0123456789abcdef")
_SHA256 = frozenset("0123456789abcdef")


def _strict_load(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) > 1024 * 1024:
        raise ValueError("runtime manifest exceeds size limit")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    try:
        loaded = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid runtime manifest JSON: {path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("runtime manifest must be an object")
    return loaded


def _sha256(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _SHA256 for char in value):
        raise ValueError("expected lowercase SHA-256")
    return value


def validate_runtime_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _REQUIRED_TOP:
        raise ValueError("runtime manifest keys do not match schema")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("schema_version must be integer 1")
    model = value["model"]
    source = value["source"]
    image = value["image"]
    recipe = value["recipe"]
    policies = value["policies"]
    expected_model = {
        "repo_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "manifest_sha256": "8b46f4b3c1d47a31341acee60b42d3340d5937744b90752605d177481a9470e4",
        "file_count": 21,
        "payload_bytes": 170764923366,
    }
    if not isinstance(model, dict) or model != expected_model:
        raise ValueError("model identity is not the frozen contract")
    if not isinstance(source, dict) or set(source) != {"repository", "commit", "tree"}:
        raise ValueError("invalid source identity")
    if source != {"repository": SOURCE_REPOSITORY, "commit": SOURCE_COMMIT, "tree": SOURCE_TREE}:
        raise ValueError("source identity is not the frozen contract")
    expected_image = {
        "repository": IMAGE_REPOSITORY,
        "image_index": IMAGE_INDEX,
        "linux_arm64_manifest": IMAGE_ARM64_MANIFEST,
        "config": IMAGE_CONFIG,
        "torchcodec_version": TORCHCODEC_VERSION,
        "torchcodec_wheel_sha256": TORCHCODEC_WHEEL_SHA256,
        "candidate_derivative_config": "sha256:0303cae18c7a8d1526ba736eca81775a735e3e45fca49b3eea977eceb2e10c68",
        "aot_cache_manifest_sha256": "32ac33e97af24f0fe50175272eafc5c66bcb3615d8c9778a23a5a50f57952980",
        "aot_cache_file_count": 809,
        "aot_cache_total_bytes": 354049685,
    }
    if not isinstance(image, dict) or image != expected_image:
        raise ValueError("image identity is not the frozen candidate contract")
    if not isinstance(recipe, dict):
        raise ValueError("recipe must be an object")
    expected_flags = {
        "trust_remote_code": True,
        "model_type": "llm",
        "model_path": "/models/Inkling-Small-NVFP4",
        "model_impl": "sglang",
        "runtime_uid": RUNTIME_UID,
        "runtime_gid": RUNTIME_GID,
        "tp": 2,
        "nnodes": 2,
        "quantization": "modelopt_fp4",
        "kv_cache_dtype": "bf16",
        "attention_backend": "triton",
        "page_size": 128,
        "fp4_gemm_backend": "flashinfer_trtllm",
        "moe_runner_backend": "flashinfer_cutlass",
        "disable_flashinfer_autotune": True,
        "enable_torch_symm_mem": False,
        "mamba_radix_cache_strategy": "extra_buffer",
        "swa_full_tokens_ratio": 0.1,
        "mamba_full_memory_ratio": 0.1,
        "enable_multimodal": True,
        "reasoning_parser": "inkling",
        "tool_call_parser": "inkling",
        "random_seed": 0,
        "context_length": 32768,
        "chunked_prefill_size": 1024,
        "max_running_requests": 1,
        "cuda_graph_backend_decode": "disabled",
        "cuda_graph_backend_prefill": "disabled",
    }
    if set(recipe) != set(expected_flags) | {"mem_fraction_static", "env"}:
        raise ValueError("recipe keys do not match the frozen contract")
    for key, expected in expected_flags.items():
        if recipe.get(key) != expected:
            raise ValueError(f"recipe flag {key} does not match frozen contract")
    if type(recipe.get("mem_fraction_static")) is not float or recipe["mem_fraction_static"] != 0.85:
        raise ValueError("initial mem_fraction_static must be 0.85")
    expected_env = {
        "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
        "SGLANG_OPT_USE_INKLING_SHEARED_BIAS": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MAX_JOBS": "6",
        "NVCC_THREADS": "2",
        "FLASHINFER_NVCC_THREADS": "2",
        "FLASHINFER_DISABLE_JIT": "1",
        "FLASHINFER_WORKSPACE_BASE": "/tmp/flashinfer",
        "HELION_AOT_AUTOTUNE": "none",
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_CROSS_NIC": "0",
        "NCCL_NET": "IB",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,NET",
        "TVM_FFI_CACHE_DIR": "/cache/user-cache/.cache/tvm-ffi",
    }
    if recipe.get("env") != expected_env:
        raise ValueError("runtime environment contract is missing")
    if not isinstance(policies, dict) or policies != {
        "native_only": True,
        "offline_model_mount": True,
        "renderer_executes": False,
        "fp4_kv_excluded": True,
        "verdict": "NO_VERDICT",
    }:
        raise ValueError("release policies are not fail-closed")
    return value


def canonical_manifest_bytes(value: dict[str, Any]) -> bytes:
    validate_runtime_manifest(value)
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def manifest_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_manifest_bytes(value)).hexdigest()


def _validate_lock(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"schema_version", "manifest_sha256"}:
        raise ValueError("runtime manifest lock keys do not match schema")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("runtime manifest lock schema_version must be integer 1")
    _sha256(value["manifest_sha256"])
    return value


def load_runtime_manifest(path: Path, *, require_lock: bool = True) -> dict[str, Any]:
    value = validate_runtime_manifest(_strict_load(path))
    lock = path.with_name("runtime-manifest.lock.json")
    if require_lock and not lock.exists():
        raise ValueError("runtime manifest lock is required")
    if lock.exists():
        locked = _validate_lock(_strict_load(lock))
        if locked["manifest_sha256"] != manifest_sha256(value):
            raise ValueError("runtime manifest lock digest mismatch")
    return value
