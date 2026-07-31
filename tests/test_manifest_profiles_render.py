from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from inkling_release.manifest import BASE_IMAGE_REF, MODEL_REVISION, SOURCE_COMMIT, load_runtime_manifest, manifest_sha256, validate_runtime_manifest
from inkling_release.profiles import list_profiles, load_profile
from inkling_release.render import render_command

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "runtime-manifest.json"


def test_frozen_runtime_manifest_and_lock() -> None:
    manifest = load_runtime_manifest(MANIFEST)
    assert manifest["model"]["revision"] == MODEL_REVISION
    assert manifest["source"]["commit"] == SOURCE_COMMIT
    lock = json.loads((ROOT / "runtime-manifest.lock.json").read_text())
    assert lock == {"schema_version": 1, "manifest_sha256": manifest_sha256(manifest)}


def test_manifest_rejects_identity_and_recipe_drift() -> None:
    manifest = load_runtime_manifest(MANIFEST)
    for section, key, value in (("model", "revision", "0" * 40), ("source", "commit", "0" * 40), ("recipe", "model_type", "auto"), ("recipe", "kv_cache_dtype", "fp4"), ("policies", "verdict", "PASS")):
        changed = deepcopy(manifest)
        changed[section][key] = value
        with pytest.raises(ValueError):
            validate_runtime_manifest(changed)


def test_manifest_lock_tampering_fails(tmp_path: Path) -> None:
    manifest_path = tmp_path / "runtime-manifest.json"
    lock_path = tmp_path / "runtime-manifest.lock.json"
    manifest_path.write_bytes(MANIFEST.read_bytes())
    lock_path.write_text(json.dumps({"schema_version": 1, "manifest_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="lock digest mismatch"):
        load_runtime_manifest(manifest_path)


def test_all_three_profiles_load_and_are_unique() -> None:
    profiles = list_profiles(ROOT / "profiles")
    assert {p["name"] for p in profiles} == {"inkling-small-nvfp4", "inkling-small-mtp", "inkling-small-mxfp8-candidate"}


def test_profiles_freeze_runtime_critical_pairs() -> None:
    for path in sorted((ROOT / "profiles").glob("*.json")):
        profile = load_profile(path)
        args = profile["base_args"]
        for flag in ("--model-path", "--model-type", "--model-impl", "--tp", "--nnodes", "--quantization", "--kv-cache-dtype", "--attention-backend", "--fp4-gemm-backend", "--moe-runner-backend", "--random-seed"):
            assert args.count(flag) == 1
        assert "--model" not in args
        assert "--enable-torch-symm-mem" not in args


def test_render_is_nonexecuting_and_replaces_only_model_value() -> None:
    profile = load_profile(ROOT / "profiles" / "inkling-small-nvfp4.json")
    result = render_command(profile, node_rank=1, dist_init_addr="192.0.2.1:25160", image_ref=BASE_IMAGE_REF)
    assert result["executes"] is False
    assert result["argv"][:2] == ["sglang", "serve"]
    assert result["argv"].count("--model-path") == 1
    index = result["argv"].index("--model-path")
    assert result["argv"][index + 1] == "/models/Inkling-Small-NVFP4"
    assert result["argv"][result["argv"].index("--node-rank") + 1] == "1"
    assert result["image"] == BASE_IMAGE_REF


def test_render_rejects_injection_and_unpinned_image() -> None:
    profile = load_profile(ROOT / "profiles" / "inkling-small-nvfp4.json")
    for bad_addr in ("host;id:1", "user@host:1", "http://host:1", "host:0"):
        with pytest.raises(ValueError):
            render_command(profile, node_rank=0, dist_init_addr=bad_addr)
    with pytest.raises(ValueError, match="pinned"):
        render_command(profile, node_rank=0, dist_init_addr="127.0.0.1:1", image_ref="repo:latest")


def test_render_rejects_duplicate_or_legacy_model_options() -> None:
    profile = load_profile(ROOT / "profiles" / "inkling-small-nvfp4.json")
    duplicate = deepcopy(profile)
    duplicate["base_args"] += ["--model-path", "/tmp/other"]
    with pytest.raises(ValueError, match="duplicate"):
        render_command(duplicate, node_rank=0, dist_init_addr="127.0.0.1:1")
    legacy = deepcopy(profile)
    legacy["base_args"] += ["--model", "bad"]
    with pytest.raises(ValueError, match="forbidden"):
        render_command(legacy, node_rank=0, dist_init_addr="127.0.0.1:1")