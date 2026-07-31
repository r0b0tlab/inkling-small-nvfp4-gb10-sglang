from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import pytest

from inkling_release.image import validate_derivative_metadata, validate_image_metadata
from inkling_release.manifest import IMAGE_ARM64_MANIFEST, IMAGE_CONFIG, MODEL_REVISION, SOURCE_COMMIT, SOURCE_TREE, TORCHCODEC_VERSION, TORCHCODEC_WHEEL_SHA256, TORCHCODEC_WHEEL_URL

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_json_schemas_are_valid_draft_2020_12() -> None:
    schemas = sorted((ROOT / "schemas").glob("*.schema.json"))
    assert schemas
    for path in schemas:
        Draft202012Validator.check_schema(_load(path))


def test_runtime_manifest_and_profiles_validate_against_schemas() -> None:
    Draft202012Validator(_load(ROOT / "schemas" / "runtime-manifest.schema.json")).validate(_load(ROOT / "runtime-manifest.json"))
    validator = Draft202012Validator(_load(ROOT / "schemas" / "profile.schema.json"))
    for path in sorted((ROOT / "profiles").glob("*.json")):
        validator.validate(_load(path))


def test_schema_rejects_runtime_drift_and_profile_extra_fields() -> None:
    schema = _load(ROOT / "schemas" / "runtime-manifest.schema.json")
    changed = deepcopy(_load(ROOT / "runtime-manifest.json"))
    changed["source"]["commit"] = "0" * 40
    assert list(Draft202012Validator(schema).iter_errors(changed))
    p_schema = _load(ROOT / "schemas" / "profile.schema.json")
    profile = _load(ROOT / "profiles" / "inkling-small-nvfp4.json")
    profile["extra"] = True
    assert list(Draft202012Validator(p_schema).iter_errors(profile))


def test_image_metadata_contracts() -> None:
    metadata = {"repository": "lmsysorg/sglang", "image_index": "sha256:" + "1" * 64, "linux_arm64_manifest": IMAGE_ARM64_MANIFEST, "config": IMAGE_CONFIG, "source_commit_label": SOURCE_COMMIT}
    assert validate_image_metadata(metadata) == metadata
    derivative = {
        "image_ref": "example.invalid/inkling@sha256:" + "2" * 64,
        "base_image_manifest": IMAGE_ARM64_MANIFEST,
        "base_image_config": IMAGE_CONFIG,
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "model_revision": MODEL_REVISION,
        "torchcodec_version": TORCHCODEC_VERSION,
        "torchcodec_wheel_url": TORCHCODEC_WHEEL_URL,
        "torchcodec_wheel_sha256": TORCHCODEC_WHEEL_SHA256,
    }
    assert validate_derivative_metadata(derivative) == derivative
    derivative["model_revision"] = "0" * 40
    with pytest.raises(ValueError):
        validate_derivative_metadata(derivative)


def test_benchmark_schema_requires_exact_eligible_cardinality() -> None:
    schema = _load(ROOT / "schemas" / "benchmark-artifact.schema.json")
    value = {"schema_version": 1, "status": "ELIGIBLE", "runtime_manifest_sha256": "a" * 64, "rows": [], "infrastructure_failures": 0}
    assert list(Draft202012Validator(schema).iter_errors(value))