from __future__ import annotations

import re
from typing import Any

from .manifest import (
    IMAGE_ARM64_MANIFEST,
    IMAGE_CONFIG,
    MODEL_REVISION,
    SOURCE_COMMIT,
    SOURCE_TREE,
    TORCHCODEC_VERSION,
    TORCHCODEC_WHEEL_SHA256,
    TORCHCODEC_WHEEL_URL,
)

_DIGEST = re.compile(r"^([A-Za-z0-9][A-Za-z0-9_.:/-]{0,254})@sha256:[0-9a-f]{64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_image_ref(value: Any) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError("image reference must be digest pinned")
    return value


def validate_image_metadata(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("image metadata must be an object")
    required = {"repository", "image_index", "linux_arm64_manifest", "config", "source_commit_label"}
    if set(value) != required:
        raise ValueError("image metadata keys do not match schema")
    if not isinstance(value["repository"], str) or not value["repository"]:
        raise ValueError("image repository is required")
    for key in ("image_index", "linux_arm64_manifest", "config"):
        if not isinstance(value[key], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value[key]):
            raise ValueError(f"{key} must be a digest")
    if value["source_commit_label"] != SOURCE_COMMIT:
        raise ValueError("Docker source commit label does not match")
    return value


def audit_image_metadata(
    metadata: dict[str, Any], *, expected_arm64_manifest: str, expected_config: str, expected_source_commit: str
) -> dict[str, Any]:
    checked = validate_image_metadata(metadata)
    if checked["linux_arm64_manifest"] != expected_arm64_manifest:
        raise ValueError("arm64 manifest digest mismatch")
    if checked["config"] != expected_config:
        raise ValueError("config digest mismatch")
    if checked["source_commit_label"] != expected_source_commit:
        raise ValueError("source commit label mismatch")
    return {"status": "PASS", "image": checked["repository"], "manifest": checked["linux_arm64_manifest"]}


def validate_derivative_metadata(value: Any) -> dict[str, Any]:
    """Validate the labels/inputs that make a derivative image reusable."""
    if not isinstance(value, dict):
        raise ValueError("derivative image metadata must be an object")
    required = {
        "image_ref", "base_image_manifest", "base_image_config", "source_commit",
        "source_tree", "model_revision", "torchcodec_version", "torchcodec_wheel_url",
        "torchcodec_wheel_sha256",
    }
    if set(value) != required:
        raise ValueError("derivative image metadata keys do not match schema")
    validate_image_ref(value["image_ref"])
    if value["base_image_manifest"] != IMAGE_ARM64_MANIFEST or value["base_image_config"] != IMAGE_CONFIG:
        raise ValueError("derivative base image identity is not frozen")
    if value["source_commit"] != SOURCE_COMMIT or value["source_tree"] != SOURCE_TREE:
        raise ValueError("derivative source identity is not frozen")
    if value["model_revision"] != MODEL_REVISION:
        raise ValueError("derivative model revision is not frozen")
    if value["torchcodec_version"] != TORCHCODEC_VERSION or value["torchcodec_wheel_url"] != TORCHCODEC_WHEEL_URL:
        raise ValueError("derivative TorchCodec input is not frozen")
    if value["torchcodec_wheel_sha256"] != TORCHCODEC_WHEEL_SHA256 or not _SHA256.fullmatch(value["torchcodec_wheel_sha256"]):
        raise ValueError("derivative TorchCodec wheel hash is not frozen")
    return value
