from __future__ import annotations

import re
from typing import Any


class EvidenceError(ValueError):
    """Evidence is malformed, stale, or claims an unsupported result."""


_ALLOWED = {"NO_VERDICT", "NOT_RUN", "DIAGNOSTIC", "DISQUALIFIED"}
_SHA = re.compile(r"^[0-9a-f]{64}$")


def validate_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceError("evidence must be an object")
    required = {"schema_version", "status", "claim", "runtime_manifest_sha256", "rows"}
    if set(value) != required:
        raise EvidenceError("evidence keys do not match schema")
    if value["schema_version"] != 1 or type(value["schema_version"]) is not int:
        raise EvidenceError("unsupported evidence schema")
    if value["status"] not in _ALLOWED:
        raise EvidenceError("PASS/VERDICT evidence is not admissible in the foundation")
    if not isinstance(value["claim"], str) or not value["claim"]:
        raise EvidenceError("claim must be non-empty")
    if not isinstance(value["runtime_manifest_sha256"], str) or not _SHA.fullmatch(value["runtime_manifest_sha256"]):
        raise EvidenceError("runtime_manifest_sha256 must be lowercase SHA-256")
    if not isinstance(value["rows"], list):
        raise EvidenceError("rows must be an array")
    return value


def accept_evidence(value: Any) -> dict[str, Any]:
    checked = validate_evidence(value)
    # The non-live package can record posture only; runtime claims require a later live gate.
    if checked["status"] not in {"NO_VERDICT", "NOT_RUN", "DIAGNOSTIC", "DISQUALIFIED"}:
        raise EvidenceError("evidence status is not accepted")
    return {**checked, "accepted": True}
