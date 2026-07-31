from __future__ import annotations

import re
from collections import Counter
from typing import Any

EXPECTED_ROW_COUNT = 1764
ALLOWED_STATUSES = {"NOT_RUN", "ELIGIBLE", "DIAGNOSTIC", "DISQUALIFIED"}
REQUIRED_KEYS = {
    "row_id", "status", "profile", "prompt_tokens", "completion_tokens", "latency_ms",
}


def placeholder_rows(*, profile: str = "inkling-small-nvfp4") -> list[dict[str, Any]]:
    """Return no rows: scaffolding must not manufacture result-looking placeholders."""
    if not isinstance(profile, str) or not profile:
        raise ValueError("profile is required")
    return []


def validate_rows(rows: Any) -> dict[str, Any]:
    if not isinstance(rows, list) or len(rows) != EXPECTED_ROW_COUNT:
        raise ValueError(f"benchmark must contain exactly {EXPECTED_ROW_COUNT} rows")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != REQUIRED_KEYS:
            raise ValueError(f"benchmark row {index} has invalid schema")
        row_id = row["row_id"]
        if not isinstance(row_id, str) or not row_id or row_id in seen:
            raise ValueError(f"benchmark row {index} has duplicate/invalid row_id")
        seen.add(row_id)
        if row["status"] not in ALLOWED_STATUSES:
            raise ValueError(f"benchmark row {index} has invalid status")
        if not isinstance(row["profile"], str) or not row["profile"]:
            raise ValueError(f"benchmark row {index} profile is required")
        for field in ("prompt_tokens", "completion_tokens", "latency_ms"):
            value = row[field]
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"benchmark row {index} {field} must be a non-negative integer or null")
        if row["status"] == "NOT_RUN" and any(row[field] is not None for field in ("prompt_tokens", "completion_tokens", "latency_ms")):
            raise ValueError("NOT_RUN rows cannot claim measurements")
    return {
        "schema_version": 1,
        "status": "VALID",
        "row_count": len(rows),
        "statuses": dict(sorted(Counter(row["status"] for row in rows).items())),
    }


def validate_benchmark_artifact(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("benchmark artifact must be an object")
    required = {"schema_version", "status", "runtime_manifest_sha256", "rows", "infrastructure_failures"}
    if set(value) != required:
        raise ValueError("benchmark artifact keys do not match schema")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported benchmark artifact schema")
    if not isinstance(value["runtime_manifest_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", value["runtime_manifest_sha256"]):
        raise ValueError("benchmark artifact runtime manifest digest is invalid")
    infra = value["infrastructure_failures"]
    if type(infra) is not int or infra < 0:
        raise ValueError("infrastructure_failures must be a non-negative integer")
    status = value["status"]
    if status == "NOT_RUN":
        if value["rows"] != [] or infra != 0:
            raise ValueError("NOT_RUN benchmark artifacts must contain no result rows and zero failures")
        return {"schema_version": 1, "status": "NOT_RUN", "row_count": 0, "infrastructure_failures": 0}
    if status not in {"ELIGIBLE", "DIAGNOSTIC", "DISQUALIFIED"}:
        raise ValueError("benchmark artifact status is invalid")
    row_summary = validate_rows(value["rows"])
    return {**row_summary, "artifact_status": status, "infrastructure_failures": infra}


def accept_benchmark_artifact(value: Any) -> dict[str, Any]:
    checked = validate_benchmark_artifact(value)
    if checked.get("artifact_status") != "ELIGIBLE":
        raise ValueError("only an eligible benchmark artifact can be accepted")
    if checked["infrastructure_failures"] != 0:
        raise ValueError("eligible benchmark acceptance requires zero infrastructure failures")
    if checked["statuses"].get("NOT_RUN"):
        raise ValueError("eligible benchmark acceptance cannot contain NOT_RUN rows")
    return {**checked, "accepted": True}
