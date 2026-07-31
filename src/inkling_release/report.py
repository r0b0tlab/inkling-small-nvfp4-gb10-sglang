from __future__ import annotations

import json
from typing import Any

from .benchmark import validate_rows
from .evidence import validate_evidence


def render_report(evidence: dict[str, Any], rows: list[dict[str, Any]] | None = None) -> str:
    checked = validate_evidence(evidence)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": checked["status"],
        "claim": checked["claim"],
        "runtime_manifest_sha256": checked["runtime_manifest_sha256"],
    }
    if rows is not None:
        summary["benchmark"] = validate_rows(rows)
    return "# Inkling-Small SGLang evidence report\n\n```json\n" + json.dumps(summary, indent=2, sort_keys=True) + "\n```\n"
