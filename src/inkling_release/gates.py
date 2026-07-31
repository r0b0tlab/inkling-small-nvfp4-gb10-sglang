from __future__ import annotations

from typing import Any


REQUIRED_GATES = ("source_integrity", "structure", "privacy", "dedup", "objective")


def evaluate_gates(
    gates: dict[str, Any],
    *,
    runtime_manifest_sha256: str,
    evidence_manifest_sha256: str,
) -> dict[str, Any]:
    if set(gates) != set(REQUIRED_GATES):
        return {"schema_version": 1, "status": "NO_VERDICT", "reason": "gate set mismatch"}
    if runtime_manifest_sha256 != evidence_manifest_sha256:
        return {"schema_version": 1, "status": "NO_VERDICT", "reason": "manifest identity mismatch"}
    if any(gates[name] is not True for name in REQUIRED_GATES):
        return {"schema_version": 1, "status": "NO_VERDICT", "reason": "required gate did not pass"}
    # This foundation can qualify admission but never manufacture a runtime verdict.
    return {"schema_version": 1, "status": "PHASE0_PASS", "gates": dict(gates), "runtime_manifest_sha256": runtime_manifest_sha256}
