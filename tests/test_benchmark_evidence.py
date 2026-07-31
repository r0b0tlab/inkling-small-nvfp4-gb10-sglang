from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from inkling_release.bench import render_benchmark_plan
from inkling_release.benchmark import EXPECTED_ROW_COUNT, accept_benchmark_artifact, placeholder_rows, validate_benchmark_artifact, validate_rows
from inkling_release.evidence import EvidenceError, accept_evidence
from inkling_release.gates import evaluate_gates
from inkling_release.profiles import load_profile
from inkling_release.report import render_report
from inkling_release.tuning import plan_tuning


def _rows(status: str = "ELIGIBLE") -> list[dict[str, object]]:
    return [{"row_id": f"row-{i:04d}", "status": status, "profile": "inkling-small-nvfp4", "prompt_tokens": 10, "completion_tokens": 5, "latency_ms": 100} for i in range(EXPECTED_ROW_COUNT)]


def _artifact(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {"schema_version": 1, "status": "ELIGIBLE", "runtime_manifest_sha256": "a" * 64, "rows": _rows(), "infrastructure_failures": 0}
    value.update(updates)
    return value


def test_exact_row_count_and_unique_ids() -> None:
    assert validate_rows(_rows())["row_count"] == EXPECTED_ROW_COUNT
    with pytest.raises(ValueError, match="exactly"):
        validate_rows(_rows()[:-1])
    rows = _rows()
    rows[-1]["row_id"] = rows[0]["row_id"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_rows(rows)


def test_row_numeric_contract() -> None:
    for bad in (True, -1, 1.5):
        rows = _rows()
        rows[0]["latency_ms"] = bad
        with pytest.raises(ValueError):
            validate_rows(rows)
    with pytest.raises(ValueError, match="cannot claim"):
        validate_rows(_rows("NOT_RUN"))


def test_placeholders_never_fabricate_results() -> None:
    assert placeholder_rows() == []
    with pytest.raises(ValueError):
        placeholder_rows(profile="")


def test_acceptance_requires_eligible_zero_infra_and_no_not_run() -> None:
    assert accept_benchmark_artifact(_artifact())["accepted"] is True
    with pytest.raises(ValueError, match="zero infrastructure"):
        accept_benchmark_artifact(_artifact(infrastructure_failures=1))
    with pytest.raises(ValueError, match="eligible"):
        accept_benchmark_artifact(_artifact(status="DIAGNOSTIC"))
    with pytest.raises(ValueError, match="NOT_RUN"):
        accept_benchmark_artifact(_artifact(rows=_rows("NOT_RUN")))


def test_not_run_artifact_must_be_empty() -> None:
    value = _artifact(status="NOT_RUN", rows=[], infrastructure_failures=0)
    assert validate_benchmark_artifact(value)["row_count"] == 0
    value["rows"] = _rows()
    with pytest.raises(ValueError, match="NOT_RUN"):
        validate_benchmark_artifact(value)


def test_foundation_evidence_rejects_claim_verdicts() -> None:
    base = {"schema_version": 1, "status": "DIAGNOSTIC", "claim": "diagnostic only", "runtime_manifest_sha256": "b" * 64, "rows": []}
    assert accept_evidence(base)["accepted"] is True
    for status in ("PASS", "QUALIFIED", "GO"):
        changed = deepcopy(base)
        changed["status"] = status
        with pytest.raises(EvidenceError):
            accept_evidence(changed)


def test_gate_set_and_manifest_identity_fail_closed() -> None:
    gates = {name: True for name in ("source_integrity", "structure", "privacy", "dedup", "objective")}
    assert evaluate_gates(gates, runtime_manifest_sha256="a", evidence_manifest_sha256="a")["status"] == "PHASE0_PASS"
    assert evaluate_gates(gates, runtime_manifest_sha256="a", evidence_manifest_sha256="b")["status"] == "NO_VERDICT"
    del gates["dedup"]
    assert evaluate_gates(gates, runtime_manifest_sha256="a", evidence_manifest_sha256="a")["status"] == "NO_VERDICT"


def test_benchmark_plan_is_empty_and_nonexecuting() -> None:
    root = Path(__file__).resolve().parents[1]
    profile = load_profile(root / "profiles" / "inkling-small-nvfp4.json")
    plan = render_benchmark_plan(profile, node_rank=0, dist_init_addr="192.0.2.1:25160")
    assert plan["row_count_contract"] == EXPECTED_ROW_COUNT
    assert plan["rows"] == []
    assert plan["executes"] is False


def test_tuning_plan_is_bounded() -> None:
    plans = plan_tuning(mem_fraction_static=[0.8, 0.85], context_lengths=[8192], chunk_sizes=[1024], concurrency=[1, 2])
    assert len(plans) == 4
    with pytest.raises(ValueError):
        plan_tuning(mem_fraction_static=[0.99])
    with pytest.raises(ValueError):
        plan_tuning(concurrency=[0])


def test_report_contains_validated_posture() -> None:
    evidence = {"schema_version": 1, "status": "NO_VERDICT", "claim": "no runtime verdict", "runtime_manifest_sha256": "c" * 64, "rows": []}
    report = render_report(evidence)
    assert "NO_VERDICT" in report and "no runtime verdict" in report


def _subset_module(name: str):
    script = Path(__file__).resolve().parents[1] / "scripts" / "generate_core_subset_contract.py"
    spec = importlib.util.spec_from_file_location(name, script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_core_subset_contract_is_deterministic_and_disjoint() -> None:
    module = _subset_module("subset_contract_test")
    first = module.generate_contract()
    assert first == module.generate_contract()
    assert (first["scored_rows"], first["quality_rows"], first["bfcl_rows"]) == (1764, 964, 800)
    assert first["qa_gsm8k_overlap_ids"] == []
    assert len(first["lanes"]["qa"]["ids"]) == 400
    assert first["lanes"]["qa"]["eligible_after_gsm8k_exclusion"] == 7515


def test_checked_in_core_subset_contract_matches_generator() -> None:
    root = Path(__file__).resolve().parents[1]
    checked = json.loads((root / "benchmark" / "contracts" / "core-subset-aligned-v1.json").read_text())
    assert checked == _subset_module("subset_contract_checked").generate_contract()