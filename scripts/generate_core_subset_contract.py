#!/usr/bin/env python3
"""Generate the immutable r0b0bench core-subset-aligned quality ID contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

NAMESPACE = "r0b0tlab.r0b0bench-core-subset-aligned.v1"
DATASETS: dict[str, dict[str, Any]] = {
    "gsm8k": {"n": 1319, "fingerprint": "59ec1b7f9357c7a2", "content_sha256": "32c548f08195e19e33408b844dd7be6aa4bcae457d957bc05909ff4bd4a00595"},
    "arc_challenge": {"n": 1172, "fingerprint": "a4361c3f3e560fcd", "content_sha256": "a00c3127fa2437025957049bb97760ce0e7e974bcf5a918b7621f36ab9c3fed8"},
    "piqa": {"n": 1838, "fingerprint": "506e3cab853d8e73", "content_sha256": "256cd78377a0090efcbc05662c15821c9b60f98a708bf48115d0d9af80c67313"},
    "winogrande": {"n": 1267, "fingerprint": "5b125086384c0403", "content_sha256": "8e8671d6097f314f4ddc6c9734e382810926a1e0c3e76664715786d25b4a78d4"},
    "truthfulqa_mc1": {"n": 817, "fingerprint": "8ba81adae744fd06", "content_sha256": "4a78ed8155f86d58a52b8b58da92092d5ca6b98f17228db9cfeb6345f697e3a6"},
    "mmlu_abstract_algebra": {"n": 100, "fingerprint": "a405e35a20e96c3e", "content_sha256": "3847c2c3f9e15bd9359af4125be519da267ddbd7a5caa3665b6e528c73a47bf8"},
    "mmlu_business_ethics": {"n": 100, "fingerprint": "73d48c7e397703f2", "content_sha256": "37bc72f90db6919fe18821dc86d7eff756a74c304d81ff99167733300d80d4f8"},
    "mmlu_clinical_knowledge": {"n": 265, "fingerprint": "52dba1b9a7a1c715", "content_sha256": "fc6d4d450d00433b5ac03ffb7315eb5dbd31c8fa78937830441bc2a0dbc562ac"},
    "mmlu_college_biology": {"n": 144, "fingerprint": "f481f714da28f58e", "content_sha256": "2f2fbd2ff015131bb77205cffc9c493d1480eebb99b275b684b90d7bb7d8fc75"},
    "mmlu_computer_security": {"n": 100, "fingerprint": "c00551a55138390b", "content_sha256": "c5090424a259b37350f5c609be663cea15e1ba5b40e0b26b6d0ffd24a1d645fb"},
    "mmlu_conceptual_physics": {"n": 235, "fingerprint": "d9d38a5e9279b6ad", "content_sha256": "e975729bf9f3ff178d7eb0d7e27fba8d21ffcfe9759d181d18bba0c82ae549cb"},
    "mmlu_high_school_world_history": {"n": 237, "fingerprint": "7bc6897c1ad886c5", "content_sha256": "ab62d32433df5edf824b8be12d308ad404a1e1bf1335c65a890c5c2e0302e99a"},
    "mmlu_international_law": {"n": 121, "fingerprint": "74c39d84b2740ac6", "content_sha256": "9c9f48150f8100e5c189e1b3797a9a321922d051bd75302d9401bbb63b2efd50"},
    "ifeval": {"n": 541, "fingerprint": "62b01b907665bf3d", "content_sha256": "1876c93e8187cbcd4f863fe79ab86a22f6d34b5823553d4a4d020a4659d9ddba"},
    "humaneval": {"n": 164, "fingerprint": "2e5aa734ef24e324", "content_sha256": "e5ea46f61edfa7e3cbd5a86af0fb11e4a8a1048d9b905f2436be8bcdcf66f6d6"},
}
QA_TASKS = tuple(name for name in DATASETS if name not in {"ifeval", "humaneval"})


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def ranked_ids(lane: str, items: list[str], count: int) -> list[str]:
    if len(items) != len(set(items)) or not 0 < count <= len(items):
        raise ValueError("invalid selection population")
    key = lambda item: (hashlib.sha256(f"{NAMESPACE}\0{lane}\0{item}".encode()).hexdigest(), item)
    return sorted(items, key=key)[:count]


def generate_contract() -> dict[str, Any]:
    qa_population = [f"{task}:{index}" for task in QA_TASKS for index in range(DATASETS[task]["n"])]
    if len(qa_population) != 7715:
        raise RuntimeError(f"QA population drift: {len(qa_population)}")
    gsm8k = ranked_ids("gsm8k", [f"gsm8k:{index}" for index in range(DATASETS["gsm8k"]["n"])], 200)
    qa_eligible = [item for item in qa_population if item not in set(gsm8k)]
    if len(qa_eligible) != 7515:
        raise RuntimeError(f"QA eligible population drift: {len(qa_eligible)}")
    qa = ranked_ids("qa", qa_eligible, 400)
    ifeval = ranked_ids("ifeval", [f"ifeval:{index}" for index in range(DATASETS["ifeval"]["n"])], 200)
    humaneval = [f"humaneval:{index}" for index in range(DATASETS["humaneval"]["n"])]
    contract: dict[str, Any] = {
        "schema_version": 1,
        "contract": "r0b0bench-core-subset-aligned-v1",
        "selection": {"algorithm": "sha256(namespace\\0lane\\0stable_id), ascending; stable_id tie-break", "namespace": NAMESPACE},
        "r0b0bench": {
            "commit": "b69249337244d5d07dbdbabedda806456a96fe02",
            "tree": "cb378f0ea477afd76fcdbba32b99c7495f5bb879",
            "profile": "core-subset",
            "profile_sha256": "ae1d0208a615b5db6f51d421e0296fd15202e25d3fc9acd1fd916fbbba1539df",
            "image_digest": "sha256:08f5c1d39a068fc4c3f5204841149c00a4f17537756de507f33de0d70a8af8ef",
            "quality_status": "NOT_IMPLEMENTED_IN_RC1_OFFICIAL_SCORERS_REQUIRED",
            "bfcl_status": "EXTERNAL_IMPORT_REQUIRED_IN_RC1",
        },
        "software": {"datasets": "4.3.0", "lm_eval": "0.4.12", "requests": "2.34.2"},
        "scorers": {"ifeval_utils_sha256": "1ab8f14808c826f93f2364883487ed63cf4267980bf4761fda8053899c013632", "ifeval_instructions_sha256": "511cc41a53787d818c292d8335c8e98aa833296a0789ac470db56bcd6496345e"},
        "datasets": DATASETS,
        "lanes": {
            "qa": {"population": 7715, "eligible_after_gsm8k_exclusion": 7515, "excluded_lane": "gsm8k", "expected": 400, "ids": qa},
            "ifeval": {"population": 541, "expected": 200, "ids": ifeval},
            "humaneval": {"population": 164, "expected": 164, "ids": humaneval},
            "gsm8k": {"population": 1319, "expected": 200, "ids": gsm8k, "method": "0-shot flexible-extract"},
        },
        "quality_rows": 964,
        "bfcl_rows": 800,
        "scored_rows": 1764,
        "qa_gsm8k_overlap_ids": sorted(set(qa) & set(gsm8k)),
    }
    contract["selection_sha256"] = hashlib.sha256(canonical_bytes(contract["lanes"])).hexdigest()
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = generate_contract()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode() + b"\n")
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
