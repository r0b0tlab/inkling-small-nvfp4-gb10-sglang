from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "sglang-b7252-sm121" / "inkling_common_moe.py"


def test_cutlass_uses_standard_topk_routing() -> None:
    source = PATCH.read_text()
    assignment = source.split("self.gate.emit_packed_topk =", 1)[1].split(")\n\n    def _forward_routed", 1)[0]
    assert "get_moe_runner_backend().is_flashinfer_trtllm_routed()" in assignment
    assert "is_flashinfer_cutlass" not in assignment


def test_patched_source_is_not_upstream_copy() -> None:
    source = PATCH.read_text()
    assert "Only the FlashInfer TRT-LLM routed runner consumes packed top-k" in source
    assert "numerically corrupt output" in source
