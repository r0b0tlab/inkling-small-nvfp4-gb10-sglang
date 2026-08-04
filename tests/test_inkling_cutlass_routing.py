from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "sglang-b7252-sm121" / "inkling_common_moe.py"


def test_quantized_stock_paths_preserve_packed_topk_routing() -> None:
    source = PATCH.read_text()
    assignment = source.split("self.gate.emit_packed_topk =", 1)[1].split(")\n\n    def _forward_routed", 1)[0]
    assert "not isinstance(self.experts.quant_method, UnquantizedFusedMoEMethod)" in assignment
    assert "bf16_routed_uses_stock_fused_moe" in assignment
    assert "is_flashinfer_trtllm_routed()" not in assignment


def test_pinned_inkling_common_moe_matches_upstream() -> None:
    import hashlib

    digest = hashlib.sha256(PATCH.read_bytes()).hexdigest()
    assert digest == "36cbc6b7c717024cfccc04d1f530c36ddf7e363fc2ac0cd20d256039721c3094"
