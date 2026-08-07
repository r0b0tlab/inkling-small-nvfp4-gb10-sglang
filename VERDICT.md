# Verdict

## COMPLETE — Marlin FP4/MoE on SM121

### Runtime identity

- **Image**: `lmsysorg/sglang:dev-inkling-small-dgx-spark`
  - SGLang source: `sgl-project/sglang@a74222ef6e690f851e2e4ff1c0be7dc1357be313`
- **Model**: `thinkingmachines/Inkling-Small-NVFP4@b6a99534467840620d411e4cd4ad5819b2610d9c`
- **Tensor cores**: sm_121a HMMA/UMMA (6,184 TC instructions verified)

### r0b0bench core-subset (11 lanes) — ALL PASS

| Lane | Result |
|------|--------|
| canary | 5/5 PASS |
| BFCL-MT | 54.0% |
| BFCL-AST | 30.2% |
| throughput | 13.8 tok/s decode, 14,109 tok/s prefill |
| concurrency | c4 = 46.8 tok/s |
| NIAH | 3/3 PASS (61K/122K/220K) |
| QA | 24.5% |
| IFEval | 42.5% |
| HumanEval | 73.2% pass@1 |
| GSM8K | 81.0% |

Context: 245K (physical KV ceiling ~261K on 2x 128GB GB10).

### MTP comparison

MTP 8-1-9: BFCL-MT 57.5%, decode +15%, but KV capacity drops to 8K tokens.
See `evidence/BENCHMARK.md` for full comparison.
