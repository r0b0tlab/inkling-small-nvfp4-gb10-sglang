# Verdict

## PHASE0_PASS — Marlin FP4/MoE on SM121

### Runtime identity

- **Image**: `lmsysorg/sglang:dev-inkling-small-dgx-spark`
  - SGLang source: `sgl-project/sglang@a74222ef6e690f851e2e4ff1c0be7dc1357be313`
  - CUDA 13.0.1, FlashInfer 0.6.15.post1, PyTorch with sm_120a + sm_121a targets
- **Model**: `thinkingmachines/Inkling-Small-NVFP4@b6a99534467840620d411e4cd4ad5819b2610d9c`
- **sgl-kernel**: `sm100/common_ops.abi3.so` contains fatbin ELF for `sm_121a`
  with 6,156 HMMA + 28 UMMA tensor core instructions across 1,098 kernels

### Verified configuration

```
--fp4-gemm-backend marlin
--moe-runner-backend marlin
--attention-backend triton
--quantization modelopt_fp4
--disable-prefill-cuda-graph
```

### r0b0bench core-subset (11 lanes)

10 PASS, 1 ERROR (NIAH KV capacity at 50%+ of 1M context).

| Lane | Result |
|------|--------|
| canary | 5/5 PASS |
| BFCL-MT | 54.0% (108/200) |
| BFCL-AST | 30.2% micro (181/600) |
| throughput | 13.8 tok/s decode, 14,109 tok/s prefill |
| concurrency | c4 = 46.8 tok/s |
| QA | 24.5% |
| IFEval | 42.5% |
| HumanEval | 73.2% pass@1 |
| GSM8K | 81.0% |
| NIAH | 262K PASS, 524K ERROR |

### Key finding

FlashInfer CUTLASS MoE produces incoherent output on SM121 for NVFP4 Inkling.
Marlin FP4/MoE correctly activates SM121 Blackwell tensor cores.

### MTP comparison

MTP 8-1-9 provides +16% decode throughput (16.0 vs 13.8 tok/s) at c1 but
degrades latency (+41% for short outputs) and prefill (-17%). Not recommended
for this 2-node TP=2 deployment.
