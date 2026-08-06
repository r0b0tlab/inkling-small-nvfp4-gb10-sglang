# Verdict

## PHASE0_PASS — Marlin FP4/MoE on SM121

Serving configuration verified on 2× NVIDIA GB10 / DGX Spark (SM121) using the
official SGLang DGX Spark day-0 image with Marlin FP4 GEMM and Marlin MoE runner.

### Runtime identity

- Image: `lmsysorg/sglang:dev-inkling-small-dgx-spark`
  - SGLang source: `sgl-project/sglang@a74222ef6e690f851e2e4ff1c0be7dc1357be313`
  - CUDA 13.0.1, FlashInfer 0.6.15.post1, PyTorch with `sm_120a`+`sm_121a` targets
- Model: `thinkingmachines/Inkling-Small-NVFP4@b6a99534467840620d411e4cd4ad5819b2610d9c`
- sgl-kernel `sm100/common_ops.abi3.so` contains fatbin ELF for `sm_121a` with
  6,156 HMMA + 28 UMMA tensor core instructions across 1,098 kernels

### Working configuration

```
--trust-remote-code
--model-path /models/Inkling-Small-NVFP4
--tp 2 --nnodes 2
--quantization modelopt_fp4
--attention-backend triton --page-size 128
--fp4-gemm-backend marlin
--moe-runner-backend marlin
--mamba-radix-cache-strategy extra_buffer
--mem-fraction-static 0.85
--swa-full-tokens-ratio 0.1 --mamba-full-memory-ratio 0.1
--enable-multimodal
--reasoning-parser inkling --tool-call-parser inkling
--disable-prefill-cuda-graph
```

### Key finding

FlashInfer CUTLASS MoE runner produces incoherent output on SM121 for NVFP4 Inkling.
The SGLang cookbook's verified DGX Spark recipe specifies Marlin FP4/MoE backends,
which correctly activate SM121 Blackwell tensor cores (HMMA/UMMA).

### Evidence

See `evidence/` directory for r0b0bench core-subset results, throughput measurements,
and smoke-test canary outputs.
