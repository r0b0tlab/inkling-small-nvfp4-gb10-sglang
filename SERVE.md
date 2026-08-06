# Serving configuration — verified

## Working TP=2 configuration on 2× DGX Spark (SM121)

Image: `lmsysorg/sglang:dev-inkling-small-dgx-spark`
Model: `thinkingmachines/Inkling-Small-NVFP4@b6a99534467840620d411e4cd4ad5819b2610d9c`

### Key flags

```text
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
--trust-remote-code
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

### Environment

```text
NCCL_IB_DISABLE=1
NCCL_NVLS_ENABLE=0
NCCL_SOCKET_IFNAME=<direct-link-interface>
TRITON_CACHE_DIR=<writable-cache-dir>
HOME=<writable-home-dir>
```

### Backends NOT to use on SM121

FlashInfer CUTLASS MoE (`--moe-runner-backend flashinfer_cutlass`) produces
incoherent output on SM121 for NVFP4 Inkling. Always use Marlin.
