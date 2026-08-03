# Serving contract

Serving is a later, explicitly admitted operation. This repository only
renders and validates the command. `scripts/render_launch.py` and
`scripts/launch_tp2.py` are safe by default; `launch_tp2.py --execute` is the
separate live boundary and is not used by local tests.

## Official recipe

The command must use `--model-path /models/Inkling-Small-NVFP4`, never the
legacy `--model` spelling, and must include:

```text
SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
--trust-remote-code
--model-impl sglang --tp 2 --nnodes 2 --node-rank RANK
--dist-init-addr RANK0_HOST:PORT
--quantization modelopt_fp4 --attention-backend triton --page-size 128
--fp4-gemm-backend flashinfer_trtllm
--moe-runner-backend flashinfer_cutlass --disable-flashinfer-autotune
--cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled
--mamba-radix-cache-strategy extra_buffer
--mem-fraction-static 0.85
--swa-full-tokens-ratio 0.1 --mamba-full-memory-ratio 0.1
--enable-multimodal --reasoning-parser inkling --tool-call-parser inkling
```

The derivative image includes a content-addressed overlay for the official
Inkling image/audio processor: its CPU Numba patch helper is `cache=False` so
processor discovery works under the read-only editable source tree without
writing an in-image cache. Image/audio requests still require separate live
semantic canaries; Inkling video is not admitted by this profile.

The conservative profile is context 32768, concurrency 1, BF16/default KV,
chunk 1024, and both CUDA-graph phases disabled. `--mem-fraction-static` is tunable
only through the bounded tuning driver. The MTP candidate is exact `8-1-9`
and includes `--enable-multi-layer-eagle`. MXFP8 is a separate candidate and
is not evidence for the FP4 profile.

## Admission and lifecycle

Before a live launch, an admission file must have `status: PHASE0_PASS`, match
the runtime manifest digest, and refer to a model directory owned by the
invoking user. Docker is digest-pinned, uses host networking and a read-only
model mount, and names only `inkling-sglang-tp2-rank0` or
`inkling-sglang-tp2-rank1`. Stop operations accept one owned name and never
perform broad kills. Receipts contain PID, container name, command, and
manifest identity.
