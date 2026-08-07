# Repository instructions

This repository publishes a weight-free, reproducible SGLang runtime and
benchmark package for `thinkingmachines/Inkling-Small-NVFP4` on two NVIDIA
GB10 / DGX Spark (SM121) nodes.

## What works (verified)

- **Image**: `lmsysorg/sglang:dev-inkling-small-dgx-spark` (official SGLang DGX Spark build)
- **Backends**: Marlin FP4 GEMM + Marlin MoE runner + Triton attention
- **Tensor cores**: sm_121a HMMA/UMMA (6,184 TC instructions in sgl-kernel)
- **11-lane r0b0bench core-subset**: 10 PASS, 1 NIAH ERROR (KV capacity at 50%+ of 1M context)

## What does NOT work on SM121

FlashInfer CUTLASS MoE (`--moe-runner-backend flashinfer_cutlass`) produces
incoherent output on SM121 for NVFP4 Inkling. Always use Marlin.

## Reproducibility

- Use the official DGX Spark image directly — no custom build needed
- Mount weights read-only at `/models/Inkling-Small-NVFP4`
- See `SERVE.md` for the exact serve command
- See `evidence/BENCHMARK.md` for verified results
- See `docker/` and `BUILD.md` for the historical custom build (retained for provenance)

## Running benchmarks

```bash
# Install r0b0bench
pip install git+https://github.com/r0b0tlab/r0b0bench.git

# Run full 11-lane core-subset
r0b0bench run \
  --profile core-subset \
  --base-url http://<node0-ip>:30000/v1 \
  --model "/models/Inkling-Small-NVFP4" \
  --tokenizer /path/to/Inkling-Small-NVFP4 \
  --output /tmp/results \
  --run-id my-run
```

## Constraints

- Native CUDA 13/aarch64/SM121 only
- Do not embed weights in the image
- Never embed private addresses, credentials, or raw logs in tracked files
- Any source, image, or model mutation invalidates downstream evidence
