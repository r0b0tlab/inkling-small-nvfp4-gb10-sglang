# Inkling-Small NVFP4 SGLang release foundation

This repository is the weight-free, non-live foundation for serving
`thinkingmachines/Inkling-Small-NVFP4` at revision
`b6a99534467840620d411e4cd4ad5819b2610d9c` on two native CUDA 13/aarch64/SM121
GB10 nodes.

The runtime source is `sgl-project/sglang` commit
`b7252cc6b0c78b25ecea7ee5efa91a6ae37d0f19`, tree
`b8c966b81dcce80824261ccae7aa9d33441935a7`.

## Status

`VERDICT.md` is authoritative and remains `NO_VERDICT`. No model weights,
private endpoints, raw logs, or live-runtime results are included. Renderers
produce JSON/argv only; they never execute Docker or an SGLang server.

## Contracts

* `runtime-manifest.json` and its lock bind model, source, image index,
  linux/arm64 manifest, config, cookbook recipe, and safety policies.
* `profiles/` contains a conservative explicit-BF16-KV profile, an exact MTP
  `8-1-9` candidate, and a separately labeled MXFP8 candidate.
* `schemas/` contains machine-readable contracts. An eligible benchmark
  artifact must contain exactly 1,764 real rows. `NOT_RUN` scaffolding contains
  zero rows so it cannot be mistaken for measurements.
* `scripts/public_safety_scan.py` rejects private paths, credentials, binary
  text, and tracked symlinks.

## Local verification

```text
make test
make validate
make safety
```

All commands are local and offline. See `BUILD.md`, `SERVE.md`, and
`BENCHMARK.md` for the bounded workflows. The live launcher is deliberately
separate and will only be admitted by a `PHASE0_PASS` JSON artifact plus a
user-owned model root.
