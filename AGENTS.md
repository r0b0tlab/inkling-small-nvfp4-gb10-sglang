# Repository instructions

This repository publishes a weight-free, exact-provenance SGLang runtime and benchmark package for `thinkingmachines/Inkling-Small-NVFP4@b6a99534467840620d411e4cd4ad5819b2610d9c` on two NVIDIA GB10/SM121 nodes.

Hard constraints:

- Native CUDA 13/aarch64/SM121 only. Never add emulation, Marlin, CPU offload, dequantized routed-expert execution, Transformers model fallback, or metadata-only shortcuts.
- Runtime source is `sgl-project/sglang@b7252cc6b0c78b25ecea7ee5efa91a6ae37d0f19`; parent is immutable arm64 manifest `sha256:c60f221f8f42929469bedf74716b4314a1951ff97556dd9e17d9e11040512ac6`, config `sha256:a2364fcb06508b66f464ecd16921144619bdac9aa883391bcdec95be2b632293`.
- Do not embed/download/publish weights. Runtime model mounts are exact, read-only, offline snapshots.
- Renderers do not execute. Launch and stop paths are separate, fail closed, and preserve all-rank logs.
- Never embed private addresses, hostnames, users, local paths, credentials, raw logs, or model payloads in tracked/public files.
- Preserve model failures; repair/rerun infrastructure failures. Eligible, diagnostic, and disqualified evidence are separate.
- Any source, package, image, command, profile, or model mutation invalidates downstream evidence.
- Contract tests precede implementation changes. Run `make test`, `make lint`, `make scan`, and `git diff --check` before commit.
- No external push or publication without the release owner's exact-SHA gate.
