# Build contract

The image is a thin overlay over an immutable linux/arm64 SGLang parent. The
Dockerfile uses the exact arm64 manifest digest recorded in
`runtime-manifest.json` and labels the exact SGLang source commit. It does not
copy, download, or package model weights.

## Offline checks

```text
python3 -m compileall -q src scripts
make test
make validate
make safety
```

No Docker build is part of this foundation check. A future live build must
re-check the parent image index, arm64 manifest, config digest, and Docker Hub
source-commit label before any evidence can be reused.

## Immutable inputs

* Model: `thinkingmachines/Inkling-Small-NVFP4@b6a99534467840620d411e4cd4ad5819b2610d9c`
* Source: `sgl-project/sglang@b7252cc6b0c78b25ecea7ee5efa91a6ae37d0f19`
* Source tree: `b8c966b81dcce80824261ccae7aa9d33441935a7`
* Image index: `sha256:fbea1a4e25b26660dbc2384a27ead8817e9b7670f257b5c3143e0450d14524d7`
* Linux/arm64 manifest: `sha256:c60f221f8f42929469bedf74716b4314a1951ff97556dd9e17d9e11040512ac6`
* Config: `sha256:a2364fcb06508b66f464ecd16921144619bdac9aa883391bcdec95be2b632293`
