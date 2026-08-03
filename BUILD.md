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

## Embedded AOT cache input

The precompiled runtime cache is a release artifact, not source-controlled data.
`aot-cache/` and `aot-cache-manifest.json` are intentionally ignored by Git but
are admitted into the Docker build context through `.dockerignore` exceptions.
A clean build must stage both inputs before invoking BuildKit:

```text
rm -rf aot-cache aot-cache-manifest.json
tar --zstd -xf "$AOT_CACHE_ARCHIVE" -C .
cp "$AOT_CACHE_MANIFEST" aot-cache-manifest.json
CACHE_SHA256=$(sha256sum aot-cache-manifest.json | cut -d' ' -f1)
test "$CACHE_SHA256" = "$EXPECTED_AOT_CACHE_MANIFEST_SHA256"
```

The archive must contain the exact `aot-cache/` directory produced by the
native SM121 AOT campaign. Do not substitute an empty directory, a config-only
JSON, or a host bind mount. The Dockerfile recursively removes write bits from
`/cache`, embeds the manifest, and fails if the supplied manifest hash or
required cache roots do not match.

After staging, the final image build supplies
`--build-arg AOT_CACHE_MANIFEST_SHA256="$CACHE_SHA256"` together with the
source, patch-bundle, and Helion configuration hashes. Preserve the archive,
manifest, and their checksums beside the image export as independent build
provenance.

## Immutable inputs

* Model: `thinkingmachines/Inkling-Small-NVFP4@b6a99534467840620d411e4cd4ad5819b2610d9c`
* Source: `sgl-project/sglang@b7252cc6b0c78b25ecea7ee5efa91a6ae37d0f19`
* Source tree: `b8c966b81dcce80824261ccae7aa9d33441935a7`
* Image index: `sha256:fbea1a4e25b26660dbc2384a27ead8817e9b7670f257b5c3143e0450d14524d7`
* Linux/arm64 manifest: `sha256:c60f221f8f42929469bedf74716b4314a1951ff97556dd9e17d9e11040512ac6`
* Config: `sha256:a2364fcb06508b66f464ecd16921144619bdac9aa883391bcdec95be2b632293`
