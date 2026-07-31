# syntax=docker/dockerfile:1.7
FROM lmsysorg/sglang@sha256:c60f221f8f42929469bedf74716b4314a1951ff97556dd9e17d9e11040512ac6

ARG RELEASE_BUILD_REVISION
ARG PATCH_BUNDLE_SHA256
ARG HELION_CONFIG_SHA256

LABEL org.opencontainers.image.title="SGLang Inkling-Small NVFP4 GB10 runtime" \
      org.opencontainers.image.source="https://github.com/r0b0tlab/inkling-small-nvfp4-gb10-sglang" \
      org.opencontainers.image.base.name="lmsysorg/sglang@sha256:c60f221f8f42929469bedf74716b4314a1951ff97556dd9e17d9e11040512ac6" \
      org.opencontainers.image.base.digest="sha256:c60f221f8f42929469bedf74716b4314a1951ff97556dd9e17d9e11040512ac6" \
      org.opencontainers.image.revision="$RELEASE_BUILD_REVISION" \
      io.r0b0tlab.sglang.commit="b7252cc6b0c78b25ecea7ee5efa91a6ae37d0f19" \
      io.r0b0tlab.sglang.tree="b8c966b81dcce80824261ccae7aa9d33441935a7" \
      io.r0b0tlab.sglang.patch-bundle-sha256="$PATCH_BUNDLE_SHA256" \
      io.r0b0tlab.helion.silu-interleaved-sm121-sha256="$HELION_CONFIG_SHA256" \
      io.r0b0tlab.torchcodec.version="0.12.0+cu130" \
      io.r0b0tlab.torchcodec.wheel.sha256="7293d3bbf3e27621f7d5888cc10e233d828faa5896f76f4d3f3ab1457e9c8c9e" \
      io.r0b0tlab.model.id="thinkingmachines/Inkling-Small-NVFP4" \
      io.r0b0tlab.model.revision="b6a99534467840620d411e4cd4ad5819b2610d9c"

ADD --checksum=sha256:7293d3bbf3e27621f7d5888cc10e233d828faa5896f76f4d3f3ab1457e9c8c9e \
  https://download.pytorch.org/whl/cu130/torchcodec-0.12.0%2Bcu130-cp312-cp312-manylinux_2_28_aarch64.whl \
  /tmp/torchcodec-0.12.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl

RUN python -m pip install --no-cache-dir --no-deps \
      /tmp/torchcodec-0.12.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl \
    && python -c 'import importlib.metadata as m; assert m.version("torchcodec") == "0.12.0+cu130"' \
    && rm -f /tmp/torchcodec-0.12.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl

COPY patches/sglang-b7252-sm121/inkling_moe.py \
  /sgl-workspace/sglang/python/sglang/kernels/ops/moe/inkling_moe.py
COPY patches/sglang-b7252-sm121/flashinfer_cutlass.py \
  /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py
COPY patches/sglang-b7252-sm121/inkling_common_moe.py \
  /sgl-workspace/sglang/python/sglang/srt/models/inkling_common/moe.py
COPY patches/sglang-b7252-sm121/silu_and_mul_interleaved_sm_121.json \
  /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/silu_and_mul_interleaved_sm_121.json
COPY patches/sglang-b7252-sm121/bundle-descriptor.json \
  /usr/share/inkling-sglang/patch-bundle.json

RUN set -eux; \
    test -n "$RELEASE_BUILD_REVISION"; \
    test -n "$PATCH_BUNDLE_SHA256"; \
    test -n "$HELION_CONFIG_SHA256"; \
    test "$(sha256sum /usr/share/inkling-sglang/patch-bundle.json | cut -d' ' -f1)" = "$PATCH_BUNDLE_SHA256"; \
    test "$(sha256sum /sgl-workspace/sglang/python/sglang/kernels/ops/moe/inkling_moe.py | cut -d' ' -f1)" = "d473736bdf300b46c6f2f546ee13ab3c2769abca5a2761a78c0bfcc229f4e777"; \
    test "$(sha256sum /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py | cut -d' ' -f1)" = "4b3057068aebc7d80583d3249818f6a4ac40d34a1d8eba0ffa116a813ef17260"; \
    test "$(sha256sum /sgl-workspace/sglang/python/sglang/srt/models/inkling_common/moe.py | cut -d' ' -f1)" = "2addced79a2a2132e89edba9a96efebd6568f291fa99ec06fc5098dd62002d9a"; \
    test "$(sha256sum /sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/configs/silu_and_mul_interleaved_sm_121.json | cut -d' ' -f1)" = "$HELION_CONFIG_SHA256"

# FlashInfer materializes checksummed routed-MoE artifacts and compatibility
# symlinks lazily below this cache root. Keep site-packages immutable except for
# the exact sticky cache directory required by the non-root runtime UID.
RUN set -eux; \
    cubin_root=/usr/local/lib/python3.12/dist-packages/flashinfer_cubin/cubins; \
    test -d "$cubin_root"; \
    chmod 1777 "$cubin_root"; \
    test "$(stat -c '%a' "$cubin_root")" = 1777
