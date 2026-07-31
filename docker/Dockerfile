# syntax=docker/dockerfile:1.7
FROM lmsysorg/sglang@sha256:c60f221f8f42929469bedf74716b4314a1951ff97556dd9e17d9e11040512ac6

LABEL org.opencontainers.image.title="SGLang Inkling-Small NVFP4 GB10 runtime" \
      org.opencontainers.image.source="https://github.com/r0b0tlab/inkling-small-nvfp4-gb10-sglang" \
      org.opencontainers.image.base.digest="sha256:c60f221f8f42929469bedf74716b4314a1951ff97556dd9e17d9e11040512ac6" \
      io.r0b0tlab.sglang.commit="b7252cc6b0c78b25ecea7ee5efa91a6ae37d0f19" \
      io.r0b0tlab.torchcodec.version="0.12.0+cu130" \
      io.r0b0tlab.torchcodec.wheel.sha256="7293d3bbf3e27621f7d5888cc10e233d828faa5896f76f4d3f3ab1457e9c8c9e"

ADD --checksum=sha256:7293d3bbf3e27621f7d5888cc10e233d828faa5896f76f4d3f3ab1457e9c8c9e \
  https://download.pytorch.org/whl/cu130/torchcodec-0.12.0%2Bcu130-cp312-cp312-manylinux_2_28_aarch64.whl \
  /tmp/torchcodec-0.12.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl
RUN python -m pip install --no-cache-dir --no-deps \
      /tmp/torchcodec-0.12.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl \
    && python -c 'import importlib.metadata as m; assert m.version("torchcodec") == "0.12.0+cu130"' \
    && rm -f /tmp/torchcodec-0.12.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl
