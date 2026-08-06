# Inkling-Small NVFP4 — SGLang on NVIDIA DGX Spark (GB10 / SM121)

Serving `thinkingmachines/Inkling-Small-NVFP4` on two NVIDIA DGX Spark nodes
(2× GB10 / SM121) using the official SGLang day-0 DGX Spark image with native
Marlin FP4/MoE backends.

## Status

**PHASE0_PASS** — the model loads, serves coherent output, and passes the
r0b0bench core-subset canary suite. See [VERDICT.md](VERDICT.md).

## Quick start

### 1. Download weights

```bash
huggingface-cli download thinkingmachines/Inkling-Small-NVFP4 \
  --local-dir /path/to/Inkling-Small-NVFP4
```

### 2. Pull the image

```bash
docker pull lmsysorg/sglang:dev-inkling-small-dgx-spark
```

### 3. Launch TP=2 across two DGX Spark nodes

**Node 0 (rank 0):**

```bash
docker run -d --name inkling-rank0 \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  -v /path/to/Inkling-Small-NVFP4:/models/Inkling-Small-NVFP4:ro \
  -e SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
  -e NCCL_IB_DISABLE=1 -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_SOCKET_IFNAME=<node0-iface> \
  -e TRITON_CACHE_DIR=/tmp/triton -e HOME=/tmp \
  lmsysorg/sglang:dev-inkling-small-dgx-spark \
  sglang serve \
    --trust-remote-code \
    --model-path /models/Inkling-Small-NVFP4 \
    --tp 2 --nnodes 2 --node-rank 0 \
    --dist-init-addr <node0-ip>:25200 \
    --quantization modelopt_fp4 \
    --attention-backend triton --page-size 128 \
    --fp4-gemm-backend marlin \
    --moe-runner-backend marlin \
    --mamba-radix-cache-strategy extra_buffer \
    --mem-fraction-static 0.85 \
    --swa-full-tokens-ratio 0.1 --mamba-full-memory-ratio 0.1 \
    --enable-multimodal \
    --reasoning-parser inkling --tool-call-parser inkling \
    --disable-prefill-cuda-graph \
    --host 0.0.0.0 --port 30000
```

**Node 1 (rank 1):** identical except `--node-rank 1`.

### 4. Verify

```bash
curl http://<node0-ip>:30000/health
curl http://<node0-ip>:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/models/Inkling-Small-NVFP4","messages":[{"role":"user","content":"Hello"}],"max_tokens":64}'
```

## Why Marlin (not FlashInfer CUTLASS)?

The SGLang cookbook's verified DGX Spark recipe specifies Marlin FP4/MoE.
FlashInfer CUTLASS produces incoherent output on SM121 for NVFP4 Inkling;
Marlin correctly activates SM121 Blackwell tensor cores (HMMA/UMMA, 6,184 TC
instructions in the `sm_121a` code path).

## Evidence

- `evidence/` — r0b0bench core-subset results, throughput measurements
- `VERDICT.md` — serving verdict and runtime identity

## Credits

- Model: [Thinking Machines](https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4)
- Runtime: [SGLang](https://github.com/sgl-project/sglang) day-0 support
- Benchmark: [r0b0bench](https://github.com/r0b0tlab/r0b0bench) core-subset
- Hardware: NVIDIA DGX Spark (GB10 / SM121)
