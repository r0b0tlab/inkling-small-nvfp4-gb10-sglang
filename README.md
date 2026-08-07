# Inkling-Small NVFP4 — SGLang on NVIDIA DGX Spark (GB10 / SM121)

Serve `thinkingmachines/Inkling-Small-NVFP4` on two NVIDIA DGX Spark nodes
(2x GB10 / SM121) using the official SGLang day-0 image with native Marlin
FP4/MoE backends that correctly activate Blackwell tensor cores.

## Results (r0b0bench core-subset, 11 lanes)

| Lane | Status | Result |
|------|--------|--------|
| canary | PASS | 5/5 checks |
| BFCL-MT | PASS | **54.0%** (108/200, official BFCL v4) |
| BFCL-AST | PASS | **30.2%** micro (181/600) |
| latency | PASS | c1 128-token ~8.5s |
| concurrency | PASS | c4 = 46.8 tok/s aggregate |
| throughput | PASS | decode **13.8 tok/s**, prefill **14,109 tok/s** |
| NIAH | ERROR | 262K PASS, 524K KV capacity limit |
| QA (ARC-Easy) | PASS | **24.5%** (98/400) |
| IFEval | PASS | **42.5%** |
| HumanEval | PASS | **73.2%** pass@1 (120/164) |
| GSM8K | PASS | **81.0%** (162/200, 0-shot) |

MTP 8-1-9 comparison: decode 16.0 tok/s (+16% vs baseline) but worse latency
and prefill. See `evidence/BENCHMARK.md`.

## Quick start (click-run)

### 1. Prerequisites

- Two NVIDIA DGX Spark (GB10) nodes with a direct network link
- Docker with NVIDIA Container Toolkit
- ~170 GB free disk per node for model weights

### 2. Download weights (both nodes)

```bash
huggingface-cli download thinkingmachines/Inkling-Small-NVFP4 \
  --local-dir /opt/models/Inkling-Small-NVFP4
```

### 3. Pull the image (both nodes)

```bash
docker pull lmsysorg/sglang:dev-inkling-small-dgx-spark
```

### 4. Launch TP=2

**Node 0 (rank 0):**

```bash
docker run -d --name inkling-rank0 \
  --gpus all --network host --ipc host \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  -v /opt/models/Inkling-Small-NVFP4:/models/Inkling-Small-NVFP4:ro \
  -v /tmp/inkling-cache:/cache:rw \
  -e SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e NCCL_IB_DISABLE=1 -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_SOCKET_IFNAME=<node0-direct-link-iface> \
  -e TRITON_CACHE_DIR=/cache/triton -e HOME=/cache/home \
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

**Node 1 (rank 1):** identical except `--node-rank 1` and
`-e NCCL_SOCKET_IFNAME=<node1-direct-link-iface>`.

### 5. Verify

```bash
# Wait ~5 min for model load + CUDA graph capture
curl http://<node0-ip>:30000/health
# Expect: 200

curl http://<node0-ip>:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/models/Inkling-Small-NVFP4","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

### 6. Run benchmarks (optional)

```bash
pip install git+https://github.com/r0b0tlab/r0b0bench.git

r0b0bench run \
  --profile core-subset \
  --base-url http://<node0-ip>:30000/v1 \
  --model "/models/Inkling-Small-NVFP4" \
  --tokenizer /opt/models/Inkling-Small-NVFP4 \
  --output /tmp/r0b0bench-results
```

## Why Marlin (not FlashInfer CUTLASS)?

FlashInfer CUTLASS MoE produces **incoherent output** on SM121 for NVFP4
Inkling. The [SGLang cookbook](https://docs.sglang.io/cookbook/autoregressive/ThinkingMachines/Inkling-Small)
specifies Marlin FP4/MoE for DGX Spark, which correctly activates SM121
Blackwell tensor cores (HMMA/UMMA instructions in the `sm_121a` code path).

## Configuration reference

See [SERVE.md](SERVE.md) for the full verified configuration.

## Evidence

- `evidence/BENCHMARK.md` — full 11-lane results with MTP comparison
- `evidence/r0b0bench/` — raw lane JSON outputs (sanitized)
- `VERDICT.md` — serving verdict and runtime identity

## Credits

- Model: [Thinking Machines — Inkling-Small-NVFP4](https://huggingface.co/thinkingmachines/Inkling-Small-NVFP4)
- Runtime: [SGLang](https://github.com/sgl-project/sglang) day-0 support
- Benchmark: [r0b0bench](https://github.com/r0b0tlab/r0b0bench) core-subset
- Hardware: NVIDIA DGX Spark (GB10 / SM121)
- Results ledger: [r0b0bench results](https://github.com/r0b0tlab/r0b0bench/blob/main/results/LEADERBOARD.md)
