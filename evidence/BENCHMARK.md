# Benchmark Results — r0b0bench core-subset (11 lanes)

Run date: 2026-08-06/07
Hardware: 2x NVIDIA DGX Spark (GB10/SM121), TP=2, socket NCCL

## Quality

| Lane | Status | Result |
|------|--------|--------|
| canary | PASS | 5/5 checks |
| BFCL-MT | PASS | **54.0%** (108/200, official BFCL v4 multi_turn_base) |
| BFCL-AST | PASS | **30.2%** micro (181/600) |
| QA (ARC-Easy) | PASS | **24.5%** (98/400) |
| IFEval | PASS | **42.5%** |
| HumanEval | PASS | **73.2%** pass@1 (120/164) |
| GSM8K | PASS | **81.0%** (162/200, 0-shot) |

## Performance — Baseline (no speculative)

| Lane | Status | Result |
|------|--------|--------|
| latency | PASS | c1 128-token: ~8.5s e2e (stream) |
| concurrency | PASS | c1=14.1, c2=26.7, c4=46.8, c6=48.7 tok/s aggregate |
| throughput | PASS | decode **13.8 tok/s** median, prefill **14,109 tok/s** |
| NIAH | ERROR | 262K PASS, 524K HTTP 400 (KV capacity at mem_fraction_static=0.85) |

## Performance — MTP 8-1-9 (speculative comparison)

| Metric | Baseline | MTP 8-1-9 | Delta |
|--------|---------:|----------:|------:|
| c1 decode (2048 tok) | 13.8 tok/s | 16.0 tok/s | +1.16x |
| prefill (~22K prompt) | 14,109 tok/s | 11,736 tok/s | -17% |
| c1 concurrency | 14.1 | 13.7 | -3% |
| c2 concurrency | 26.7 | 29.3 | +10% |
| c4 concurrency | 46.8 | 47.4 | +1% |
| c6 concurrency | 48.7 | 47.2 | -3% |
| c1 latency (128 tok) | 8,537 ms | 12,042 ms | +41% |

MTP provides marginal decode benefit (+16%) but degrades latency and prefill.
Not recommended for 2-node TP=2 deployments due to cross-node draft verification
overhead and reduced CUDA graph batch coverage ([1,2,4,5] vs [1...95]).

## BFCL AST breakdown

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| multiple | 38 | 200 | 19.0% |
| parallel | 105 | 200 | 52.5% |
| parallel_multiple | 38 | 200 | 19.0% |
| **micro** | **181** | **600** | **30.2%** |

## NIAH

- max_model_len (from /v1/models): 1,048,576
- 25% (262,128 tokens): **PASS** (correct needle retrieval, 372s)
- 50% (524,256 tokens): ERROR (HTTP 400, physical KV cache exceeded)
- 90% (943,660 tokens): not reached

The NIAH failure at 50%+ is an infrastructure capacity limit (2x 128GB GB10,
mem_fraction_static=0.85), not a model correctness issue.

## Tensor core verification

- GPU: NVIDIA GB10, capability (12, 1) = SM121
- sgl-kernel `sm100/common_ops.abi3.so` fatbin ELF contains `sm_121a` code
- sm_121a section: 6,156 HMMA + 28 UMMA instructions across 1,098 kernels
- Runtime: 96% GPU utilization at 2,418 MHz during decode
