# Benchmark Results — r0b0bench core-subset (11 lanes)

Run ID: `inkling-small-sm121-marlin-full-20260806`
Date: 2026-08-06

## Summary

| Lane | Status | Key Metric |
|------|--------|-----------|
| canary | PASS | 5/5 checks |
| BFCL-MT | PASS | 108/200 = **54.0%** |
| BFCL-AST | PASS | 181/600 = **30.2%** micro |
| latency | PASS | c1 128-token ~8.5s e2e |
| concurrency | PASS | c4 = 46.8 tok/s aggregate |
| throughput | PASS | decode **13.8 tok/s**, prefill **14,109 tok/s** |
| NIAH | ERROR | 262K PASS, 524K KV capacity limit |
| QA | PASS | **24.5%** (98/400 ARC-Easy) |
| IFEval | PASS | **42.5%** |
| HumanEval | PASS | **73.2%** pass@1 (120/164) |
| GSM8K | PASS | **81.0%** (162/200, 0-shot) |

10 PASS, 1 ERROR (NIAH infra capacity, not correctness).

## BFCL Multi-Turn (multi_turn_base)

Official BFCL v4, 200 cases, 0 infrastructure errors.
Accuracy: **54.0%** (108/200)

## BFCL AST (non-live)

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| multiple | 38 | 200 | 19.0% |
| parallel | 105 | 200 | 52.5% |
| parallel_multiple | 38 | 200 | 19.0% |
| **micro** | **181** | **600** | **30.2%** |

## Quality

- QA (ARC-Easy): **24.5%** (98/400)
- IFEval: **42.5%**
- HumanEval pass@1: **73.2%** (120/164)
- GSM8K (0-shot): **81.0%** (162/200)

## Throughput

- c1 decode (2048 tokens): **13.8 tok/s** median
- c1 prefill (~22K prompt): **14,109 tok/s** (warm cache)

## Concurrency

| Concurrency | Aggregate tok/s | Per-request tok/s |
|-------------|----------------|-------------------|
| 1 | 14.1 | 14.1 |
| 2 | 26.7 | 13.3 |
| 4 | 46.8 | 11.7 |
| 6 | 48.7 | 8.1 |

## NIAH

- max_model_len: 1,048,576
- 25% (262K): **PASS**
- 50% (524K): ERROR (KV capacity)
- 90% (944K): not reached
