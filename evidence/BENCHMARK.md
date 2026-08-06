<<<<<<< HEAD
# Benchmark Results — r0b0bench core-subset

Run ID: `inkling-small-sm121-marlin-20260806`
Date: 2026-08-06
Duration: 10,605 seconds (~2.9 hours)
=======
# Benchmark Results — r0b0bench core-subset (11 lanes)

Run ID: `inkling-small-sm121-marlin-full-20260806`
Date: 2026-08-06
>>>>>>> origin/publication/marlin-sm121

## Summary

| Lane | Status | Key Metric |
|------|--------|-----------|
<<<<<<< HEAD
| canary | PASS | 5/5 checks (identity, needle, structured, tool_call, zh_arithmetic) |
| BFCL-MT | PASS | 108/200 = **54.0%** accuracy (official BFCL v4 multi_turn_base) |
| BFCL-AST | PASS | 181/600 = **30.2%** micro accuracy (multiple, parallel, parallel_multiple) |
| latency | PASS | c1 streaming 128 tokens: ~8.5s e2e |
| concurrency | PASS | c1=14.1, c4=46.8, c6=48.7 aggregate tok/s |
| throughput | PASS | decode 13.8 tok/s median, prefill 14,109 tok/s |
| NIAH | ERROR | 262K PASS, 524K HTTP 400 (physical KV capacity limit) |
=======
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
>>>>>>> origin/publication/marlin-sm121

## BFCL Multi-Turn (multi_turn_base)

Official BFCL v4, 200 cases, 0 infrastructure errors.
<<<<<<< HEAD

- Correct: 108
- Total: 200
- Accuracy: **54.0%**
=======
Accuracy: **54.0%** (108/200)
>>>>>>> origin/publication/marlin-sm121

## BFCL AST (non-live)

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| multiple | 38 | 200 | 19.0% |
| parallel | 105 | 200 | 52.5% |
| parallel_multiple | 38 | 200 | 19.0% |
| **micro** | **181** | **600** | **30.2%** |

<<<<<<< HEAD
=======
## Quality

- QA (ARC-Easy): **24.5%** (98/400)
- IFEval: **42.5%**
- HumanEval pass@1: **73.2%** (120/164)
- GSM8K (0-shot): **81.0%** (162/200)

>>>>>>> origin/publication/marlin-sm121
## Throughput

- c1 decode (2048 tokens): **13.8 tok/s** median
- c1 prefill (~22K prompt): **14,109 tok/s** (warm cache)

<<<<<<< HEAD
## Concurrency Ladder
=======
## Concurrency
>>>>>>> origin/publication/marlin-sm121

| Concurrency | Aggregate tok/s | Per-request tok/s |
|-------------|----------------|-------------------|
| 1 | 14.1 | 14.1 |
| 2 | 26.7 | 13.3 |
| 4 | 46.8 | 11.7 |
| 6 | 48.7 | 8.1 |

<<<<<<< HEAD
Scaling plateaus at c4-c6 due to single-GPU-per-node TP=2 overhead.

## NIAH

- max_model_len (from /v1/models): 1,048,576
- Depths tested: 25% (262,128), 50% (524,256), 90% (943,660)
- 262K: **PASS** (correctly retrieved needle, 372s)
- 524K: ERROR (HTTP 400 — physical KV cache capacity exceeded)

The NIAH failure at 50%+ context is an infrastructure capacity limit
(2x 128GB GB10, mem_fraction_static=0.85), not a model correctness issue.
The 262K depth passed with correct needle retrieval.
=======
## NIAH

- max_model_len: 1,048,576
- 25% (262K): **PASS**
- 50% (524K): ERROR (KV capacity)
- 90% (944K): not reached
>>>>>>> origin/publication/marlin-sm121
