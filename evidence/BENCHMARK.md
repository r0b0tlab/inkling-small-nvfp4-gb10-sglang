# Benchmark Results — r0b0bench core-subset

## Baseline (no speculative) — 11 lanes, all PASS

| Lane | Result |
|------|--------|
| canary | 5/5 PASS |
| BFCL-MT | **54.0%** (108/200) |
| BFCL-AST | **30.2%** micro (181/600) |
| latency | c1 128-token ~8.5s |
| concurrency | c4 = 46.8 tok/s |
| throughput | decode **13.8 tok/s**, prefill **14,109 tok/s** |
| NIAH | **3/3 PASS** (61K/122K/220K at 245K context) |
| QA (ARC-Easy) | **24.5%** (98/400) |
| IFEval | **42.5%** |
| HumanEval | **73.2%** pass@1 (120/164) |
| GSM8K | **81.0%** (162/200) |

Context length constrained to 245,000 tokens (physical KV ceiling ~261K).
Model advertises 1M natively but 2x 128GB GB10 cannot hold enough BF16 KV
beyond ~261K tokens. NIAH generation_reserve=256 (64 insufficient for reasoning).

## MTP 8-1-9 (speculative) — 10 lanes PASS, NIAH not possible

| Lane | Baseline | MTP | Delta |
|------|---------:|----:|------:|
| BFCL-MT | 54.0% | **57.5%** | +3.5% |
| BFCL-AST | 30.2% | 30.2% | ~ |
| QA | 24.5% | 23.8% | -0.8% |
| IFEval | 42.5% | **48.0%** | +5.5% |
| HumanEval | 73.2% | 71.3% | -1.8% |
| GSM8K | 81.0% | 77.0% | -4.0% |
| decode tok/s | 13.8 | **15.8** | +1.15x |
| c1 aggregate | 14.1 | **16.4** | +16% |
| c6 aggregate | 48.7 | **63.2** | +30% |
| NIAH | 3/3 PASS | N/A | MTP KV=8K |

MTP improves BFCL-MT (+3.5%), IFEval (+5.5%), and decode throughput (+15%),
but slightly degrades GSM8K (-4%) and HumanEval (-1.8%). MTP's 8 draft layers
reduce KV capacity from 261K to 8K tokens, making NIAH impossible.
