# Benchmark contract

Benchmark execution is not part of this non-live foundation. The wrapper
renders a plan and refuses `--execute`. A future runner must bind every row to
the exact runtime manifest, source commit, image digest, profile, endpoint
process epoch, request, and result bytes before producing eligible evidence.

The public schema requires exactly **1,764** rows for an eligible artifact. The
tracked plan is a count contract only; it contains no runtime measurements.
`NOT_RUN` scaffolding contains zero rows. Eligible, diagnostic, and disqualified
evidence are distinct statuses, and model failures must not be rewritten as
infrastructure failures.

## Offline commands

```text
python3 scripts/bench_serving.py \
  --profile profiles/inkling-small-nvfp4.json \
  --node-rank 0 --dist-init-addr rank0.example:5000
python3 scripts/tuning_driver.py --mem-fraction-static 0.85
```

These commands emit render-only JSON. They do not contact an endpoint, start a
container, download a model, or publish a result.
