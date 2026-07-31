from __future__ import annotations

from typing import Any


SAFE_KEYS = {"mem_fraction_static", "context_length", "chunk_size", "concurrency"}


def plan_tuning(
    *,
    mem_fraction_static: list[float] | None = None,
    context_lengths: list[int] | None = None,
    chunk_sizes: list[int] | None = None,
    concurrency: list[int] | None = None,
) -> list[dict[str, Any]]:
    fractions = mem_fraction_static or [0.85]
    contexts = context_lengths or [32768]
    chunks = chunk_sizes or [2048]
    concurrencies = concurrency or [1]
    if any(not 0.50 <= value <= 0.95 for value in fractions):
        raise ValueError("mem_fraction_static must stay in [0.50, 0.95]")
    if any(type(value) is not int or value <= 0 or value > 262144 for value in contexts + chunks + concurrencies):
        raise ValueError("tuning values exceed bounded contract")
    plans: list[dict[str, Any]] = []
    for fraction in fractions:
        for context in contexts:
            for chunk in chunks:
                for parallel in concurrencies:
                    plans.append({
                        "mem_fraction_static": fraction,
                        "context_length": context,
                        "chunk_size": chunk,
                        "concurrency": parallel,
                        "status": "NOT_RUN",
                    })
    return plans
