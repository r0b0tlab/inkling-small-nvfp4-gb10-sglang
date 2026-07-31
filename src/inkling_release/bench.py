from __future__ import annotations

from typing import Any

from .benchmark import EXPECTED_ROW_COUNT
from .render import render_command


def render_benchmark_plan(profile: dict[str, Any], *, node_rank: int, dist_init_addr: str) -> dict[str, Any]:
    rendered = render_command(profile, node_rank=node_rank, dist_init_addr=dist_init_addr)
    return {
        "schema_version": 1,
        "status": "NOT_RUN",
        "row_count_contract": EXPECTED_ROW_COUNT,
        "profile": profile["name"],
        "command": rendered,
        "rows": [],
        "result_rows_emitted": False,
        "infrastructure_failures": 0,
        "executes": False,
    }
