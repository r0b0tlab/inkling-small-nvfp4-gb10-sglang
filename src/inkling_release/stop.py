from __future__ import annotations

import re

_NAME = re.compile(r"^inkling-sglang-tp2-rank[01]$")


def build_stop_argv(container_name: str, *, timeout_seconds: int = 30) -> list[str]:
    if not _NAME.fullmatch(container_name):
        raise ValueError("refusing to stop an unowned container name")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 120:
        raise ValueError("stop timeout is outside the bounded range")
    return ["docker", "stop", "--time", str(timeout_seconds), container_name]
