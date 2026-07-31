from __future__ import annotations

from typing import Any


REQUIRED_CANARY_KEYS = {"schema_version", "status", "request_id", "response", "checks"}


def validate_canary(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != REQUIRED_CANARY_KEYS:
        raise ValueError("canary result keys do not match schema")
    if value["schema_version"] != 1 or value["status"] not in {"NOT_RUN", "PASS", "FAIL"}:
        raise ValueError("invalid canary result status")
    if not isinstance(value["request_id"], str) or not value["request_id"]:
        raise ValueError("canary request_id is required")
    if not isinstance(value["response"], dict):
        raise ValueError("canary response must be an object")
    if not isinstance(value["checks"], dict) or any(type(v) is not bool for v in value["checks"].values()):
        raise ValueError("canary checks must be boolean")
    if value["status"] == "PASS" and (not value["checks"] or not all(value["checks"].values())):
        raise ValueError("passing canary requires every check to pass")
    return value


def render_canary_plan(*, endpoint: str = "http://127.0.0.1:30000", request_id: str = "placeholder") -> dict[str, Any]:
    if not endpoint.startswith("http://127.0.0.1:") and not endpoint.startswith("http://[::1]:"):
        raise ValueError("canary endpoint must be loopback")
    return {
        "schema_version": 1,
        "status": "NOT_RUN",
        "endpoint": endpoint,
        "request_id": request_id,
        "checks": {"health": False, "chat": False, "reasoning_parser": False, "tool_call_parser": False},
        "executes": False,
    }
