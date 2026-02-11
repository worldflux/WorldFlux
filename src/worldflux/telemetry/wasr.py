"""Local JSONL telemetry helpers for WASR-style product metrics."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

REQUIRED_EVENT_FIELDS = (
    "event",
    "timestamp",
    "run_id",
    "scenario",
    "success",
    "duration_sec",
    "ttfi_sec",
    "artifacts",
    "error",
)
OPTIONAL_EVENT_FIELDS = (
    "epoch",
    "step",
    "throughput_steps_per_sec",
    "flops_estimate",
    "watts_estimate",
    "flops_per_watt",
    "suggestions",
)


def _coerce_artifacts(artifacts: dict[str, str] | None) -> dict[str, str]:
    if artifacts is None:
        return {}
    return {str(k): str(v) for k, v in artifacts.items()}


def _coerce_optional_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _coerce_optional_int(value: int | None) -> int | None:
    if value is None:
        return None
    return int(value)


def default_metrics_path() -> Path:
    """Return metrics path, preferring explicit env override."""
    override = os.environ.get("WORLDFLUX_METRICS_PATH", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (Path.cwd() / ".worldflux" / "metrics.jsonl").resolve()


def make_run_id() -> str:
    """Generate a stable-enough run id for local instrumentation."""
    return uuid.uuid4().hex


def write_event(
    *,
    event: str,
    scenario: str,
    success: bool,
    duration_sec: float,
    ttfi_sec: float,
    artifacts: dict[str, str] | None = None,
    error: str | None = None,
    run_id: str | None = None,
    timestamp: float | None = None,
    path: str | Path | None = None,
    epoch: int | None = None,
    step: int | None = None,
    throughput_steps_per_sec: float | None = None,
    flops_estimate: float | None = None,
    watts_estimate: float | None = None,
    flops_per_watt: float | None = None,
    suggestions: list[str] | None = None,
) -> dict[str, Any]:
    """Append one telemetry event as JSON Lines and return the event payload."""
    payload: dict[str, Any] = {
        "event": str(event),
        "timestamp": float(timestamp if timestamp is not None else time.time()),
        "run_id": str(run_id or make_run_id()),
        "scenario": str(scenario),
        "success": bool(success),
        "duration_sec": float(duration_sec),
        "ttfi_sec": float(ttfi_sec),
        "artifacts": _coerce_artifacts(artifacts),
        "error": str(error) if error else "",
        "epoch": _coerce_optional_int(epoch),
        "step": _coerce_optional_int(step),
        "throughput_steps_per_sec": _coerce_optional_float(throughput_steps_per_sec),
        "flops_estimate": _coerce_optional_float(flops_estimate),
        "watts_estimate": _coerce_optional_float(watts_estimate),
        "flops_per_watt": _coerce_optional_float(flops_per_watt),
        "suggestions": [str(item) for item in suggestions] if suggestions else [],
    }

    target = Path(path) if path is not None else default_metrics_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":")) + "\n")

    return payload


def read_events(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load telemetry events from JSONL."""
    target = Path(path) if path is not None else default_metrics_path()
    if not target.exists():
        return []

    events: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                events.append(item)
    return events
