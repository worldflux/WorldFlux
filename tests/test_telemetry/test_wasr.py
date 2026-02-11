"""Tests for local WASR telemetry helpers."""

from __future__ import annotations

import json
from pathlib import Path

from worldflux.telemetry.wasr import (
    REQUIRED_EVENT_FIELDS,
    default_metrics_path,
    read_events,
    write_event,
)


def test_default_metrics_path_uses_env_override(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "custom" / "metrics.jsonl"
    monkeypatch.setenv("WORLDFLUX_METRICS_PATH", str(path))
    assert default_metrics_path() == path.resolve()


def test_write_event_and_read_events_roundtrip(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    payload = write_event(
        event="run_complete",
        scenario="quickstart_cpu",
        success=True,
        duration_sec=1.25,
        ttfi_sec=0.75,
        artifacts={"summary": "outputs/summary.json"},
        run_id="run-1",
        timestamp=1_760_000_000.0,
        path=metrics_path,
    )

    for key in REQUIRED_EVENT_FIELDS:
        assert key in payload

    events = read_events(metrics_path)
    assert len(events) == 1
    assert events[0]["event"] == "run_complete"
    assert events[0]["scenario"] == "quickstart_cpu"
    assert events[0]["success"] is True

    parsed = json.loads(metrics_path.read_text(encoding="utf-8").strip())
    assert parsed["run_id"] == "run-1"
