"""Tests for WASR summary computation script."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "compute_wasr.py"
    spec = importlib.util.spec_from_file_location("compute_wasr", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load compute_wasr module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_wasr_counts_unique_successful_runs_within_window() -> None:
    mod = _load_module()
    now = 1_760_100_000.0

    events = [
        {
            "event": "run_complete",
            "timestamp": now - 3600,
            "run_id": "a",
            "scenario": "quickstart_cpu",
            "success": True,
        },
        {
            "event": "run_complete",
            "timestamp": now - 1800,
            "run_id": "b",
            "scenario": "comparison_unified_training",
            "success": True,
        },
        {
            "event": "run_complete",
            "timestamp": now - 100,
            "run_id": "b",
            "scenario": "comparison_unified_training",
            "success": True,
        },
        {
            "event": "run_complete",
            "timestamp": now - 500,
            "run_id": "c",
            "scenario": "quickstart_cpu",
            "success": False,
        },
    ]

    result = mod.compute_wasr(events, now=now, lookback_days=7)
    assert result["wasr"] == 2
    assert result["quickstart"]["attempts"] == 2
    assert result["quickstart"]["successes"] == 1
    assert result["quickstart"]["success_rate"] == 0.5


def test_compute_wasr_returns_zero_when_no_events() -> None:
    mod = _load_module()
    result = mod.compute_wasr([], now=1_760_100_000.0, lookback_days=7)
    assert result["wasr"] == 0
    assert result["quickstart"]["attempts"] == 0
    assert result["quickstart"]["success_rate"] == 0.0
