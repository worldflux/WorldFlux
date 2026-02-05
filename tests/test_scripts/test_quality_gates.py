"""Tests for quality gate summary schema."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_measure_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "measure_quality_gates.py"
    spec = importlib.util.spec_from_file_location("measure_quality_gates", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load measure_quality_gates module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_includes_common_confidence_interval_fields():
    mod = _load_measure_module()
    runs = [
        {
            "model": "dreamerv3:ci",
            "seed": 0,
            "steps": 100,
            "final_loss": 1.0,
            "loss_drop": 0.20,
            "is_finite": True,
            "is_success": True,
        },
        {
            "model": "dreamerv3:ci",
            "seed": 1,
            "steps": 100,
            "final_loss": 0.9,
            "loss_drop": 0.15,
            "is_finite": True,
            "is_success": True,
        },
        {
            "model": "dreamerv3:ci",
            "seed": 2,
            "steps": 100,
            "final_loss": 1.1,
            "loss_drop": 0.05,
            "is_finite": True,
            "is_success": False,
        },
    ]
    summary = mod._summary(runs)
    gates = summary["dreamerv3:ci"]["gates"]["common"]
    assert "ci_low" in gates
    assert "ci_high" in gates
    assert gates["ci_low"] <= gates["ci_high"]


def test_summary_includes_family_pass_flag():
    mod = _load_measure_module()
    runs = [
        {
            "model": "token:ci",
            "seed": 0,
            "steps": 100,
            "final_loss": 2.0,
            "loss_drop": 0.20,
            "is_finite": True,
            "is_success": True,
        },
        {
            "model": "token:ci",
            "seed": 1,
            "steps": 100,
            "final_loss": 1.8,
            "loss_drop": 0.22,
            "is_finite": True,
            "is_success": True,
        },
    ]
    summary = mod._summary(runs)
    assert "family_pass" in summary["token:ci"]["gates"]
    assert isinstance(summary["token:ci"]["gates"]["family_pass"], bool)
