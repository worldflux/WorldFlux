"""Tests for Bayesian parity statistics helper."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "parity" / "stats_bayesian.py"
    spec = importlib.util.spec_from_file_location("stats_bayesian", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load stats_bayesian module")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_bayesian_report_is_deterministic_for_fixed_seed() -> None:
    mod = _load_module()
    effects = [0.0, 0.005, -0.002, 0.004, 0.001]
    kwargs = {
        "effects": effects,
        "draws": 4000,
        "seed": 20260220,
        "lower_equivalence": math.log(0.95),
        "upper_equivalence": math.log(1.05),
        "lower_noninferiority": math.log(0.95),
        "probability_threshold_equivalence": 0.95,
        "probability_threshold_noninferiority": 0.975,
        "min_pairs": 2,
    }
    first = mod.bayesian_equivalence_report(**kwargs)
    second = mod.bayesian_equivalence_report(**kwargs)
    assert first == second


def test_bayesian_report_passes_for_within_margin_effects() -> None:
    mod = _load_module()
    effects = [0.0, 0.01, 0.005, -0.005, 0.002, 0.004]
    report = mod.bayesian_equivalence_report(
        effects=effects,
        draws=5000,
        seed=42,
        lower_equivalence=math.log(0.95),
        upper_equivalence=math.log(1.05),
        lower_noninferiority=math.log(0.95),
        probability_threshold_equivalence=0.95,
        probability_threshold_noninferiority=0.975,
        min_pairs=2,
    )
    assert report["status"] == "ok"
    assert report["pass_equivalence"] is True
    assert report["pass_noninferiority"] is True
    assert report["pass_all"] is True


def test_bayesian_report_fails_for_large_underperformance() -> None:
    mod = _load_module()
    effects = [math.log(0.70), math.log(0.72), math.log(0.68), math.log(0.71)]
    report = mod.bayesian_equivalence_report(
        effects=effects,
        draws=5000,
        seed=7,
        lower_equivalence=math.log(0.95),
        upper_equivalence=math.log(1.05),
        lower_noninferiority=math.log(0.95),
        probability_threshold_equivalence=0.95,
        probability_threshold_noninferiority=0.975,
        min_pairs=2,
    )
    assert report["status"] == "ok"
    assert report["pass_equivalence"] is False
    assert report["pass_noninferiority"] is False
    assert report["pass_all"] is False


def test_bayesian_report_requires_min_pairs() -> None:
    mod = _load_module()
    report = mod.bayesian_equivalence_report(
        effects=[0.01],
        draws=1000,
        seed=1,
        lower_equivalence=math.log(0.95),
        upper_equivalence=math.log(1.05),
        lower_noninferiority=math.log(0.95),
        probability_threshold_equivalence=0.95,
        probability_threshold_noninferiority=0.975,
        min_pairs=2,
    )
    assert report["status"] == "insufficient_pairs"
