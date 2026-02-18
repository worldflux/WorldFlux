"""Tests for parity metric transform utilities."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "parity" / "metric_transforms.py"
    )
    spec = importlib.util.spec_from_file_location("metric_transforms", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load metric_transforms")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_paired_log_ratio_improves_when_worldflux_higher() -> None:
    mod = _load_module()
    effect = mod.transform_pair(
        transform="paired_log_ratio",
        official=100.0,
        worldflux=105.0,
        higher_is_better=True,
        eps=1e-8,
    )
    assert effect > 0.0


def test_normalized_difference_respects_directionality() -> None:
    mod = _load_module()
    effect = mod.transform_pair(
        transform="normalized_difference",
        official=10.0,
        worldflux=9.0,
        higher_is_better=False,
        eps=1e-8,
    )
    assert effect > 0.0


def test_equivalence_bounds_for_log_ratio() -> None:
    mod = _load_module()
    bounds = mod.equivalence_bounds(
        transform="paired_log_ratio",
        equivalence_margin=0.05,
        noninferiority_margin=0.05,
    )
    assert bounds.lower_equivalence == pytest.approx(math.log(0.95))
    assert bounds.upper_equivalence == pytest.approx(math.log(1.05))
    assert bounds.lower_noninferiority == pytest.approx(math.log(0.95))


def test_transform_rejects_unknown_name() -> None:
    mod = _load_module()
    with pytest.raises(ValueError, match="Unsupported effect transform"):
        mod.transform_pair(
            transform="unknown",
            official=1.0,
            worldflux=1.0,
            higher_is_better=True,
            eps=1e-8,
        )
