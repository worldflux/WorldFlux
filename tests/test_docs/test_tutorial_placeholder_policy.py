# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Checks for public tutorial readiness policy."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_tutorial_rollout_policy_requires_real_content_before_promotion() -> None:
    policy = _read("docs/reference/tutorial-policy.md")
    assert "Promoted tutorials must contain runnable guidance" in policy
    assert "Current Placeholder Tutorials" not in policy


def test_promoted_tutorials_are_not_placeholders() -> None:
    comparison = _read("docs/tutorials/dreamer-vs-tdmpc2.md")
    reproduction = _read("docs/tutorials/reproduce-dreamer-tdmpc2.md")

    assert "Tutorial Temporarily Unavailable" not in comparison
    assert "worldflux models info dreamer:ci" in comparison
    assert "worldflux models info tdmpc2:ci" in comparison
    assert "worldflux verify --target ./outputs --mode quick" in comparison

    assert "Tutorial Temporarily Unavailable" not in reproduction
    assert "examples/quickstart_cpu_success.py --quick" in reproduction
    assert "worldflux parity" in reproduction


def test_train_first_model_is_a_real_supported_tutorial() -> None:
    text = _read("docs/tutorials/train-first-model.md")
    assert "Tutorial Temporarily Unavailable" not in text
    assert "worldflux init" in text
    assert "worldflux train" in text
    assert "worldflux verify --target ./outputs --mode quick" in text
    assert "inference.py" in text
