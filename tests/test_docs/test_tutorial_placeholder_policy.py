# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Checks for tutorial placeholder publication policy."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_placeholder_tutorials_are_published_and_link_to_supported_docs() -> None:
    policy = _read("docs/reference/tutorial-policy.md")
    assert "Placeholder tutorial pages remain published" in policy

    for path in (
        "docs/tutorials/dreamer-vs-tdmpc2.md",
        "docs/tutorials/reproduce-dreamer-tdmpc2.md",
    ):
        text = _read(path)
        assert "Tutorial Temporarily Unavailable" in text
        assert "getting-started/quickstart.md" in text or "../getting-started/quickstart.md" in text


def test_train_first_model_is_a_real_supported_tutorial() -> None:
    text = _read("docs/tutorials/train-first-model.md")
    assert "Tutorial Temporarily Unavailable" not in text
    assert "worldflux init" in text
    assert "worldflux train" in text
    assert "worldflux verify --target ./outputs --mode quick" in text
    assert "inference.py" in text
