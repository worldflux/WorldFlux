# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for documentation information architecture policy."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_sidebars_expose_cpu_success_and_reference_guides():
    sidebars = _read("website/sidebars.ts")
    assert "'Getting Started'" in sidebars
    assert "'getting-started/installation'" in sidebars
    assert "'getting-started/quickstart'" in sidebars
    assert "'getting-started/cpu-success'" in sidebars
    assert "'Tutorials'" in sidebars
    assert "'tutorials/train-first-model'" in sidebars
    assert "'tutorials/dreamer-vs-tdmpc2'" in sidebars
    assert "'tutorials/reproduce-dreamer-tdmpc2'" in sidebars

    assert "'Reference'" in sidebars
    assert "'reference/benchmarks'" in sidebars
    assert "'reference/observation-action'" in sidebars
    assert "'reference/parity'" in sidebars
    assert "'reference/unified-comparison'" in sidebars
    assert "'reference/publishing'" in sidebars
    assert "'reference/wasr'" in sidebars


def test_docs_index_exposes_primary_paths_for_cpu_success_and_reference():
    index = _read("docs/index.md")
    assert "(getting-started/installation.md)" in index
    assert "(getting-started/quickstart.md)" in index
    assert "(getting-started/quickstart.md#4-choosing-a-model-in-worldflux-init)" in index
    assert "(getting-started/cpu-success.md)" in index
    assert "(tutorials/dreamer-vs-tdmpc2.md)" in index
    assert "(tutorials/reproduce-dreamer-tdmpc2.md)" in index
    assert "(api/factory.md)" in index
    assert "(reference/benchmarks.md)" in index
    assert "(reference/observation-action.md)" in index
    assert "(reference/unified-comparison.md)" in index
    assert "(reference/parity.md)" in index


def test_quickstart_points_to_cpu_path_and_api_guides():
    quickstart = _read("docs/getting-started/quickstart.md")
    assert "Choosing a Model in `worldflux init`" in quickstart
    assert "dreamer:ci" in quickstart
    assert "tdmpc2:ci" in quickstart
    assert "(cpu-success.md)" in quickstart
    assert "(../reference/observation-action.md)" in quickstart
    assert "(../api/factory.md)" in quickstart
    assert "(../api/training.md)" in quickstart
    assert "(../api/protocol.md)" in quickstart


def test_observation_action_page_links_model_choice_guidance():
    observation_action = _read("docs/reference/observation-action.md")
    assert (
        "(../getting-started/quickstart.md#4-choosing-a-model-in-worldflux-init)"
        in observation_action
    )
