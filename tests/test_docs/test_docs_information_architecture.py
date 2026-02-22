"""Tests for documentation information architecture policy."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_mkdocs_nav_exposes_cpu_success_and_reference_guides():
    mkdocs = _read("mkdocs.yml")
    assert "- Getting Started:" in mkdocs
    assert "    - Installation: getting-started/installation.md" in mkdocs
    assert "    - Quick Start: getting-started/quickstart.md" in mkdocs
    assert "    - CPU Success Path: getting-started/cpu-success.md" in mkdocs

    # Core Concepts (Legacy) and Tutorials sections removed in nav cleanup
    assert "Core Concepts (Legacy)" not in mkdocs

    assert "- Reference:" in mkdocs
    assert "    - Benchmarks: reference/benchmarks.md" in mkdocs
    assert "    - Observation Shape and Action Dim: reference/observation-action.md" in mkdocs
    assert "    - Parity Harness: reference/parity.md" in mkdocs
    assert "    - Unified Comparison: reference/unified-comparison.md" in mkdocs
    assert "    - Publishing: reference/publishing.md" in mkdocs
    assert "    - WASR Metrics: reference/wasr.md" in mkdocs


def test_docs_index_exposes_primary_paths_for_cpu_success_and_reference():
    index = _read("docs/index.md")
    assert "(getting-started/installation.md)" in index
    assert "(getting-started/quickstart.md)" in index
    assert "(getting-started/quickstart.md#4-choosing-a-model-in-worldflux-init)" in index
    assert "(getting-started/cpu-success.md)" in index
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
