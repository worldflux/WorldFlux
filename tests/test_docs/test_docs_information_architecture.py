"""Tests for docs information architecture policy."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_mkdocs_uses_minimal_onboarding_nav_without_tutorial_comparison_pages():
    mkdocs = _read("mkdocs.yml")
    assert "- Getting Started:" in mkdocs
    assert "    - Installation: getting-started/installation.md" in mkdocs
    assert "    - Quick Start: getting-started/quickstart.md" in mkdocs
    assert "Core Concepts" not in mkdocs
    assert "Train Your First Model" not in mkdocs
    assert "DreamerV3 vs TD-MPC2" not in mkdocs
    assert "Reproduce Dreamer/TD-MPC2" not in mkdocs
    assert "Investor Overview" not in mkdocs


def test_docs_index_exposes_only_neutral_primary_navigation_cards():
    index = _read("docs/index.md")
    assert "(getting-started/installation.md)" in index
    assert "(getting-started/quickstart.md)" in index
    assert "(api/factory.md)" in index
    assert "(tutorials/dreamer-vs-tdmpc2.md)" not in index
    assert "(tutorials/reproduce-dreamer-tdmpc2.md)" not in index


def test_quickstart_stays_minimal_and_api_oriented_without_reproduction_links():
    quickstart = _read("docs/getting-started/quickstart.md")
    assert "Train a Model" not in quickstart
    assert "(concepts.md)" not in quickstart
    assert "(../tutorials/train-first-model.md)" not in quickstart
    assert "(../tutorials/reproduce-dreamer-tdmpc2.md)" not in quickstart
    assert "(../api/factory.md)" in quickstart
    assert "(../api/training.md)" in quickstart
