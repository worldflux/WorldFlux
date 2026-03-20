# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for release metadata alignment across public docs."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_readme_points_to_current_release_authority_and_program_roadmap() -> None:
    readme = _read("README.md")

    assert "docs/reference/release-checklist.md" in readme
    assert "docs/operations/release-runbook.md" in readme
    assert "docs/roadmap/2026-q2-worldflux-quality-program.md" in readme
    assert "[ROADMAP.md](ROADMAP.md)" not in readme


def test_release_checklist_declares_metadata_authority_sources() -> None:
    checklist = _read("docs/reference/release-checklist.md")

    assert "Release authority sources:" in checklist
    assert "`pyproject.toml`" in checklist
    assert "`CHANGELOG.md`" in checklist
    assert "`docs/operations/release-runbook.md`" in checklist
    assert "`docs/roadmap/2026-q2-worldflux-quality-program.md`" in checklist


def test_pyproject_urls_include_release_checklist_and_current_program_roadmap() -> None:
    pyproject = tomllib.loads(_read("pyproject.toml"))
    urls = pyproject["project"]["urls"]

    assert urls["Release Checklist"].endswith("/docs/reference/release-checklist.md")
    assert urls["Roadmap"].endswith("/docs/roadmap/2026-q2-worldflux-quality-program.md")
