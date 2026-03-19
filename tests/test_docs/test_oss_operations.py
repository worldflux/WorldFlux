# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for OSS operating documentation."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_governance_and_maintainers_reference_operational_runbooks() -> None:
    governance = _read("GOVERNANCE.md")
    maintainers = _read("MAINTAINERS.md")

    assert "docs/operations/release-runbook.md" in governance
    assert "docs/operations/maintainer-onboarding.md" in maintainers


def test_contributing_and_governance_reference_roadmap_and_release_process() -> None:
    contributing = _read("CONTRIBUTING.md")
    governance = _read("GOVERNANCE.md")

    assert "docs/roadmap.md" in contributing
    assert "release-runbook" in governance
