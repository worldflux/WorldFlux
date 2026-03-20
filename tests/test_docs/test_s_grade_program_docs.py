# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for the S-grade program documentation structure."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PRD_FILES = (
    "docs/prd/architecture-s-grade.md",
    "docs/prd/api-s-grade.md",
    "docs/prd/ml-correctness-s-grade.md",
    "docs/prd/code-quality-s-grade.md",
    "docs/prd/differentiation-s-grade.md",
    "docs/prd/production-maturity-s-grade.md",
    "docs/prd/oss-readiness-s-grade.md",
    "docs/prd/scalability-s-grade.md",
)

PRD_REQUIRED_KEYS = (
    "id:",
    "title:",
    "owner:",
    "status:",
    "priority:",
    "target_window:",
    "depends_on:",
    "in_scope:",
    "out_of_scope:",
    "allowed_paths:",
    "blocked_paths:",
    "verification_commands:",
    "acceptance_gates:",
)

PRD_REQUIRED_SECTIONS = (
    "## 1. Problem",
    "## 2. Why Now",
    "## 3. S-Level End State",
    "## 4. 90-Day Target",
    "## 5. Requirements",
    "## 6. Non-Goals",
    "## 7. Implementation Plan",
    "## 8. Verification",
    "## 9. Rollout and Rollback",
    "## 10. Open Questions",
)


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_s_grade_program_docs_exist_and_reference_required_phase_zero_artifacts() -> None:
    agents = _read("AGENTS.md")
    roadmap = _read("docs/roadmap/2026-q2-worldflux-quality-program.md")
    task_docs = _read("docs/tasks/README.md")
    docs_roadmap = _read("docs/roadmap.md")
    contributing = _read("CONTRIBUTING.md")
    governance = _read("GOVERNANCE.md")

    assert "allowed_paths" in agents
    assert "blocked_paths" in agents
    assert "verification_commands" in agents
    assert "done_when" in agents

    assert "Phase 0: Program Setup" in roadmap
    assert "Phase 1: Correctness and Contract Recovery" in roadmap
    assert "collect-only CI gate" in roadmap
    assert "release metadata alignment" in roadmap

    assert "docs/tasks/" in task_docs
    assert "task_id:" in task_docs
    assert "allowed_paths:" in task_docs
    assert "done_when:" in task_docs

    assert "docs/roadmap/2026-q2-worldflux-quality-program.md" in docs_roadmap
    assert "docs/roadmap/2026-q2-worldflux-quality-program.md" in contributing
    assert "docs/roadmap/2026-q2-worldflux-quality-program.md" in governance


def test_all_s_grade_prds_follow_the_required_template_shape() -> None:
    for path in PRD_FILES:
        content = _read(path)
        assert content.startswith("---\n"), path
        for key in PRD_REQUIRED_KEYS:
            assert key in content, f"{path} missing {key}"
        for section in PRD_REQUIRED_SECTIONS:
            assert section in content, f"{path} missing {section}"
