# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for the S-grade program status collector."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "collect_s_grade_program_status.py"
    )
    spec = importlib.util.spec_from_file_location("collect_s_grade_program_status", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load collect_s_grade_program_status module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_s_grade_program_status_reports_missing_and_present_artifacts(
    tmp_path: Path,
) -> None:
    mod = _load_module()
    monkey_repo = tmp_path / "repo"
    (monkey_repo / "docs" / "prd").mkdir(parents=True)
    (monkey_repo / "AGENTS.md").write_text("# agents\n", encoding="utf-8")

    required = ("AGENTS.md", "docs/prd/architecture-s-grade.md")
    report = mod.collect_status(monkey_repo, required_paths=required)

    assert report["summary"]["present"] == 1
    assert report["summary"]["missing"] == 1
    assert report["artifacts"][0]["path"] == "AGENTS.md"
    assert report["artifacts"][0]["present"] is True
    assert report["artifacts"][1]["path"] == "docs/prd/architecture-s-grade.md"
    assert report["artifacts"][1]["present"] is False


def test_main_writes_json_and_exits_zero_even_when_artifacts_are_missing(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_module()
    repo_root = tmp_path / "repo"
    output_path = tmp_path / "report.json"
    repo_root.mkdir()

    monkeypatch.setattr(mod, "REQUIRED_ARTIFACTS", ("AGENTS.md",))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_s_grade_program_status.py",
            "--repo-root",
            str(repo_root),
            "--output-json",
            str(output_path),
        ],
    )

    exit_code = mod.main()
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["summary"]["missing"] == 1
    assert payload["artifacts"][0]["path"] == "AGENTS.md"


def test_collect_only_workflow_uploads_program_status_artifact() -> None:
    workflow = (
        Path(__file__).resolve().parents[2] / ".github" / "workflows" / "s-grade-collect.yml"
    ).read_text(encoding="utf-8")

    assert "Collect S-Grade Program Status" in workflow
    assert "scripts/collect_s_grade_program_status.py" in workflow
    assert "actions/upload-artifact" in workflow
    assert "s-grade-program-status" in workflow
