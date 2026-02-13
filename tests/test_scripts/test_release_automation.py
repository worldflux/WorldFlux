"""Tests for release automation helper scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module(script_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {script_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_release_metadata_normalizes_tags() -> None:
    mod = _load_module("check_release_metadata.py")
    assert mod._normalize_tag("v0.1.0") == "0.1.0"
    assert mod._normalize_tag("refs/tags/v0.1.0") == "0.1.0"
    assert mod._normalize_tag("0.1.0") == "0.1.0"


def test_generate_verification_report_parses_check_entries() -> None:
    mod = _load_module("generate_verification_report.py")
    parsed = mod._parse_check("docs_strict=pass")
    assert parsed["name"] == "docs_strict"
    assert parsed["status"] == "pass"


def test_generate_verification_report_renders_parity_suite_table() -> None:
    mod = _load_module("generate_verification_report.py")
    payload = {
        "generated_at_utc": "2026-02-12T00:00:00Z",
        "checks": [],
        "summary": {"pass": 0, "fail": 0, "skipped": 0},
        "parity": {
            "aggregate_present": True,
            "all_suites_pass": True,
            "suite_fail_count": 0,
            "suites": [
                {
                    "suite_id": "dreamer_atari100k",
                    "sample_size": 10,
                    "ci_upper_ratio": 0.01,
                    "margin_ratio": 0.05,
                    "pass_non_inferiority": True,
                    "verdict_reason": "PASS: ...",
                }
            ],
        },
    }
    markdown = mod._render_markdown(payload)
    assert "| Suite | Samples | Upper CI | Margin | Pass | Reason |" in markdown
    assert "dreamer_atari100k" in markdown


def test_release_checklist_gate_snippets_present_in_repo() -> None:
    mod = _load_module("check_release_checklist_gate.py")
    ci = (Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml").read_text(
        encoding="utf-8"
    )
    release = (Path(__file__).resolve().parents[2] / ".github/workflows/release.yml").read_text(
        encoding="utf-8"
    )
    checklist = (
        Path(__file__).resolve().parents[2] / "docs/reference/release-checklist.md"
    ).read_text(encoding="utf-8")
    for snippet in mod.REQUIRED_CI_SNIPPETS:
        assert snippet in ci
    for snippet in mod.REQUIRED_CHECKLIST_SNIPPETS:
        assert snippet in checklist
    for snippet in mod.REQUIRED_RELEASE_SNIPPETS:
        assert snippet in release
