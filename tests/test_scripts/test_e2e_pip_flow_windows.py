# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Checks for Windows newcomer e2e smoke wiring."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_ci_runs_pip_flow_smoke_on_windows() -> None:
    ci = _read(".github/workflows/ci.yml")
    assert "windows-latest" in ci
    assert "scripts/e2e/pip_flow_smoke.ps1" in ci


def test_windows_pip_flow_script_exists_and_uses_quick_verify() -> None:
    script = _read("scripts/e2e/pip_flow_smoke.ps1")
    assert "worldflux init demo --force" in script
    assert "worldflux train --steps 2 --device cpu" in script
    assert "worldflux verify --target ./outputs --mode quick" in script
