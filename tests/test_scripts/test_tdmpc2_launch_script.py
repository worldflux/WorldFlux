# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Checks for TD-MPC2 parity launch helper wiring."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_tdmpc2_launch_script_exists_and_targets_proof_profile() -> None:
    script = _read("scripts/parity/launch_tdmpc2_parity.sh")
    assert "aws_distributed_orchestrator.py" in script
    assert "proof_5m" in script
    assert "two_stage_proof" in script
