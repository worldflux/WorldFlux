"""Tests for two-stage proof wait guard in AWS parity orchestrator."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "parity"
        / "aws_distributed_orchestrator.py"
    )
    spec = importlib.util.spec_from_file_location("aws_distributed_orchestrator", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load aws_distributed_orchestrator")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_two_stage_proof_requires_wait() -> None:
    mod = _load_module()
    args = argparse.Namespace(wait=False)
    with pytest.raises(RuntimeError, match="two_stage_proof requires --wait"):
        mod._run_two_stage_proof(args)
