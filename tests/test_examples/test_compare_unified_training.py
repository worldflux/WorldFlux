# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Regression tests for the public unified comparison demo."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_demo(*args: str, output_dir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = (
        src_path if not current_pythonpath else f"{src_path}{os.pathsep}{current_pythonpath}"
    )
    cmd = [
        sys.executable,
        "examples/compare_unified_training.py",
        "--quick",
        "--output-dir",
        str(output_dir),
        *args,
    ]
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )


def test_compare_unified_training_includes_verification_results_by_default(
    tmp_path: Path,
) -> None:
    completed = _run_demo(output_dir=tmp_path)
    assert completed.returncode == 0, completed.stderr

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["success"] is True
    assert summary["models"]["dreamerv3"]["verification"]["target"]
    assert summary["models"]["dreamerv3"]["verification"]["verdict_reason"]
    assert summary["models"]["tdmpc2"]["verification"]["target"]
    assert summary["models"]["tdmpc2"]["verification"]["verdict_reason"]


def test_compare_unified_training_allows_skipping_verification(tmp_path: Path) -> None:
    completed = _run_demo("--skip-verify", output_dir=tmp_path)
    assert completed.returncode == 0, completed.stderr

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert "verification" not in summary["models"]["dreamerv3"]
    assert "verification" not in summary["models"]["tdmpc2"]
