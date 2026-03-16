"""Tests for TD-MPC2 alignment report generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _payload(*, model_id: str, model_profile: str) -> dict:
    return {
        "schema_version": "parity.v1",
        "metrics": {
            "metadata": {
                "model_id": model_id,
                "model_profile": model_profile,
                "train_budget": {"steps": 100},
                "eval_protocol": {"eval_interval": 6, "eval_episodes": 1, "eval_window": 2},
                "artifact_manifest": {"metrics_paths": ["metrics.json"]},
            }
        },
    }


def test_tdmpc2_alignment_report_marks_aligned_for_proof_profile(tmp_path: Path) -> None:
    root = _repo_root()
    official = tmp_path / "official.json"
    worldflux = tmp_path / "worldflux.json"
    output = tmp_path / "alignment.json"
    official.write_text(
        json.dumps(_payload(model_id="tdmpc2:5m", model_profile="5m")), encoding="utf-8"
    )
    worldflux.write_text(
        json.dumps(_payload(model_id="tdmpc2:proof_5m", model_profile="proof_5m")),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        "scripts/parity/tdmpc2_alignment_report.py",
        "--official-input",
        str(official),
        "--worldflux-input",
        str(worldflux),
        "--output",
        str(output),
    ]
    completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "aligned"


def test_tdmpc2_alignment_report_marks_mismatched_for_noncanonical_profile(tmp_path: Path) -> None:
    root = _repo_root()
    official = tmp_path / "official.json"
    worldflux = tmp_path / "worldflux.json"
    output = tmp_path / "alignment.json"
    official.write_text(
        json.dumps(_payload(model_id="tdmpc2:5m", model_profile="5m")), encoding="utf-8"
    )
    worldflux.write_text(
        json.dumps(_payload(model_id="tdmpc2:5m", model_profile="5m")), encoding="utf-8"
    )

    cmd = [
        sys.executable,
        "scripts/parity/tdmpc2_alignment_report.py",
        "--official-input",
        str(official),
        "--worldflux-input",
        str(worldflux),
        "--output",
        str(output),
    ]
    completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "mismatched"
