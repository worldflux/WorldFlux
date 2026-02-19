"""Tests for validate_matrix_completeness.py."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _manifest(path: Path) -> Path:
    payload = {
        "schema_version": "parity.manifest.v1",
        "defaults": {"alpha": 0.05, "equivalence_margin": 0.05},
        "seed_policy": {
            "mode": "fixed",
            "values": [0],
            "pilot_seeds": 10,
            "min_seeds": 20,
            "max_seeds": 50,
            "power_target": 0.8,
        },
        "tasks": [
            {
                "task_id": "atari100k_pong",
                "family": "dreamerv3",
                "required_metrics": ["final_return_mean", "auc_return"],
                "official": {
                    "adapter": "official_dreamerv3",
                    "cwd": ".",
                    "command": ["python3", "-c", "print('ok')"],
                    "env": {},
                },
                "worldflux": {
                    "adapter": "worldflux_dreamerv3_native",
                    "cwd": ".",
                    "command": ["python3", "-c", "print('ok')"],
                    "env": {},
                },
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def test_validate_matrix_completeness_fails_when_pair_missing(tmp_path: Path) -> None:
    root = _repo_root()
    manifest_path = _manifest(tmp_path / "manifest.json")

    runs_path = tmp_path / "runs.jsonl"
    rows = [
        {
            "schema_version": "parity.v1",
            "task_id": "atari100k_pong",
            "seed": 0,
            "system": "official",
            "status": "success",
            "metrics": {"final_return_mean": 1.0, "auc_return": 1.0},
        }
    ]
    runs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    output = tmp_path / "coverage_report.json"
    cmd = [
        "python3",
        "scripts/parity/validate_matrix_completeness.py",
        "--manifest",
        str(manifest_path),
        "--runs",
        str(runs_path),
        "--output",
        str(output),
        "--seeds",
        "0",
        "--max-missing-pairs",
        "0",
    ]

    completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
    assert completed.returncode == 1

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["missing_pairs"] == 1
    assert report["pass"] is False


def test_validate_matrix_completeness_writes_rerun_manifest(tmp_path: Path) -> None:
    root = _repo_root()
    manifest_path = _manifest(tmp_path / "manifest.json")

    runs_path = tmp_path / "runs.jsonl"
    rows = [
        {
            "schema_version": "parity.v1",
            "task_id": "atari100k_pong",
            "seed": 0,
            "system": "official",
            "status": "success",
            "metrics": {"final_return_mean": 1.0, "auc_return": 1.0},
        }
    ]
    runs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    output = tmp_path / "coverage_report.json"
    rerun_manifest = tmp_path / "rerun_manifest.json"
    cmd = [
        "python3",
        "scripts/parity/validate_matrix_completeness.py",
        "--manifest",
        str(manifest_path),
        "--runs",
        str(runs_path),
        "--output",
        str(output),
        "--seeds",
        "0",
        "--max-missing-pairs",
        "0",
        "--rerun-manifest-output",
        str(rerun_manifest),
    ]

    completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
    assert completed.returncode == 1
    assert rerun_manifest.exists()

    rerun_payload = json.loads(rerun_manifest.read_text(encoding="utf-8"))
    assert rerun_payload["seed_policy"]["mode"] == "fixed"
    assert rerun_payload["seed_policy"]["values"] == [0]
    assert len(rerun_payload["tasks"]) == 1
    assert rerun_payload["tasks"][0]["task_id"] == "atari100k_pong"
