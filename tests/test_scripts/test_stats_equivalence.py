"""Tests for parity statistical equivalence script."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _record(task_id: str, seed: int, system: str, final_return: float, auc_return: float) -> dict:
    return {
        "schema_version": "parity.v1",
        "run_id": "test",
        "task_id": task_id,
        "family": "dreamerv3",
        "seed": seed,
        "system": system,
        "adapter": "dummy",
        "status": "success",
        "metrics": {
            "final_return_mean": final_return,
            "auc_return": auc_return,
        },
    }


def _run_stats(input_path: Path, output_path: Path) -> dict:
    root = _repo_root()
    cmd = [
        "python3",
        "scripts/parity/stats_equivalence.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--alpha",
        "0.05",
        "--equivalence-margin",
        "0.05",
        "--noninferiority-margin",
        "0.05",
        "--min-pairs",
        "2",
    ]
    completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    return json.loads(output_path.read_text(encoding="utf-8"))


def test_stats_equivalence_passes_when_ratios_within_margin(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.jsonl"
    rows: list[dict] = []
    for seed in range(20):
        official = 100.0 + seed * 0.1
        worldflux = official * 1.01
        rows.append(_record("atari100k_pong", seed, "official", official, official * 0.9))
        rows.append(_record("atari100k_pong", seed, "worldflux", worldflux, worldflux * 0.9))

    runs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = _run_stats(runs_path, tmp_path / "report.json")

    assert report["schema_version"] == "parity.v1"
    assert report["global"]["parity_pass_primary"] is True
    assert report["tasks"][0]["task_pass_primary"] is True


def test_stats_equivalence_fails_when_large_underperformance(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.jsonl"
    rows: list[dict] = []
    for seed in range(20):
        official = 100.0 + seed * 0.1
        worldflux = official * 0.7
        rows.append(_record("dog-run", seed, "official", official, official * 0.9))
        rows.append(_record("dog-run", seed, "worldflux", worldflux, worldflux * 0.9))

    runs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = _run_stats(runs_path, tmp_path / "report.json")

    assert report["global"]["parity_pass_primary"] is False
    assert report["tasks"][0]["task_pass_primary"] is False
    metric = report["tasks"][0]["metrics"]["final_return_mean"]
    assert metric["tost"]["pass_raw"] is False


def test_stats_equivalence_applies_holm_correction_across_primary_hypotheses(
    tmp_path: Path,
) -> None:
    runs_path = tmp_path / "runs.jsonl"
    rows: list[dict] = []

    for seed in range(20):
        off_a = 100.0 + seed * 0.1
        wf_a = off_a * 1.01
        off_b = 100.0 + seed * 0.1
        wf_b = off_b * 1.06

        rows.append(_record("task_a", seed, "official", off_a, off_a))
        rows.append(_record("task_a", seed, "worldflux", wf_a, wf_a))
        rows.append(_record("task_b", seed, "official", off_b, off_b))
        rows.append(_record("task_b", seed, "worldflux", wf_b, wf_b))

    runs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = _run_stats(runs_path, tmp_path / "report.json")

    task_map = {task["task_id"]: task for task in report["tasks"]}

    assert task_map["task_a"]["metrics"]["final_return_mean"]["pass_with_holm_primary"] is True
    assert task_map["task_b"]["metrics"]["final_return_mean"]["pass_with_holm_primary"] is False
    assert report["global"]["parity_pass_primary"] is False
