"""Tests for parity equivalence report schemas and markdown rendering."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_equivalence_report_json_has_stable_top_level_keys(tmp_path: Path) -> None:
    root = _repo_root()
    runs = []
    for seed in range(4):
        runs.append(
            {
                "schema_version": "parity.v1",
                "run_id": "schema",
                "task_id": "task",
                "family": "dreamerv3",
                "seed": seed,
                "system": "official",
                "adapter": "x",
                "status": "success",
                "metrics": {"final_return_mean": 100.0, "auc_return": 90.0},
            }
        )
        runs.append(
            {
                "schema_version": "parity.v1",
                "run_id": "schema",
                "task_id": "task",
                "family": "dreamerv3",
                "seed": seed,
                "system": "worldflux",
                "adapter": "x",
                "status": "success",
                "metrics": {"final_return_mean": 101.0, "auc_return": 91.0},
            }
        )

    runs_path = tmp_path / "runs.jsonl"
    runs_path.write_text("\n".join(json.dumps(r) for r in runs) + "\n", encoding="utf-8")

    report_path = tmp_path / "equivalence_report.json"
    cmd = [
        "python3",
        "scripts/parity/stats_equivalence.py",
        "--input",
        str(runs_path),
        "--output",
        str(report_path),
        "--min-pairs",
        "2",
    ]
    completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(report.keys()) == {
        "schema_version",
        "generated_at",
        "input",
        "config",
        "completeness",
        "validity",
        "bayesian",
        "holm",
        "tasks",
        "global",
    }
    assert report["schema_version"] == "parity.v1"


def test_report_markdown_renders_expected_sections(tmp_path: Path) -> None:
    root = _repo_root()
    report = {
        "schema_version": "parity.v1",
        "generated_at": "2026-02-13T00:00:00+00:00",
        "input": "runs.jsonl",
        "config": {"primary_metric": "final_return_mean", "metrics": ["final_return_mean"]},
        "holm": {"primary": {}, "all_metrics": {}},
        "tasks": [
            {
                "task_id": "atari100k_pong",
                "task_pass_primary": True,
                "task_pass_all_metrics": True,
                "metrics": {
                    "final_return_mean": {
                        "status": "ok",
                        "n_pairs": 20,
                        "official_mean": 100.0,
                        "worldflux_mean": 101.0,
                        "ratio_mean": 1.01,
                        "ci90_ratio": [0.99, 1.03],
                        "tost": {"p_value": 0.01},
                        "holm_primary": {"adjusted_p": 0.02},
                        "holm_all_metrics": {"adjusted_p": 0.02},
                        "noninferiority": {"pass_raw": True},
                    }
                },
            }
        ],
        "global": {
            "tasks_total": 1,
            "tasks_pass_primary": 1,
            "tasks_pass_all_metrics": 1,
            "parity_pass_primary": True,
            "parity_pass_all_metrics": True,
        },
    }

    report_path = tmp_path / "equivalence_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md_path = tmp_path / "equivalence_report.md"
    cmd = [
        "python3",
        "scripts/parity/report_markdown.py",
        "--input",
        str(report_path),
        "--output",
        str(md_path),
    ]
    completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr

    rendered = md_path.read_text(encoding="utf-8")
    assert "# WorldFlux Parity Equivalence Report" in rendered
    assert "## Global Verdict" in rendered
    assert "## Per-Metric Results" in rendered
    assert "atari100k_pong" in rendered


def test_report_markdown_renders_bayesian_sections_when_enabled(tmp_path: Path) -> None:
    root = _repo_root()
    report = {
        "schema_version": "parity.v1",
        "generated_at": "2026-02-13T00:00:00+00:00",
        "input": "runs.jsonl",
        "config": {
            "primary_metric": "final_return_mean",
            "metrics": ["final_return_mean"],
            "bayesian": {
                "enabled": True,
                "draws": 20000,
                "seed": 20260220,
                "probability_threshold_equivalence": 0.95,
                "probability_threshold_noninferiority": 0.975,
                "dual_pass_required": True,
            },
        },
        "completeness": {"expected_pairs": 2, "missing_pairs": 0},
        "validity": {"pass": True, "proof_mode": True, "required_policy_mode": "parity_candidate"},
        "bayesian": {
            "enabled": True,
            "tasks_total": 1,
            "tasks_pass_all_metrics": 1,
            "parity_pass_all_metrics": True,
            "posterior_summary": {"mean_p_equivalence": 0.99, "mean_p_noninferior": 1.0},
        },
        "holm": {"primary": {}, "all_metrics": {}},
        "tasks": [
            {
                "task_id": "atari100k_pong",
                "task_pass_primary": True,
                "task_pass_all_metrics": True,
                "task_pass_bayesian": True,
                "metrics": {
                    "final_return_mean": {
                        "status": "ok",
                        "n_pairs": 20,
                        "official_mean": 100.0,
                        "worldflux_mean": 101.0,
                        "ratio_mean": 1.01,
                        "ci90_ratio": [0.99, 1.03],
                        "tost": {"p_value": 0.01},
                        "holm_primary": {"adjusted_p": 0.02},
                        "holm_all_metrics": {"adjusted_p": 0.02},
                        "noninferiority": {"pass_raw": True},
                        "bayesian": {
                            "status": "ok",
                            "p_equivalence": 0.99,
                            "p_noninferior": 1.0,
                            "posterior_ci90": [-0.01, 0.02],
                            "pass_all": True,
                        },
                    }
                },
            }
        ],
        "global": {
            "tasks_total": 1,
            "tasks_pass_primary": 1,
            "tasks_pass_all_metrics": 1,
            "parity_pass_primary": True,
            "parity_pass_all_metrics": True,
            "parity_pass_frequentist": True,
            "parity_pass_bayesian": True,
            "parity_pass_final": True,
            "validity_pass": True,
            "missing_pairs": 0,
            "strict_mode_failed": False,
        },
    }

    report_path = tmp_path / "equivalence_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md_path = tmp_path / "equivalence_report.md"
    cmd = [
        "python3",
        "scripts/parity/report_markdown.py",
        "--input",
        str(report_path),
        "--output",
        str(md_path),
    ]
    completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr

    rendered = md_path.read_text(encoding="utf-8")
    assert "## Bayesian Summary" in rendered
    assert "## Bayesian Per-Metric Results" in rendered
