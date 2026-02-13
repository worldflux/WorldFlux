"""End-to-end parity smoke test with mock adapters."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_parity_pipeline_smoke_generates_all_artifacts(tmp_path: Path) -> None:
    root = _repo_root()
    manifest = {
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
                    "cwd": str(root),
                    "env": {},
                    "command": [
                        "python3",
                        "scripts/parity/wrappers/official_dreamerv3.py",
                        "--task-id",
                        "{task_id}",
                        "--seed",
                        "{seed}",
                        "--steps",
                        "1000",
                        "--device",
                        "cpu",
                        "--run-dir",
                        "{run_root}/executions/{task_id}/seed_{seed}/official",
                        "--metrics-out",
                        "{metrics_out}",
                        "--mock",
                    ],
                },
                "worldflux": {
                    "adapter": "worldflux_dreamerv3_native",
                    "cwd": str(root),
                    "env": {},
                    "command": [
                        "python3",
                        "scripts/parity/wrappers/worldflux_dreamerv3_native.py",
                        "--task-id",
                        "{task_id}",
                        "--seed",
                        "{seed}",
                        "--steps",
                        "1000",
                        "--device",
                        "cpu",
                        "--run-dir",
                        "{run_root}/executions/{task_id}/seed_{seed}/worldflux",
                        "--metrics-out",
                        "{metrics_out}",
                        "--mock",
                    ],
                },
            },
            {
                "task_id": "dog-run",
                "family": "tdmpc2",
                "required_metrics": ["final_return_mean", "auc_return"],
                "official": {
                    "adapter": "official_tdmpc2",
                    "cwd": str(root),
                    "env": {},
                    "command": [
                        "python3",
                        "scripts/parity/wrappers/official_tdmpc2.py",
                        "--task-id",
                        "{task_id}",
                        "--seed",
                        "{seed}",
                        "--steps",
                        "1000",
                        "--device",
                        "cpu",
                        "--run-dir",
                        "{run_root}/executions/{task_id}/seed_{seed}/official",
                        "--metrics-out",
                        "{metrics_out}",
                        "--mock",
                    ],
                },
                "worldflux": {
                    "adapter": "worldflux_tdmpc2_native",
                    "cwd": str(root),
                    "env": {},
                    "command": [
                        "python3",
                        "scripts/parity/wrappers/worldflux_tdmpc2_native.py",
                        "--task-id",
                        "{task_id}",
                        "--seed",
                        "{seed}",
                        "--steps",
                        "1000",
                        "--device",
                        "cpu",
                        "--run-dir",
                        "{run_root}/executions/{task_id}/seed_{seed}/worldflux",
                        "--metrics-out",
                        "{metrics_out}",
                        "--mock",
                    ],
                },
            },
        ],
    }

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    run_cmd = [
        "python3",
        "scripts/parity/run_parity_matrix.py",
        "--manifest",
        str(manifest_path),
        "--run-id",
        "smoke",
        "--output-dir",
        str(tmp_path / "reports"),
    ]
    run = subprocess.run(run_cmd, cwd=root, check=False, text=True, capture_output=True)
    assert run.returncode == 0, run.stderr

    run_root = tmp_path / "reports" / "smoke"
    runs_jsonl = run_root / "parity_runs.jsonl"
    assert runs_jsonl.exists()

    lines = [
        json.loads(line) for line in runs_jsonl.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(lines) == 4
    assert all(line["status"] == "success" for line in lines)

    stats_cmd = [
        "python3",
        "scripts/parity/stats_equivalence.py",
        "--input",
        str(runs_jsonl),
        "--output",
        str(run_root / "equivalence_report.json"),
        "--min-pairs",
        "1",
    ]
    stats = subprocess.run(stats_cmd, cwd=root, check=False, text=True, capture_output=True)
    assert stats.returncode == 0, stats.stderr

    md_cmd = [
        "python3",
        "scripts/parity/report_markdown.py",
        "--input",
        str(run_root / "equivalence_report.json"),
        "--output",
        str(run_root / "equivalence_report.md"),
    ]
    md = subprocess.run(md_cmd, cwd=root, check=False, text=True, capture_output=True)
    assert md.returncode == 0, md.stderr

    assert (run_root / "equivalence_report.json").exists()
    assert (run_root / "equivalence_report.md").exists()
