"""Tests for run_parity_matrix sharding and task filtering."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _write_emit_script(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import argparse",
                "import json",
                "from pathlib import Path",
                "",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--metrics-out', required=True)",
                "args = parser.parse_args()",
                "metrics = {'final_return_mean': 1.0, 'auc_return': 1.0, 'success': True}",
                "out = Path(args.metrics_out)",
                "out.parent.mkdir(parents=True, exist_ok=True)",
                "out.write_text(json.dumps(metrics), encoding='utf-8')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_parity_matrix_shards_without_overlap(tmp_path: Path) -> None:
    root = _repo_root()
    emit_script = tmp_path / "emit_metrics.py"
    _write_emit_script(emit_script)

    tasks = ["atari100k_pong", "atari100k_breakout", "atari100k_qbert"]
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
        "tasks": [],
    }

    for task_id in tasks:
        manifest["tasks"].append(
            {
                "task_id": task_id,
                "family": "dreamerv3",
                "required_metrics": ["final_return_mean", "auc_return"],
                "official": {
                    "adapter": "official_dreamerv3",
                    "cwd": str(root),
                    "env": {},
                    "command": [
                        "python3",
                        str(emit_script),
                        "--metrics-out",
                        "{metrics_out}",
                    ],
                },
                "worldflux": {
                    "adapter": "worldflux_dreamerv3_native",
                    "cwd": str(root),
                    "env": {},
                    "command": [
                        "python3",
                        str(emit_script),
                        "--metrics-out",
                        "{metrics_out}",
                    ],
                },
            }
        )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    shard_records: list[list[dict]] = []
    for shard_index in (0, 1):
        run_id = f"shard_{shard_index}"
        cmd = [
            "python3",
            "scripts/parity/run_parity_matrix.py",
            "--manifest",
            str(manifest_path),
            "--run-id",
            run_id,
            "--output-dir",
            str(tmp_path / "reports"),
            "--num-shards",
            "2",
            "--shard-index",
            str(shard_index),
        ]
        completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
        assert completed.returncode == 0, completed.stderr

        jsonl = tmp_path / "reports" / run_id / "parity_runs.jsonl"
        rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line]
        assert all(row["status"] == "success" for row in rows)
        shard_records.append(rows)

    shard_task_sets = [
        {str(row["task_id"]) for row in rows if row["status"] == "success"}
        for rows in shard_records
    ]

    assert shard_task_sets[0].isdisjoint(shard_task_sets[1])
    assert shard_task_sets[0] | shard_task_sets[1] == set(tasks)

    total_success = sum(len(rows) for rows in shard_records)
    assert total_success == len(tasks) * 2  # official + worldflux
