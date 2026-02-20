"""Tests for run_parity_matrix --systems filtering."""

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


def _write_manifest(path: Path, emit_script: Path) -> None:
    root = _repo_root()
    payload = {
        "schema_version": "parity.manifest.v1",
        "defaults": {"alpha": 0.05, "equivalence_margin": 0.05},
        "seed_policy": {
            "mode": "fixed",
            "values": [0, 1],
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
                    "command": ["python3", str(emit_script), "--metrics-out", "{metrics_out}"],
                },
                "worldflux": {
                    "adapter": "worldflux_dreamerv3_native",
                    "cwd": str(root),
                    "env": {},
                    "command": ["python3", str(emit_script), "--metrics-out", "{metrics_out}"],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run(
    *,
    manifest_path: Path,
    reports_root: Path,
    run_id: str,
    systems: str | None,
) -> list[dict]:
    root = _repo_root()
    cmd = [
        "python3",
        "scripts/parity/run_parity_matrix.py",
        "--manifest",
        str(manifest_path),
        "--run-id",
        run_id,
        "--output-dir",
        str(reports_root),
    ]
    if systems is not None:
        cmd.extend(["--systems", systems])

    completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr

    run_jsonl = reports_root / run_id / "parity_runs.jsonl"
    return [json.loads(line) for line in run_jsonl.read_text(encoding="utf-8").splitlines() if line]


def test_run_parity_matrix_system_filter(tmp_path: Path) -> None:
    emit_script = tmp_path / "emit_metrics.py"
    manifest_path = tmp_path / "manifest.json"
    reports_root = tmp_path / "reports"

    _write_emit_script(emit_script)
    _write_manifest(manifest_path, emit_script)

    official_rows = _run(
        manifest_path=manifest_path,
        reports_root=reports_root,
        run_id="official_only",
        systems="official",
    )
    assert {row["system"] for row in official_rows if row["status"] == "success"} == {"official"}

    worldflux_rows = _run(
        manifest_path=manifest_path,
        reports_root=reports_root,
        run_id="worldflux_only",
        systems="worldflux",
    )
    assert {row["system"] for row in worldflux_rows if row["status"] == "success"} == {"worldflux"}

    default_rows = _run(
        manifest_path=manifest_path,
        reports_root=reports_root,
        run_id="default_systems",
        systems=None,
    )
    assert {row["system"] for row in default_rows if row["status"] == "success"} == {
        "official",
        "worldflux",
    }


def test_run_parity_matrix_pair_plan_executes_only_requested_pairs(tmp_path: Path) -> None:
    emit_script = tmp_path / "emit_metrics.py"
    manifest_path = tmp_path / "manifest.json"
    reports_root = tmp_path / "reports"
    pair_plan_path = tmp_path / "pair_plan.json"

    _write_emit_script(emit_script)
    _write_manifest(manifest_path, emit_script)

    pair_plan = {
        "pairs": [
            {"task_id": "atari100k_pong", "seed": 1, "system": "worldflux"},
            {"task_id": "atari100k_pong", "seed": 3, "system": "official"},
        ]
    }
    pair_plan_path.write_text(json.dumps(pair_plan, indent=2) + "\n", encoding="utf-8")

    root = _repo_root()
    cmd = [
        "python3",
        "scripts/parity/run_parity_matrix.py",
        "--manifest",
        str(manifest_path),
        "--run-id",
        "pair_plan_only",
        "--output-dir",
        str(reports_root),
        "--pair-plan",
        str(pair_plan_path),
    ]
    completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr

    rows = [
        json.loads(line)
        for line in (reports_root / "pair_plan_only" / "parity_runs.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert {(row["seed"], row["system"]) for row in rows if row["status"] == "success"} == {
        (1, "worldflux"),
        (3, "official"),
    }
