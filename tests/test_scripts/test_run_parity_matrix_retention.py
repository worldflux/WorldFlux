"""Tests for run_parity_matrix artifact retention behavior."""

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
                "metrics_path = Path(args.metrics_out)",
                "metrics_path.parent.mkdir(parents=True, exist_ok=True)",
                "metrics = {'final_return_mean': 1.0, 'auc_return': 1.0, 'success': True}",
                "metrics_path.write_text(json.dumps(metrics), encoding='utf-8')",
                "checkpoint_dir = metrics_path.parent / 'checkpoints'",
                "checkpoint_dir.mkdir(parents=True, exist_ok=True)",
                "(checkpoint_dir / 'model.ckpt').write_text('x', encoding='utf-8')",
                "replay_dir = metrics_path.parent / 'replay_buffer'",
                "replay_dir.mkdir(parents=True, exist_ok=True)",
                "(replay_dir / 'buffer.npy').write_text('x', encoding='utf-8')",
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


def _run_matrix(*, manifest_path: Path, reports_root: Path, run_id: str, retention: str) -> None:
    cmd = [
        "python3",
        "scripts/parity/run_parity_matrix.py",
        "--manifest",
        str(manifest_path),
        "--run-id",
        run_id,
        "--output-dir",
        str(reports_root),
        "--systems",
        "official",
        "--artifact-retention",
        retention,
    ]
    completed = subprocess.run(
        cmd,
        cwd=_repo_root(),
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_run_parity_matrix_minimal_retention_prunes_heavy_artifacts(tmp_path: Path) -> None:
    emit_script = tmp_path / "emit_metrics.py"
    manifest_path = tmp_path / "manifest.json"
    reports_root = tmp_path / "reports"
    _write_emit_script(emit_script)
    _write_manifest(manifest_path, emit_script)

    _run_matrix(
        manifest_path=manifest_path,
        reports_root=reports_root,
        run_id="retention_minimal",
        retention="minimal",
    )

    run_root = (
        reports_root / "retention_minimal" / "executions" / "atari100k_pong" / "seed_0" / "official"
    )
    assert (run_root / "metrics.json").exists()
    assert (run_root / "stdout.log").exists()
    assert (run_root / "stderr.log").exists()
    assert not (run_root / "checkpoints").exists()
    assert not (run_root / "replay_buffer").exists()
    assert (reports_root / "retention_minimal" / "run_context.json").exists()
    assert (reports_root / "retention_minimal" / "seed_plan.json").exists()
    assert (reports_root / "retention_minimal" / "run_summary.json").exists()
    assert not list((reports_root / "retention_minimal").glob("*.tmp"))


def test_run_parity_matrix_full_retention_keeps_heavy_artifacts(tmp_path: Path) -> None:
    emit_script = tmp_path / "emit_metrics.py"
    manifest_path = tmp_path / "manifest.json"
    reports_root = tmp_path / "reports"
    _write_emit_script(emit_script)
    _write_manifest(manifest_path, emit_script)

    _run_matrix(
        manifest_path=manifest_path,
        reports_root=reports_root,
        run_id="retention_full",
        retention="full",
    )

    system_dir = (
        reports_root / "retention_full" / "executions" / "atari100k_pong" / "seed_0" / "official"
    )
    assert (system_dir / "checkpoints" / "model.ckpt").exists()
    assert (system_dir / "replay_buffer" / "buffer.npy").exists()
