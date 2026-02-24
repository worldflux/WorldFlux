"""Tests for parity adapter wrappers and retry behavior."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _wrapper_common_module():
    root = _repo_root()
    wrappers_dir = root / "scripts" / "parity" / "wrappers"
    path = str(wrappers_dir)
    if path not in sys.path:
        sys.path.insert(0, path)
    from common import load_csv_curve  # type: ignore

    return load_csv_curve


def test_adapter_wrappers_emit_normalized_metrics(tmp_path: Path) -> None:
    root = _repo_root()
    wrappers = [
        (
            "official_dreamerv3.py",
            ["--task-id", "atari100k_pong", "--seed", "0", "--steps", "1000", "--mock"],
        ),
        (
            "official_tdmpc2.py",
            ["--task-id", "dog-run", "--seed", "0", "--steps", "1000", "--mock"],
        ),
        (
            "worldflux_dreamerv3_native.py",
            ["--task-id", "atari100k_pong", "--seed", "0", "--steps", "1000", "--mock"],
        ),
        (
            "worldflux_tdmpc2_native.py",
            ["--task-id", "dog-run", "--seed", "0", "--steps", "1000", "--mock"],
        ),
    ]

    for wrapper_name, extra_args in wrappers:
        run_dir = tmp_path / wrapper_name.replace(".py", "")
        metrics_out = run_dir / "metrics.json"
        wrapper = root / "scripts" / "parity" / "wrappers" / wrapper_name

        cmd = [
            "python3",
            str(wrapper),
            "--run-dir",
            str(run_dir),
            "--metrics-out",
            str(metrics_out),
            "--device",
            "cpu",
            *extra_args,
        ]
        completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
        assert completed.returncode == 0, completed.stderr
        assert metrics_out.exists()

        payload = json.loads(metrics_out.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "parity.v1"
        assert isinstance(payload["final_return_mean"], float)
        assert isinstance(payload["auc_return"], float)
        metadata = payload.get("metadata", {})
        if wrapper_name == "official_dreamerv3.py":
            assert metadata.get("env_backend") == "gymnasium"
        if wrapper_name == "official_tdmpc2.py":
            assert metadata.get("env_backend") == "dmcontrol"


def test_run_parity_matrix_retries_failed_attempt_once(tmp_path: Path) -> None:
    root = _repo_root()
    fail_once = tmp_path / "fail_once.py"
    fail_once.write_text(
        """
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--marker", required=True)
parser.add_argument("--metrics-out", required=True)
args = parser.parse_args()

marker = Path(args.marker)
if not marker.exists():
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("first_fail", encoding="utf-8")
    raise SystemExit(7)

metrics_out = Path(args.metrics_out)
metrics_out.parent.mkdir(parents=True, exist_ok=True)
metrics_out.write_text(json.dumps({"final_return_mean": 1.0, "auc_return": 1.0}), encoding="utf-8")
""".strip()
        + "\n",
        encoding="utf-8",
    )

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
                "task_id": "retry-task",
                "family": "dreamerv3",
                "required_metrics": ["final_return_mean", "auc_return"],
                "official": {
                    "adapter": "official_dreamerv3",
                    "cwd": ".",
                    "env": {},
                    "command": [
                        "python3",
                        str(fail_once),
                        "--marker",
                        str(tmp_path / "official.marker"),
                        "--metrics-out",
                        "{metrics_out}",
                    ],
                },
                "worldflux": {
                    "adapter": "worldflux_dreamerv3_native",
                    "cwd": ".",
                    "env": {},
                    "command": [
                        "python3",
                        str(fail_once),
                        "--marker",
                        str(tmp_path / "worldflux.marker"),
                        "--metrics-out",
                        "{metrics_out}",
                    ],
                },
            }
        ],
    }

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    cmd = [
        "python3",
        "scripts/parity/run_parity_matrix.py",
        "--manifest",
        str(manifest_path),
        "--run-id",
        "retry_test",
        "--output-dir",
        str(tmp_path / "reports"),
        "--max-retries",
        "1",
    ]
    completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr

    jsonl_path = tmp_path / "reports" / "retry_test" / "parity_runs.jsonl"
    records = [
        json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line
    ]

    assert len(records) == 2
    assert all(record["status"] == "success" for record in records)
    assert all(record["attempt"] == 1 for record in records)


def test_load_csv_curve_parses_numeric_strings(tmp_path: Path) -> None:
    load_csv_curve = _wrapper_common_module()
    csv_path = tmp_path / "eval.csv"
    csv_path.write_text(
        "step,episode_reward\n0.0,1.5\n5000,2.75\n",
        encoding="utf-8",
    )

    points = load_csv_curve(csv_path, value_keys=["episode_reward"])
    assert len(points) == 2
    assert points[0].step == 0.0
    assert points[0].value == 1.5
    assert points[1].step == 5000.0
    assert points[1].value == 2.75
