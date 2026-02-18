"""Tests for real-environment path in worldflux native online runner."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_worldflux_native_online_runner_uses_real_env_backend_when_not_mock(tmp_path: Path) -> None:
    root = _repo_root()
    metrics_out = tmp_path / "metrics.json"

    cmd = [
        "python3",
        "scripts/parity/wrappers/worldflux_native_online_runner.py",
        "--family",
        "dreamerv3",
        "--task-id",
        "atari100k_pong",
        "--seed",
        "0",
        "--steps",
        "12",
        "--eval-interval",
        "6",
        "--eval-episodes",
        "1",
        "--eval-window",
        "2",
        "--env-backend",
        "stub",
        "--device",
        "cpu",
        "--buffer-capacity",
        "64",
        "--warmup-steps",
        "1",
        "--train-steps-per-eval",
        "1",
        "--sequence-length",
        "2",
        "--batch-size",
        "2",
        "--max-episode-steps",
        "4",
        "--run-dir",
        str(tmp_path / "run"),
        "--metrics-out",
        str(metrics_out),
    ]

    completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr

    payload = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "parity.v1"
    assert payload["metadata"]["mode"] == "native_real_env"
    assert payload["metadata"]["env_backend"] == "stub"
    assert payload["metadata"]["policy_mode"] == "diagnostic_random"
    assert isinstance(payload["metadata"]["eval_protocol_hash"], str)
    assert payload["metadata"]["eval_protocol_hash"]
    assert payload["num_curve_points"] >= 1
