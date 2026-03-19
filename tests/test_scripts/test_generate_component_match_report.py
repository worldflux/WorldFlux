# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for component match report generator script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from worldflux.models.tdmpc2 import TDMPC2WorldModel
from worldflux.models.tdmpc2.world_model import TDMPC2Config


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_generate_tdmpc2_component_match_report_from_torch_checkpoint(tmp_path: Path) -> None:
    root = _repo_root()
    config = TDMPC2Config.from_size("5m", obs_shape=(39,), action_dim=6)
    model = TDMPC2WorldModel(config)
    checkpoint = tmp_path / "official_tdmpc2.pt"
    torch.save(model.state_dict(), checkpoint)

    output = tmp_path / "component_match_report.json"
    cmd = [
        sys.executable,
        "scripts/parity/generate_component_match_report.py",
        "--family",
        "tdmpc2",
        "--official-checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--obs-shape",
        "39",
        "--action-dim",
        "6",
    ]
    completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "parity.component_match.v1"
    assert payload["family"] == "tdmpc2"
    assert payload["all_pass"] is True
    assert len(payload["results"]) >= 1


def test_generate_dreamerv3_component_match_report_from_npz_checkpoint(tmp_path: Path) -> None:
    root = _repo_root()
    checkpoint = tmp_path / "official_dreamerv3.npz"
    np.savez(
        checkpoint,
        **{
            "encoder/mlp/linear_0/w": np.ones((10, 16), dtype=np.float32),
            "encoder/mlp/linear_0/b": np.zeros((16,), dtype=np.float32),
            "encoder/mlp/layer_norm_0/scale": np.ones((16,), dtype=np.float32),
            "encoder/mlp/layer_norm_0/offset": np.zeros((16,), dtype=np.float32),
            "encoder/mlp/linear_1/w": np.ones((16, 16), dtype=np.float32),
            "encoder/mlp/linear_1/b": np.zeros((16,), dtype=np.float32),
        },
    )
    config_json = tmp_path / "dreamer_config.json"
    config_json.write_text(
        json.dumps(
            {
                "encoder_type": "mlp",
                "decoder_type": "mlp",
                "hidden_dim": 16,
                "deter_dim": 16,
                "stoch_discrete": 2,
                "stoch_classes": 2,
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "component_match_report.json"
    cmd = [
        sys.executable,
        "scripts/parity/generate_component_match_report.py",
        "--family",
        "dreamerv3",
        "--official-checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--obs-shape",
        "10",
        "--action-dim",
        "6",
        "--config-json",
        str(config_json),
    ]
    completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "parity.component_match.v1"
    assert payload["family"] == "dreamerv3"
    assert "results" in payload
