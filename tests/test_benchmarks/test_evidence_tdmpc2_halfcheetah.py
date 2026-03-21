# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for the TD-MPC2 HalfCheetah evidence benchmark script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from worldflux import create_world_model
from worldflux.training.data import create_random_buffer

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "benchmarks" / "evidence_tdmpc2_halfcheetah.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("evidence_tdmpc2_halfcheetah", _SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["evidence_tdmpc2_halfcheetah"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_quick_mode_writes_evidence_artifacts(tmp_path, monkeypatch) -> None:
    mod = _load_module()
    buffer = create_random_buffer(
        capacity=256,
        obs_shape=(39,),
        action_dim=6,
        num_episodes=8,
        episode_length=16,
        seed=7,
    )
    manifest_path = tmp_path / "dataset.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "worldflux.dataset.manifest.v1",
                "env_id": "mujoco/HalfCheetah-v5",
                "collector_kind": "mujoco_collector",
                "collector_policy": "policy_checkpoint",
                "artifact_paths": {"replay_buffer": "replay_buffer.npz"},
            }
        ),
        encoding="utf-8",
    )
    buffer.data_provenance = {
        "kind": "dataset_manifest",
        "env_id": "mujoco/HalfCheetah-v5",
        "dataset_manifest": str(manifest_path),
    }

    monkeypatch.setattr(
        mod,
        "_prepare_buffer_and_manifest",
        lambda *args, **kwargs: (buffer, manifest_path, json.loads(manifest_path.read_text())),
    )
    monkeypatch.setattr(
        mod,
        "create_world_model",
        lambda *args, **kwargs: create_world_model("tdmpc2:ci", obs_shape=(39,), action_dim=6),
    )
    monkeypatch.setattr(mod, "_collect_policy_returns", lambda *args, **kwargs: [1.0, 2.0, 3.0])

    exit_code = mod.main(["--quick", "--output-dir", str(tmp_path)])

    assert exit_code == 0
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "returns.jsonl").exists()
    assert (tmp_path / "learning_curve.csv").exists()
    assert (tmp_path / "checkpoint_index.json").exists()
    assert (tmp_path / "report.md").exists()

    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["benchmark"] == "tdmpc2-halfcheetah-evidence"
    assert summary["artifacts"]["dataset_manifest"] == str(manifest_path.resolve())
