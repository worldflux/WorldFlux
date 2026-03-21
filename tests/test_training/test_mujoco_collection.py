# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for MuJoCo collection helpers."""

from __future__ import annotations

import numpy as np
import pytest

from worldflux.training import ReplayBuffer
from worldflux.training.dataset_manifest import build_dataset_manifest, write_dataset_manifest
from worldflux.training.mujoco_collection import _build_action_fn, load_buffer_from_dataset_manifest


def test_load_buffer_from_dataset_manifest_restores_replay_buffer(tmp_path) -> None:
    buffer = ReplayBuffer(capacity=16, obs_shape=(4,), action_dim=2)
    buffer.add_episode(
        obs=np.random.randn(4, 4).astype(np.float32),
        actions=np.random.randn(4, 2).astype(np.float32),
        rewards=np.random.randn(4).astype(np.float32),
        dones=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    replay_path = tmp_path / "replay_buffer.npz"
    buffer.save(replay_path)

    manifest = build_dataset_manifest(
        env_id="mujoco/HalfCheetah-v5",
        collector_kind="mujoco_collector",
        collector_policy="policy_checkpoint",
        seed=7,
        episodes=1,
        transitions=4,
        reward_stats={"mean": 0.0, "std": 1.0},
        source_commit="abc123",
        created_at="2026-03-20T00:00:00Z",
        preprocessing={"obs_norm": False},
        artifact_paths={"replay_buffer": replay_path.name},
    )
    manifest_path = write_dataset_manifest(tmp_path / "dataset.manifest.json", manifest)

    restored, loaded_manifest = load_buffer_from_dataset_manifest(manifest_path)

    assert len(restored) == 4
    assert restored.data_provenance["kind"] == "dataset_manifest"
    assert loaded_manifest["env_id"] == "mujoco/HalfCheetah-v5"


def test_build_action_fn_random_returns_vector() -> None:
    fn = _build_action_fn(
        collector_policy="random",
        policy_checkpoint=None,
        action_dim=3,
        device="cpu",
    )
    action = fn(np.zeros(4, dtype=np.float32), rng=np.random.default_rng(0))
    assert action.shape == (3,)


def test_build_action_fn_policy_checkpoint_requires_checkpoint() -> None:
    with pytest.raises(ValueError, match="policy-checkpoint"):
        _build_action_fn(
            collector_policy="policy_checkpoint",
            policy_checkpoint=None,
            action_dim=3,
            device="cpu",
        )
