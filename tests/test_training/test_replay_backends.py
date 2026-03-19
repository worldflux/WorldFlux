"""Tests for replay backend abstractions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from worldflux.training.data import ReplayBuffer
from worldflux.training.replay_backends import MemoryReplayBackend, ParquetReplayBackend


def _buffer() -> ReplayBuffer:
    buffer = ReplayBuffer(capacity=16, obs_shape=(4,), action_dim=2)
    obs = np.arange(24, dtype=np.float32).reshape(6, 4)
    actions = np.ones((6, 2), dtype=np.float32)
    rewards = np.linspace(0.0, 1.0, 6, dtype=np.float32)
    dones = np.zeros(6, dtype=np.float32)
    dones[-1] = 1.0
    buffer.add_episode(obs=obs, actions=actions, rewards=rewards, dones=dones)
    return buffer


def test_memory_backend_matches_existing_replay_buffer_sampling() -> None:
    buffer = _buffer()
    backend = MemoryReplayBackend(buffer)

    batch = backend.sample(batch_size=2, seq_len=3, device="cpu")
    assert batch.obs.shape == (2, 3, 4)
    assert batch.actions.shape == (2, 3, 2)
    assert backend.capacity == buffer.capacity
    assert backend.obs_shape == buffer.obs_shape
    assert backend.action_dim == buffer.action_dim


def test_parquet_backend_roundtrip_preserves_schema(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    buffer = _buffer()
    backend = MemoryReplayBackend(buffer)
    parquet_path = tmp_path / "buffer.parquet"

    backend.to_parquet(parquet_path)
    restored = ParquetReplayBackend.from_parquet(parquet_path)

    assert restored.capacity == backend.capacity
    assert restored.obs_shape == backend.obs_shape
    assert restored.action_dim == backend.action_dim

    batch = restored.sample(batch_size=1, seq_len=3, device="cpu")
    assert batch.obs.shape == (1, 3, 4)
