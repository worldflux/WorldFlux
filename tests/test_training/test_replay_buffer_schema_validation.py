"""Schema validation tests for ReplayBuffer.load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from worldflux.core.exceptions import BufferError
from worldflux.training import ReplayBuffer


def _valid_payload() -> dict[str, np.ndarray]:
    obs = np.zeros((5, 4), dtype=np.float32)
    actions = np.zeros((5, 2), dtype=np.float32)
    rewards = np.zeros(5, dtype=np.float32)
    dones = np.zeros(5, dtype=np.float32)
    return {
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "obs_shape": np.array([4], dtype=np.int64),
        "action_dim": np.array(2, dtype=np.int64),
        "capacity": np.array(8, dtype=np.int64),
    }


def _write_npz(path: Path, payload: dict[str, np.ndarray]) -> None:
    np.savez(path, **payload)


def test_replay_buffer_load_rejects_missing_required_keys(tmp_path: Path) -> None:
    path = tmp_path / "missing_keys.npz"
    payload = _valid_payload()
    payload.pop("capacity")
    _write_npz(path, payload)

    with pytest.raises(BufferError, match="missing keys"):
        ReplayBuffer.load(path)


def test_replay_buffer_load_rejects_obs_shape_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "obs_shape_mismatch.npz"
    payload = _valid_payload()
    payload["obs_shape"] = np.array([5], dtype=np.int64)
    _write_npz(path, payload)

    with pytest.raises(BufferError, match="obs shape does not match obs_shape"):
        ReplayBuffer.load(path)


def test_replay_buffer_load_rejects_action_dim_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "action_dim_mismatch.npz"
    payload = _valid_payload()
    payload["actions"] = np.zeros((5, 3), dtype=np.float32)
    _write_npz(path, payload)

    with pytest.raises(BufferError, match="actions shape does not match action_dim"):
        ReplayBuffer.load(path)


def test_replay_buffer_load_rejects_non_positive_capacity(tmp_path: Path) -> None:
    path = tmp_path / "bad_capacity.npz"
    payload = _valid_payload()
    payload["capacity"] = np.array(0, dtype=np.int64)
    _write_npz(path, payload)

    with pytest.raises(BufferError, match="Invalid capacity metadata"):
        ReplayBuffer.load(path)


def test_replay_buffer_load_rejects_inconsistent_row_lengths(tmp_path: Path) -> None:
    path = tmp_path / "inconsistent_lengths.npz"
    payload = _valid_payload()
    payload["rewards"] = np.zeros(4, dtype=np.float32)
    _write_npz(path, payload)

    with pytest.raises(BufferError, match="Inconsistent trajectory lengths"):
        ReplayBuffer.load(path)


def test_replay_buffer_load_rejects_when_rows_exceed_capacity(tmp_path: Path) -> None:
    path = tmp_path / "rows_gt_capacity.npz"
    payload = _valid_payload()
    payload["capacity"] = np.array(3, dtype=np.int64)
    _write_npz(path, payload)

    with pytest.raises(BufferError, match="exceeds capacity"):
        ReplayBuffer.load(path)


def test_replay_buffer_load_accepts_valid_schema(tmp_path: Path) -> None:
    path = tmp_path / "valid.npz"
    payload = _valid_payload()
    _write_npz(path, payload)

    buffer = ReplayBuffer.load(path)
    assert len(buffer) == 5
    assert buffer.capacity == 8
    assert buffer.obs_shape == (4,)
    assert buffer.action_dim == 2
