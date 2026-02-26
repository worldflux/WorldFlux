"""Additional safety and boundary tests for training data providers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from worldflux.core.batch import Batch
from worldflux.core.exceptions import BufferError, ConfigurationError, ShapeMismatchError
from worldflux.training import ReplayBuffer, TrajectoryDataset
from worldflux.training.data import TokenSequenceProvider, TransitionArrayProvider


@pytest.mark.parametrize(
    ("capacity", "obs_shape", "action_dim", "error_pattern"),
    [
        (0, (4,), 2, "capacity must be positive"),
        (8, (0,), 2, "obs_shape must have positive dimensions"),
        (8, (4,), 0, "action_dim must be positive"),
    ],
)
def test_replay_buffer_rejects_invalid_constructor_values(
    capacity: int,
    obs_shape: tuple[int, ...],
    action_dim: int,
    error_pattern: str,
) -> None:
    with pytest.raises(ConfigurationError, match=error_pattern):
        ReplayBuffer(capacity=capacity, obs_shape=obs_shape, action_dim=action_dim)


def test_replay_buffer_add_episode_handles_empty_and_rank_validation() -> None:
    buffer = ReplayBuffer(capacity=16, obs_shape=(4,), action_dim=2)
    empty_obs = np.zeros((0, 4), dtype=np.float32)
    empty_actions = np.zeros((0, 2), dtype=np.float32)
    empty_rewards = np.zeros((0,), dtype=np.float32)
    buffer.add_episode(empty_obs, empty_actions, empty_rewards)
    assert len(buffer) == 0

    obs = np.zeros((3, 4), dtype=np.float32)
    actions = np.zeros((3, 2), dtype=np.float32)
    rewards_bad_rank = np.zeros((3, 1), dtype=np.float32)
    with pytest.raises(ShapeMismatchError, match="Rewards must be a 1D array"):
        buffer.add_episode(obs, actions, rewards_bad_rank)

    rewards = np.zeros((3,), dtype=np.float32)
    dones_bad_rank = np.zeros((3, 1), dtype=np.float32)
    with pytest.raises(ShapeMismatchError, match="Dones must be a 1D array"):
        buffer.add_episode(obs, actions, rewards, dones=dones_bad_rank)


def test_replay_buffer_sampling_fallback_paths_and_overwrite_guards() -> None:
    buffer = ReplayBuffer(capacity=6, obs_shape=(2,), action_dim=1)
    buffer._size = 5

    # Fallback path when no episode boundaries are available.
    buffer._episode_starts = []
    buffer._episode_ends = []
    with pytest.warns(UserWarning, match="No episode boundaries"):
        indices = buffer._sample_valid_indices(batch_size=4, seq_len=2)
    assert len(indices) == 4
    assert all(0 <= idx <= 3 for idx in indices)
    assert buffer._is_overwritten(ep_start=0, ep_end=1, new_end=3) is False

    # Fallback path when episodes exist but none are long enough for seq_len.
    buffer._episode_starts = [0]
    buffer._episode_ends = [1]
    with pytest.warns(UserWarning, match="No episodes long enough"):
        indices = buffer._sample_valid_indices(batch_size=3, seq_len=2)
    assert len(indices) == 3
    assert all(0 <= idx <= 3 for idx in indices)

    # Wrapped episodes are considered overwritten when the ring is full.
    buffer._size = buffer.capacity
    assert buffer._is_overwritten(ep_start=5, ep_end=1, new_end=2) is True


def _valid_buffer_payload() -> dict[str, np.ndarray]:
    return {
        "obs": np.zeros((4, 3), dtype=np.float32),
        "actions": np.zeros((4, 2), dtype=np.float32),
        "rewards": np.zeros((4,), dtype=np.float32),
        "dones": np.zeros((4,), dtype=np.float32),
        "obs_shape": np.array([3], dtype=np.int64),
        "action_dim": np.array(2, dtype=np.int64),
        "capacity": np.array(8, dtype=np.int64),
    }


@pytest.mark.parametrize(
    ("mutator", "error_pattern"),
    [
        (lambda payload: payload.update({"obs_shape": np.array(3, dtype=np.int64)}), "obs_shape"),
        (
            lambda payload: payload.update({"action_dim": np.array([2, 3], dtype=np.int64)}),
            "action_dim",
        ),
        (
            lambda payload: payload.update({"capacity": np.array([8, 9], dtype=np.int64)}),
            "capacity",
        ),
    ],
)
def test_replay_buffer_load_rejects_invalid_metadata(
    tmp_path: Path,
    mutator,
    error_pattern: str,
) -> None:
    path = tmp_path / "invalid_metadata.npz"
    payload = _valid_buffer_payload()
    mutator(payload)
    np.savez(path, **payload)

    with pytest.raises(BufferError, match=error_pattern):
        ReplayBuffer.load(path)


def test_replay_buffer_from_trajectories_rejects_empty_input() -> None:
    with pytest.raises(BufferError, match="empty trajectories"):
        ReplayBuffer.from_trajectories([])


def test_trajectory_dataset_rejects_non_tensor_observations() -> None:
    class _BadBuffer:
        def sample(self, batch_size: int, seq_len: int, device: str = "cpu") -> Batch:
            del batch_size, seq_len, device
            return Batch(obs=np.zeros((1, 4), dtype=np.float32))

    dataset = TrajectoryDataset(_BadBuffer(), seq_len=3, samples_per_epoch=1)
    with pytest.raises(ShapeMismatchError, match="expects tensor observations"):
        _ = dataset[0]


def test_token_sequence_provider_validates_and_samples() -> None:
    with pytest.raises(ConfigurationError, match="rank-2"):
        TokenSequenceProvider(np.arange(6, dtype=np.int64))

    provider = TokenSequenceProvider(np.arange(30, dtype=np.int64).reshape(5, 6))
    assert provider.batch_layout() == {"obs": "BT", "target": "BT"}

    with pytest.raises(ConfigurationError, match="batch_size"):
        provider.sample(batch_size=0)
    with pytest.raises(ConfigurationError, match="seq_len"):
        provider.sample(batch_size=2, seq_len=7)

    batch = provider.sample(batch_size=2, seq_len=4, device="cpu")
    assert batch.obs is not None and batch.obs.shape == (2, 4)
    assert batch.target is not None and batch.target.shape == (2, 4)


def test_transition_array_provider_validates_inputs_and_sampling() -> None:
    with pytest.raises(ConfigurationError, match="obs must be rank-2"):
        TransitionArrayProvider(obs=np.zeros((8,), dtype=np.float32), actions=np.zeros((8, 2)))
    with pytest.raises(ConfigurationError, match="actions must be rank-2"):
        TransitionArrayProvider(obs=np.zeros((8, 3), dtype=np.float32), actions=np.zeros((8,)))
    with pytest.raises(ShapeMismatchError, match="length mismatch"):
        TransitionArrayProvider(obs=np.zeros((8, 3), dtype=np.float32), actions=np.zeros((7, 2)))

    provider = TransitionArrayProvider(
        obs=np.zeros((10, 3), dtype=np.float32),
        actions=np.zeros((10, 2), dtype=np.float32),
        rewards=np.zeros((10,), dtype=np.float32),
        next_obs=np.ones((10, 3), dtype=np.float32),
        terminations=np.zeros((10,), dtype=np.float32),
    )
    assert provider.batch_layout()["obs"] == "B..."

    with pytest.raises(ConfigurationError, match="batch_size must be positive"):
        provider.sample(batch_size=0)

    batch = provider.sample(batch_size=4, device="cpu")
    assert batch.obs is not None and batch.obs.shape[0] == 4
    assert batch.actions is not None and batch.actions.shape[0] == 4
    assert batch.rewards is not None and batch.rewards.shape[0] == 4
    assert batch.next_obs is not None and batch.next_obs.shape[0] == 4
    assert batch.terminations is not None and batch.terminations.shape[0] == 4
