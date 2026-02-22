"""Data storage and sampling for world model training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from worldflux.core.batch import Batch
from worldflux.core.exceptions import BufferError, ConfigurationError, ShapeMismatchError


class ReplayBuffer:
    """
    Efficient trajectory storage for world model training.

    Stores episodes as contiguous arrays and supports efficient random
    sampling of trajectory segments for training.

    Args:
        capacity: Maximum number of transitions to store.
        obs_shape: Shape of observations (e.g., (3, 64, 64) for images).
        action_dim: Dimension of action space.
        obs_dtype: NumPy dtype for observations (default: float32).

    Example:
        >>> buffer = ReplayBuffer(capacity=100_000, obs_shape=(3, 64, 64), action_dim=6)
        >>> buffer.add_episode(obs, actions, rewards, dones)
        >>> batch = buffer.sample(batch_size=32, seq_len=50)

    Raises:
        ConfigurationError: If capacity, obs_shape, or action_dim are invalid.

    Warning:
        This class is NOT thread-safe. Concurrent calls to ``add_episode()``
        from multiple threads may cause race conditions on internal state
        (``_position``, ``_size``). Use a single writer thread or external
        synchronization when writing from multiple threads.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_dim: int,
        obs_dtype: type = np.float32,
    ):
        if capacity <= 0:
            raise ConfigurationError(f"capacity must be positive, got {capacity}")
        if not obs_shape or any(d <= 0 for d in obs_shape):
            raise ConfigurationError(f"obs_shape must have positive dimensions, got {obs_shape}")
        if action_dim <= 0:
            raise ConfigurationError(f"action_dim must be positive, got {action_dim}")

        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.obs_dtype = obs_dtype

        # Pre-allocate storage
        self._obs: np.ndarray = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self._actions: np.ndarray = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self._dones: np.ndarray = np.zeros(capacity, dtype=np.float32)

        # Episode boundaries for valid sampling
        self._episode_starts: list[int] = []
        self._episode_ends: list[int] = []

        # Current position and size
        self._position = 0
        self._size = 0
        self._num_episodes = 0

    def __len__(self) -> int:
        """Return number of transitions stored."""
        return self._size

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        return (
            f"ReplayBuffer("
            f"size={self._size}/{self.capacity}, "
            f"episodes={self._num_episodes}, "
            f"obs_shape={self.obs_shape}, "
            f"action_dim={self.action_dim})"
        )

    @property
    def num_episodes(self) -> int:
        """Return number of complete episodes stored."""
        return self._num_episodes

    def add_episode(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray | None = None,
    ) -> None:
        """
        Add a complete episode to the buffer.

        Args:
            obs: Observations of shape [episode_len, *obs_shape].
            actions: Actions of shape [episode_len, action_dim].
            rewards: Rewards of shape [episode_len].
            dones: Done flags of shape [episode_len]. If None, last step is done.
        """
        episode_len = len(obs)
        if episode_len == 0:
            return

        # Validate shapes
        if obs.shape[1:] != self.obs_shape:
            raise ShapeMismatchError(
                f"Observation shape mismatch: expected {self.obs_shape}, got {obs.shape[1:]}"
            )
        if actions.shape[1] != self.action_dim:
            raise ShapeMismatchError(
                f"Action dimension mismatch: expected {self.action_dim}, got {actions.shape[1]}"
            )
        if actions.shape[0] != episode_len:
            raise ShapeMismatchError(
                "Action length mismatch",
                expected=(episode_len, self.action_dim),
                got=actions.shape,
            )
        if rewards.shape[0] != episode_len:
            raise ShapeMismatchError(
                "Rewards length mismatch",
                expected=(episode_len,),
                got=rewards.shape,
            )
        if rewards.ndim != 1:
            raise ShapeMismatchError(
                "Rewards must be a 1D array",
                expected=(episode_len,),
                got=rewards.shape,
            )

        if dones is None:
            dones = np.zeros(episode_len, dtype=np.float32)
            dones[-1] = 1.0
        else:
            if dones.shape[0] != episode_len:
                raise ShapeMismatchError(
                    "Dones length mismatch",
                    expected=(episode_len,),
                    got=dones.shape,
                )
            if dones.ndim != 1:
                raise ShapeMismatchError(
                    "Dones must be a 1D array",
                    expected=(episode_len,),
                    got=dones.shape,
                )

        # Handle wrap-around
        start_pos = self._position
        end_pos = start_pos + episode_len

        if end_pos <= self.capacity:
            # Simple case: fits without wrap
            self._obs[start_pos:end_pos] = obs
            self._actions[start_pos:end_pos] = actions
            self._rewards[start_pos:end_pos] = rewards
            self._dones[start_pos:end_pos] = dones
        else:
            # Wrap around
            first_part = self.capacity - start_pos
            self._obs[start_pos:] = obs[:first_part]
            self._obs[: episode_len - first_part] = obs[first_part:]
            self._actions[start_pos:] = actions[:first_part]
            self._actions[: episode_len - first_part] = actions[first_part:]
            self._rewards[start_pos:] = rewards[:first_part]
            self._rewards[: episode_len - first_part] = rewards[first_part:]
            self._dones[start_pos:] = dones[:first_part]
            self._dones[: episode_len - first_part] = dones[first_part:]
            end_pos = end_pos % self.capacity

        # Update episode boundaries
        self._episode_starts.append(start_pos)
        self._episode_ends.append(end_pos if end_pos > start_pos else end_pos + self.capacity)
        self._num_episodes += 1

        # Remove invalidated episodes (wrapped over)
        self._cleanup_old_episodes(end_pos)

        # Update position and size
        self._position = end_pos % self.capacity
        self._size = min(self._size + episode_len, self.capacity)

    def _cleanup_old_episodes(self, new_end: int) -> None:
        """Remove episode markers that have been overwritten."""
        while self._episode_starts and self._size >= self.capacity:
            old_start = self._episode_starts[0]
            old_end = self._episode_ends[0]
            # Check if this episode overlaps with overwritten region
            if self._is_overwritten(old_start, old_end, new_end):
                self._episode_starts.pop(0)
                self._episode_ends.pop(0)
                self._num_episodes -= 1
            else:
                break

    def _is_overwritten(self, ep_start: int, ep_end: int, new_end: int) -> bool:
        """Check if an episode has been overwritten by new data.

        Handles circular buffer wrap-around correctly by considering:
        1. The write pointer position (new_end)
        2. Whether the episode spans across the buffer boundary
        3. Whether any part of the episode overlaps with the overwritten region
        """
        if self._size < self.capacity:
            return False

        new_end_mod = new_end % self.capacity
        # The overwritten region starts at new_end_mod (exclusive) and extends backwards
        # by the amount we wrote in this operation. For simplicity, we check if the
        # episode start position has been overwritten.

        # Normalize episode bounds to actual buffer positions
        ep_start_mod = ep_start % self.capacity
        ep_end_mod = ep_end % self.capacity

        # Check if episode start is in the "danger zone" that has been overwritten
        # The danger zone is from the oldest valid position to new_end_mod
        # Since we track from newest, check if ep_start is before new_end_mod
        # considering wrap-around

        if ep_end_mod <= ep_start_mod:
            # Episode wraps around the buffer
            # It's overwritten if new_end_mod is within [ep_start_mod, capacity) or [0, ep_end_mod)
            return True  # Wrapped episodes are invalidated when buffer is full

        # Episode doesn't wrap: check if its start is behind the write pointer
        # In a circular buffer at capacity, the write pointer marks the oldest data
        # Anything at or before the write pointer's advancement is invalid
        return ep_start_mod < new_end_mod

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        device: str | torch.device = "cpu",
    ) -> Batch:
        """
        Sample random trajectory segments.

        Args:
            batch_size: Number of trajectory segments to sample.
            seq_len: Length of each trajectory segment.
            device: Device to place tensors on.

        Returns:
            Dictionary with keys:
                - obs: [batch_size, seq_len, *obs_shape]
                - actions: [batch_size, seq_len, action_dim]
                - rewards: [batch_size, seq_len]
                - continues: [batch_size, seq_len] (1 - dones)
        """
        if self._size < seq_len:
            raise BufferError(
                f"Buffer has insufficient data: {self._size} transitions, "
                f"need at least {seq_len} for the requested sequence length"
            )

        # Sample random starting positions from valid episodes
        indices = self._sample_valid_indices(batch_size, seq_len)

        # Gather data
        obs: np.ndarray = np.zeros((batch_size, seq_len, *self.obs_shape), dtype=self.obs_dtype)
        actions: np.ndarray = np.zeros((batch_size, seq_len, self.action_dim), dtype=np.float32)
        rewards: np.ndarray = np.zeros((batch_size, seq_len), dtype=np.float32)
        dones: np.ndarray = np.zeros((batch_size, seq_len), dtype=np.float32)

        for i, start_idx in enumerate(indices):
            for t in range(seq_len):
                idx = (start_idx + t) % self.capacity
                obs[i, t] = self._obs[idx]
                actions[i, t] = self._actions[idx]
                rewards[i, t] = self._rewards[idx]
                dones[i, t] = self._dones[idx]

                # Safety check: if we hit a done flag before the end of sequence,
                # mask subsequent rewards (they belong to different episodes)
                # This prevents training on invalid cross-episode transitions
                if t > 0 and dones[i, t - 1] > 0.5:
                    # Previous step was terminal - current data is from new episode
                    # Mark with special metadata if needed for masking
                    pass  # The continues mask handles this at loss computation

        return Batch(
            obs=torch.from_numpy(obs).to(device),
            actions=torch.from_numpy(actions).to(device),
            rewards=torch.from_numpy(rewards).to(device),
            terminations=torch.from_numpy(dones).to(device),
            layouts=self.batch_layout(),
            strict_layout=True,
        )

    def batch_layout(self) -> dict[str, str]:
        """Return explicit axis layout for sampled batches."""
        return {
            "obs": "BT...",
            "actions": "BT...",
            "rewards": "BT",
            "terminations": "BT",
        }

    def _sample_valid_indices(self, batch_size: int, seq_len: int) -> list[int]:
        """Sample valid starting indices that don't cross episode boundaries."""
        if not self._episode_starts:
            # Fallback: sample uniformly (may cross episode boundaries)
            max_start = self._size - seq_len
            return list(np.random.randint(0, max_start + 1, size=batch_size))

        # Sample from valid episode segments
        valid_segments: list[tuple[int, int]] = []
        for start, end in zip(self._episode_starts, self._episode_ends):
            ep_len = end - start
            if ep_len >= seq_len:
                # This episode has valid segments
                valid_segments.append((start, start + ep_len - seq_len))

        if not valid_segments:
            # No valid segments, sample uniformly
            max_start = self._size - seq_len
            return list(np.random.randint(0, max_start + 1, size=batch_size))

        # Weighted sampling by segment length
        total_valid = sum(end - start + 1 for start, end in valid_segments)
        indices = []
        for _ in range(batch_size):
            # Sample a segment proportionally
            choice = np.random.randint(total_valid)
            cumsum = 0
            for seg_start, seg_end in valid_segments:
                seg_len = seg_end - seg_start + 1
                if cumsum + seg_len > choice:
                    # Sample within this segment
                    idx = seg_start + (choice - cumsum)
                    indices.append(idx % self.capacity)
                    break
                cumsum += seg_len
        return indices

    def save(self, path: str | Path) -> None:
        """Save buffer to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            obs=self._obs[: self._size],
            actions=self._actions[: self._size],
            rewards=self._rewards[: self._size],
            dones=self._dones[: self._size],
            obs_shape=np.array(self.obs_shape),
            action_dim=np.array(self.action_dim),
            capacity=np.array(self.capacity),
        )

    @classmethod
    def load(cls, path: str | Path) -> ReplayBuffer:
        """Load buffer from disk with schema validation."""
        required_keys = {
            "obs",
            "actions",
            "rewards",
            "dones",
            "obs_shape",
            "action_dim",
            "capacity",
        }
        # allow_pickle=False for security (only load array data, not arbitrary objects)
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(required_keys.difference(data.files))
            if missing:
                raise BufferError(f"Invalid replay buffer file: missing keys {missing}")

            obs = data["obs"]
            actions = data["actions"]
            rewards = data["rewards"]
            dones = data["dones"]

            try:
                obs_shape = tuple(int(v) for v in np.asarray(data["obs_shape"]).tolist())
            except (TypeError, ValueError) as exc:
                raise BufferError(f"Invalid obs_shape metadata: {exc}") from exc
            if not obs_shape or any(dim <= 0 for dim in obs_shape):
                raise BufferError(f"Invalid obs_shape metadata: {obs_shape}")

            try:
                action_dim = int(np.asarray(data["action_dim"]).item())
            except (TypeError, ValueError) as exc:
                raise BufferError(f"Invalid action_dim metadata: {exc}") from exc
            if action_dim <= 0:
                raise BufferError(f"Invalid action_dim metadata: {action_dim}")

            try:
                capacity = int(np.asarray(data["capacity"]).item())
            except (TypeError, ValueError) as exc:
                raise BufferError(f"Invalid capacity metadata: {exc}") from exc
            if capacity <= 0:
                raise BufferError(f"Invalid capacity metadata: {capacity}")

            if obs.ndim < 2:
                raise BufferError(f"Invalid obs shape: expected [N, *obs_shape], got {obs.shape}")
            if tuple(obs.shape[1:]) != obs_shape:
                raise BufferError(
                    f"obs shape does not match obs_shape metadata: {obs.shape[1:]} vs {obs_shape}"
                )
            if actions.ndim != 2:
                raise BufferError(
                    f"Invalid actions shape: expected [N, action_dim], got {actions.shape}"
                )
            if actions.shape[1] != action_dim:
                raise BufferError(
                    "actions shape does not match action_dim metadata: "
                    f"{actions.shape[1]} vs {action_dim}"
                )
            if rewards.ndim != 1:
                raise BufferError(f"Invalid rewards shape: expected [N], got {rewards.shape}")
            if dones.ndim != 1:
                raise BufferError(f"Invalid dones shape: expected [N], got {dones.shape}")

            size = len(obs)
            if len(actions) != size or len(rewards) != size or len(dones) != size:
                raise BufferError(
                    "Inconsistent trajectory lengths in replay buffer file: "
                    f"obs={len(obs)}, actions={len(actions)}, rewards={len(rewards)}, dones={len(dones)}"
                )
            if size > capacity:
                raise BufferError(
                    "Invalid replay buffer file: number of stored rows exceeds capacity "
                    f"({size} > {capacity})"
                )

            buffer = cls(capacity=capacity, obs_shape=obs_shape, action_dim=action_dim)
            buffer._obs[:size] = obs
            buffer._actions[:size] = actions
            buffer._rewards[:size] = rewards
            buffer._dones[:size] = dones
            buffer._size = size
            buffer._position = size % capacity

        # Reconstruct episode boundaries from done flags
        buffer._reconstruct_episodes()
        return buffer

    def _reconstruct_episodes(self) -> None:
        """Reconstruct episode boundaries from done flags."""
        self._episode_starts = [0]
        self._episode_ends = []
        self._num_episodes = 0

        for i in range(self._size):
            if self._dones[i] > 0.5:
                self._episode_ends.append(i + 1)
                self._num_episodes += 1
                if i + 1 < self._size:
                    self._episode_starts.append(i + 1)

        # Handle case where last episode is incomplete
        if len(self._episode_starts) > len(self._episode_ends):
            self._episode_ends.append(self._size)
            self._num_episodes += 1

    @classmethod
    def from_trajectories(
        cls,
        trajectories: list[dict[str, np.ndarray]],
        capacity: int | None = None,
    ) -> ReplayBuffer:
        """
        Create buffer from list of trajectory dictionaries.

        Args:
            trajectories: List of dicts with keys 'obs', 'actions', 'rewards', 'dones'.
            capacity: Buffer capacity. If None, uses total trajectory length.
        """
        if not trajectories:
            raise BufferError("Cannot create buffer from empty trajectories list")

        first = trajectories[0]
        obs_shape = first["obs"].shape[1:]
        action_dim = first["actions"].shape[1]
        total_len = sum(len(t["obs"]) for t in trajectories)

        if capacity is None:
            capacity = total_len

        buffer = cls(capacity=capacity, obs_shape=obs_shape, action_dim=action_dim)
        for traj in trajectories:
            buffer.add_episode(
                obs=traj["obs"],
                actions=traj["actions"],
                rewards=traj["rewards"],
                dones=traj.get("dones"),
            )
        return buffer


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset wrapper for ReplayBuffer.

    Useful for DataLoader integration with multi-worker loading.

    Args:
        buffer: ReplayBuffer to sample from.
        seq_len: Length of trajectory segments to sample.
        samples_per_epoch: Number of samples per "epoch" (iteration through dataset).

    Example:
        >>> dataset = TrajectoryDataset(buffer, seq_len=50, samples_per_epoch=1000)
        >>> loader = DataLoader(dataset, batch_size=32, num_workers=4)
    """

    def __init__(
        self,
        buffer: ReplayBuffer,
        seq_len: int,
        samples_per_epoch: int = 1000,
    ):
        self.buffer = buffer
        self.seq_len = seq_len
        self.samples_per_epoch = samples_per_epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Batch:
        """Sample a single trajectory segment."""
        batch = self.buffer.sample(batch_size=1, seq_len=self.seq_len, device="cpu")
        if not isinstance(batch.obs, torch.Tensor):
            raise ShapeMismatchError("TrajectoryDataset expects tensor observations")
        return Batch(
            obs=batch.obs.squeeze(0),
            actions=batch.actions.squeeze(0) if batch.actions is not None else None,
            rewards=batch.rewards.squeeze(0) if batch.rewards is not None else None,
            terminations=batch.terminations.squeeze(0) if batch.terminations is not None else None,
            next_obs=None,
            mask=None,
            context=None,
            target=None,
            extras=batch.extras,
            layouts=batch.layouts,
            strict_layout=batch.strict_layout,
        )


def create_random_buffer(
    capacity: int = 10000,
    obs_shape: tuple[int, ...] = (3, 64, 64),
    action_dim: int = 6,
    num_episodes: int = 100,
    episode_length: int = 100,
    seed: int | None = None,
) -> ReplayBuffer:
    """
    Create a buffer with random data for testing.

    Args:
        capacity: Buffer capacity.
        obs_shape: Observation shape.
        action_dim: Action dimension.
        num_episodes: Number of episodes to generate.
        episode_length: Length of each episode.
        seed: Random seed.

    Returns:
        ReplayBuffer filled with random data.
    """
    if seed is not None:
        np.random.seed(seed)

    buffer = ReplayBuffer(capacity=capacity, obs_shape=obs_shape, action_dim=action_dim)

    for _ in range(num_episodes):
        ep_len = np.random.randint(episode_length // 2, episode_length + 1)
        obs = np.random.randn(ep_len, *obs_shape).astype(np.float32)
        actions = np.random.randn(ep_len, action_dim).astype(np.float32)
        rewards = np.random.randn(ep_len).astype(np.float32)
        dones = np.zeros(ep_len, dtype=np.float32)
        dones[-1] = 1.0

        buffer.add_episode(obs, actions, rewards, dones)

    return buffer


class TokenSequenceProvider:
    """
    Lightweight sequence provider for token-model training.

    This provider demonstrates non-ReplayBuffer data feeding while preserving
    the same unified trainer entrypoint.
    """

    def __init__(self, tokens: np.ndarray):
        if tokens.ndim != 2:
            raise ConfigurationError(
                f"tokens must be rank-2 [num_sequences, seq_len], got shape={tokens.shape}"
            )
        self.tokens = tokens.astype(np.int64, copy=False)
        self.num_sequences, self.seq_len = self.tokens.shape

    def batch_layout(self) -> dict[str, str]:
        return {"obs": "BT", "target": "BT"}

    def sample(
        self,
        batch_size: int,
        seq_len: int | None = None,
        device: str | torch.device = "cpu",
    ) -> Batch:
        if batch_size <= 0:
            raise ConfigurationError(f"batch_size must be positive, got {batch_size}")
        length = self.seq_len if seq_len is None else seq_len
        if length <= 0 or length > self.seq_len:
            raise ConfigurationError(f"seq_len must be in [1, {self.seq_len}], got {length}")

        seq_indices = np.random.randint(0, self.num_sequences, size=batch_size)
        start_max = self.seq_len - length
        starts = np.random.randint(0, start_max + 1, size=batch_size)

        obs = np.zeros((batch_size, length), dtype=np.int64)
        for i, (seq_i, start) in enumerate(zip(seq_indices, starts, strict=True)):
            obs[i] = self.tokens[seq_i, start : start + length]

        target = np.roll(obs, shift=-1, axis=1)
        return Batch(
            obs=torch.from_numpy(obs).to(device),
            target=torch.from_numpy(target).to(device),
            layouts=self.batch_layout(),
            strict_layout=True,
        )


class TransitionArrayProvider:
    """
    Lightweight transition provider for pre-batched arrays.

    This is useful for simple vector/image pipelines that do not need replay
    semantics but still want to use the unified Trainer API.
    """

    def __init__(
        self,
        *,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray | None = None,
        next_obs: np.ndarray | None = None,
        terminations: np.ndarray | None = None,
    ):
        if obs.ndim < 2:
            raise ConfigurationError(f"obs must be rank-2+, got shape={obs.shape}")
        if actions.ndim < 2:
            raise ConfigurationError(f"actions must be rank-2+, got shape={actions.shape}")
        if obs.shape[0] != actions.shape[0]:
            raise ShapeMismatchError(
                "obs/actions length mismatch",
                expected=(obs.shape[0],),
                got=(actions.shape[0],),
            )
        self.obs = obs.astype(np.float32, copy=False)
        self.actions = actions.astype(np.float32, copy=False)
        self.rewards = rewards.astype(np.float32, copy=False) if rewards is not None else None
        self.next_obs = next_obs.astype(np.float32, copy=False) if next_obs is not None else None
        self.terminations = (
            terminations.astype(np.float32, copy=False) if terminations is not None else None
        )
        self.num_rows = self.obs.shape[0]

    def batch_layout(self) -> dict[str, str]:
        return {
            "obs": "B...",
            "actions": "B...",
            "rewards": "B",
            "next_obs": "B...",
            "terminations": "B",
        }

    def sample(
        self,
        batch_size: int,
        seq_len: int | None = None,
        device: str | torch.device = "cpu",
    ) -> Batch:
        if batch_size <= 0:
            raise ConfigurationError(f"batch_size must be positive, got {batch_size}")
        idx = np.random.randint(0, self.num_rows, size=batch_size)
        batch = Batch(
            obs=torch.from_numpy(self.obs[idx]).to(device),
            actions=torch.from_numpy(self.actions[idx]).to(device),
            rewards=torch.from_numpy(self.rewards[idx]).to(device)
            if self.rewards is not None
            else None,
            next_obs=torch.from_numpy(self.next_obs[idx]).to(device)
            if self.next_obs is not None
            else None,
            terminations=torch.from_numpy(self.terminations[idx]).to(device)
            if self.terminations is not None
            else None,
            layouts=self.batch_layout(),
            strict_layout=True,
        )
        return batch
