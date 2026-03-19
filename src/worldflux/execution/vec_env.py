# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Vectorized environment wrappers for parallel data collection.

Provides a unified interface over gymnasium's SyncVectorEnv and
AsyncVectorEnv for batched environment stepping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VecEnvStep:
    """Result of a vectorized environment step.

    Attributes:
        obs: Batched observations, shape [num_envs, *obs_shape].
        rewards: Batched rewards, shape [num_envs].
        terminated: Batched termination flags, shape [num_envs].
        truncated: Batched truncation flags, shape [num_envs].
        infos: Per-environment info dicts.
    """

    obs: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    infos: dict[str, Any]


class VecEnvWrapper:
    """Gymnasium VectorEnv wrapper for parallel data collection.

    Wraps gymnasium.vector.SyncVectorEnv or AsyncVectorEnv to provide
    batched step/reset operations for efficient data collection.

    Auto mode selects sync for num_envs <= 4 and async for num_envs > 4.

    Args:
        env_id: Gymnasium environment ID (e.g. "CartPole-v1").
        num_envs: Number of parallel environments.
        mode: Parallelism mode - "sync", "async", or "auto".
        seed: Base random seed (each env gets seed + env_index).

    Example:
        >>> vec_env = VecEnvWrapper("CartPole-v1", num_envs=8, mode="auto")
        >>> obs = vec_env.reset()
        >>> for _ in range(100):
        ...     actions = np.random.randint(0, 2, size=8)
        ...     step = vec_env.step(actions)
        ...     # step.obs, step.rewards, step.terminated, etc.
        >>> vec_env.close()
    """

    def __init__(
        self,
        env_id: str,
        num_envs: int = 4,
        mode: Literal["sync", "async", "auto"] = "auto",
        seed: int | None = None,
    ) -> None:
        try:
            import gymnasium
        except ImportError as exc:
            raise ImportError(
                "VecEnvWrapper requires gymnasium. Install with: pip install gymnasium"
            ) from exc

        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")

        self.env_id = env_id
        self.num_envs = num_envs
        self._seed = seed

        # Resolve mode
        if mode == "auto":
            resolved_mode = "async" if num_envs > 4 else "sync"
        else:
            resolved_mode = mode
        self.mode = resolved_mode

        # Create the vectorized environment
        env_fns = [self._make_env_fn(env_id, i) for i in range(num_envs)]

        self._vec_env: Any
        if resolved_mode == "async":
            self._vec_env = gymnasium.vector.AsyncVectorEnv(env_fns)
        else:
            self._vec_env = gymnasium.vector.SyncVectorEnv(env_fns)

        # Episode tracking
        self._episode_lengths: np.ndarray = np.zeros(num_envs, dtype=np.int32)
        self._episode_rewards: np.ndarray = np.zeros(num_envs, dtype=np.float32)
        self._completed_episodes: int = 0

        logger.info(
            "VecEnvWrapper created: env=%s, num_envs=%d, mode=%s",
            env_id,
            num_envs,
            resolved_mode,
        )

    def _make_env_fn(self, env_id: str, index: int) -> Any:
        """Create a factory function for a single environment."""
        seed = self._seed

        def _fn() -> Any:
            import gymnasium

            env = gymnasium.make(env_id)
            if seed is not None:
                env.reset(seed=seed + index)
            return env

        return _fn

    @property
    def observation_space(self) -> Any:
        """The shared observation space of all environments."""
        return self._vec_env.single_observation_space

    @property
    def action_space(self) -> Any:
        """The shared action space of all environments."""
        return self._vec_env.single_action_space

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset all environments.

        Args:
            seed: Optional seed for resetting.

        Returns:
            Batched initial observations, shape [num_envs, *obs_shape].
        """
        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        obs_raw: Any
        obs_raw, _info = self._vec_env.reset(**kwargs)

        self._episode_lengths[:] = 0
        self._episode_rewards[:] = 0.0

        return np.asarray(obs_raw)

    def step(self, actions: np.ndarray) -> VecEnvStep:
        """Take a batched step across all environments.

        Environments that terminate are auto-reset by gymnasium's VectorEnv.

        Args:
            actions: Batched actions, shape [num_envs, *action_shape].

        Returns:
            VecEnvStep containing batched results.
        """
        raw: Any = self._vec_env.step(actions)
        obs_arr: np.ndarray = np.asarray(raw[0])
        rewards_arr: np.ndarray = np.asarray(raw[1], dtype=np.float32)
        terminated_arr: np.ndarray = np.asarray(raw[2], dtype=np.bool_)
        truncated_arr: np.ndarray = np.asarray(raw[3], dtype=np.bool_)
        infos_raw: Any = raw[4]

        # Track episode statistics
        self._episode_lengths += 1
        self._episode_rewards += rewards_arr

        done_mask = terminated_arr | truncated_arr
        newly_done = int(np.sum(done_mask))
        if newly_done > 0:
            self._completed_episodes += newly_done
            # Reset trackers for done environments
            self._episode_lengths[done_mask] = 0
            self._episode_rewards[done_mask] = 0.0

        return VecEnvStep(
            obs=obs_arr,
            rewards=rewards_arr,
            terminated=terminated_arr.astype(np.float32),
            truncated=truncated_arr.astype(np.float32),
            infos=infos_raw if isinstance(infos_raw, dict) else {},
        )

    @property
    def completed_episodes(self) -> int:
        """Total number of completed episodes since creation."""
        return self._completed_episodes

    def close(self) -> None:
        """Close all environments and release resources."""
        self._vec_env.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return (
            f"VecEnvWrapper(env={self.env_id!r}, num_envs={self.num_envs}, "
            f"mode={self.mode!r}, episodes={self._completed_episodes})"
        )
