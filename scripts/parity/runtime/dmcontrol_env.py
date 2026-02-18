"""DMControl runtime environment wrapper for parity native runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class DMControlEnvError(RuntimeError):
    """Raised when DMControl environment setup fails."""


@dataclass
class DMControlEnv:
    """DMControl environment abstraction used by parity native agents."""

    backend: str
    task_id: str
    obs_shape: tuple[int, ...]
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    max_episode_steps: int
    _env: Any
    _step_count: int
    _rng: np.random.Generator

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        if self.backend == "stub":
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            self._step_count = 0
            return _stub_obs(self._rng)

        if seed is not None and hasattr(self._env, "task") and hasattr(self._env.task, "_random"):
            self._env.task._random = np.random.RandomState(seed)  # type: ignore[attr-defined]

        timestep = self._env.reset()
        self._step_count = 0
        return _flatten_observation(timestep.observation)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.backend == "stub":
            self._step_count += 1
            clipped = np.clip(action.astype(np.float32), self.action_low, self.action_high)
            reward = 1.5 - float(np.linalg.norm(clipped) / max(1, self.action_dim))
            done = bool(self._step_count >= self.max_episode_steps)
            return _stub_obs(self._rng), reward, done, False, {}

        clipped = np.clip(action.astype(np.float32), self.action_low, self.action_high)
        timestep = self._env.step(clipped)
        self._step_count += 1
        reward = float(timestep.reward or 0.0)
        terminated = bool(timestep.last())
        truncated = bool(self._step_count >= self.max_episode_steps and not terminated)
        return _flatten_observation(timestep.observation), reward, terminated, truncated, {}

    def sample_action(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.action_low, self.action_high).astype(np.float32)

    def to_model_action(self, action: np.ndarray) -> np.ndarray:
        return np.asarray(action, dtype=np.float32)

    def close(self) -> None:
        close_fn = getattr(self._env, "close", None)
        if callable(close_fn):
            close_fn()


def _parse_task(task_id: str) -> tuple[str, str]:
    if "-" not in task_id:
        raise DMControlEnvError(f"DMControl task_id must be '<domain>-<task>', got {task_id!r}.")
    domain, task = task_id.split("-", 1)
    domain = domain.strip().lower()
    task_name = task.strip().lower().replace("-", "_")
    if not domain or not task_name:
        raise DMControlEnvError(f"Invalid DMControl task_id: {task_id!r}")
    return domain, task_name


def _flatten_observation(observation: Any) -> np.ndarray:
    if isinstance(observation, dict):
        parts = []
        for key in sorted(observation):
            arr = np.asarray(observation[key], dtype=np.float32).reshape(-1)
            parts.append(arr)
        if not parts:
            return np.zeros((1,), dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32)

    arr = np.asarray(observation, dtype=np.float32).reshape(-1)
    return arr.astype(np.float32)


def _stub_obs(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((39,), dtype=np.float32)


def build_dmcontrol_env(
    *,
    task_id: str,
    seed: int,
    backend: str,
    max_episode_steps: int = 1000,
) -> DMControlEnv:
    """Build DMControl backend for native parity runs."""
    backend_normalized = backend.strip().lower()
    if backend_normalized not in {"auto", "dmcontrol", "stub"}:
        raise DMControlEnvError("backend must be one of: auto, dmcontrol, stub")

    if backend_normalized == "stub":
        action_dim = 6
        return DMControlEnv(
            backend="stub",
            task_id=task_id,
            obs_shape=(39,),
            action_dim=action_dim,
            action_low=-np.ones((action_dim,), dtype=np.float32),
            action_high=np.ones((action_dim,), dtype=np.float32),
            max_episode_steps=max(1, int(max_episode_steps)),
            _env=None,
            _step_count=0,
            _rng=np.random.default_rng(seed),
        )

    domain, task_name = _parse_task(task_id)

    try:
        from dm_control import suite
    except ModuleNotFoundError as exc:
        raise DMControlEnvError(
            "dm_control is required for DMControl parity runs. Install dm-control, "
            "or use backend=stub for local smoke tests."
        ) from exc

    try:
        env = suite.load(domain_name=domain, task_name=task_name, task_kwargs={"random": seed})
    except Exception as exc:  # pragma: no cover - external dependency behavior
        raise DMControlEnvError(
            f"failed to create dm_control task domain={domain!r} task={task_name!r}: {exc}"
        ) from exc

    action_spec = env.action_spec()
    action_low = np.asarray(action_spec.minimum, dtype=np.float32).reshape(-1)
    action_high = np.asarray(action_spec.maximum, dtype=np.float32).reshape(-1)
    action_dim = int(action_low.shape[0])

    first = env.reset()
    obs = _flatten_observation(first.observation)

    return DMControlEnv(
        backend="dmcontrol",
        task_id=task_id,
        obs_shape=tuple(obs.shape),
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        max_episode_steps=max(1, int(max_episode_steps)),
        _env=env,
        _step_count=0,
        _rng=np.random.default_rng(seed),
    )
