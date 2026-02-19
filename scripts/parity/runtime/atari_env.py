"""Atari runtime environment wrapper for parity native runs."""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class AtariEnvError(RuntimeError):
    """Raised when Atari environment setup fails."""


_ATARI_GAME_MAP: dict[str, str] = {
    "alien": "Alien",
    "amidar": "Amidar",
    "assault": "Assault",
    "asterix": "Asterix",
    "bank_heist": "BankHeist",
    "battle_zone": "BattleZone",
    "boxing": "Boxing",
    "breakout": "Breakout",
    "chopper_command": "ChopperCommand",
    "crazy_climber": "CrazyClimber",
    "demon_attack": "DemonAttack",
    "freeway": "Freeway",
    "frostbite": "Frostbite",
    "gopher": "Gopher",
    "hero": "Hero",
    "jamesbond": "Jamesbond",
    "kangaroo": "Kangaroo",
    "krull": "Krull",
    "kung_fu_master": "KungFuMaster",
    "ms_pacman": "MsPacman",
    "pong": "Pong",
    "private_eye": "PrivateEye",
    "qbert": "Qbert",
    "road_runner": "RoadRunner",
    "seaquest": "Seaquest",
    "up_n_down": "UpNDown",
}


@dataclass
class AtariEnv:
    """Atari environment abstraction used by parity native agents."""

    backend: str
    task_id: str
    obs_shape: tuple[int, ...]
    action_dim: int
    max_episode_steps: int
    _env: Any
    _seed: int
    _rng: np.random.Generator

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        if self.backend == "stub":
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            self._env["step"] = 0
            return _stub_obs(self._rng)

        obs, _info = self._env.reset(seed=seed)
        return _preprocess_atari_obs(obs)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.backend == "stub":
            self._env["step"] += 1
            done = bool(self._env["step"] >= self.max_episode_steps)
            target = self._env["step"] % max(1, self.action_dim)
            reward = 1.0 if int(action) == int(target) else -0.05
            return _stub_obs(self._rng), float(reward), done, False, {"target_action": int(target)}

        obs, reward, terminated, truncated, info = self._env.step(int(action))
        return (
            _preprocess_atari_obs(obs),
            float(reward),
            bool(terminated),
            bool(truncated),
            dict(info) if isinstance(info, dict) else {},
        )

    def sample_action(self, rng: np.random.Generator) -> int:
        if self.backend == "stub":
            return int(rng.integers(0, max(1, self.action_dim)))
        return int(self._env.action_space.sample())

    def to_model_action(self, action: int) -> np.ndarray:
        out = np.zeros((self.action_dim,), dtype=np.float32)
        out[int(action)] = 1.0
        return out

    def close(self) -> None:
        if self.backend == "stub":
            return
        close_fn = getattr(self._env, "close", None)
        if callable(close_fn):
            close_fn()


def _canonical_game_name(task_id: str) -> str:
    if task_id.startswith("atari100k_"):
        game = task_id[len("atari100k_") :]
    elif task_id.startswith("atari_"):
        game = task_id[len("atari_") :]
    else:
        raise AtariEnvError(
            f"Atari task_id must start with 'atari100k_' or 'atari_'; got {task_id!r}."
        )
    game_key = game.strip().lower()
    if game_key not in _ATARI_GAME_MAP:
        raise AtariEnvError(
            f"Unsupported Atari task {task_id!r}. Known games: {sorted(_ATARI_GAME_MAP)}"
        )
    return _ATARI_GAME_MAP[game_key]


def _ale_env_id(task_id: str) -> str:
    return f"ALE/{_canonical_game_name(task_id)}-v5"


def _preprocess_atari_obs(obs: Any, size: tuple[int, int] = (64, 64)) -> np.ndarray:
    arr = np.asarray(obs)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim != 3:
        raise AtariEnvError(f"Expected Atari observation with rank 2 or 3, got shape {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]

    tensor = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor / 255.0
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).contiguous().cpu().numpy().astype(np.float32)


def _stub_obs(rng: np.random.Generator) -> np.ndarray:
    return rng.random((3, 64, 64), dtype=np.float32)


def build_atari_env(
    *,
    task_id: str,
    seed: int,
    backend: str,
    max_episode_steps: int = 108_000,
) -> AtariEnv:
    """Build Atari environment backend for native parity runs."""
    backend_normalized = backend.strip().lower()
    if backend_normalized not in {"auto", "gymnasium", "stub"}:
        raise AtariEnvError("backend must be one of: auto, gymnasium, stub")

    if backend_normalized == "stub":
        return AtariEnv(
            backend="stub",
            task_id=task_id,
            obs_shape=(3, 64, 64),
            action_dim=6,
            max_episode_steps=max(1, int(max_episode_steps)),
            _env={"step": 0},
            _seed=seed,
            _rng=np.random.default_rng(seed),
        )

    try:
        import gymnasium as gym
    except ModuleNotFoundError as exc:
        raise AtariEnvError(
            "gymnasium is required for Atari parity runs. Install gymnasium + ale-py, "
            "or use backend=stub for local smoke tests."
        ) from exc

    try:
        import ale_py  # type: ignore
    except ModuleNotFoundError as exc:
        raise AtariEnvError(
            "ale-py is required for Atari parity runs. Install gymnasium + ale-py, "
            "or use backend=stub for local smoke tests."
        ) from exc

    # Gymnasium requires explicit ALE namespace registration on some versions.
    register_envs = getattr(gym, "register_envs", None)
    if callable(register_envs):  # pragma: no branch - simple compatibility guard
        register_envs(ale_py)

    env_id = _ale_env_id(task_id)
    try:
        env = gym.make(
            env_id,
            frameskip=4,
            repeat_action_probability=0.0,
            full_action_space=False,
            max_episode_steps=max(1, int(max_episode_steps)),
        )
    except Exception as exc:  # pragma: no cover - external dependency behavior
        raise AtariEnvError(f"failed to create Atari env {env_id!r}: {exc}") from exc

    obs, _info = env.reset(seed=seed)
    obs_pre = _preprocess_atari_obs(obs)

    action_space_n = getattr(getattr(env, "action_space", None), "n", None)
    if not isinstance(action_space_n, numbers.Integral) or int(action_space_n) <= 0:
        env.close()
        raise AtariEnvError(f"Atari action_space.n must be positive int, got {action_space_n!r}")

    return AtariEnv(
        backend="gymnasium",
        task_id=task_id,
        obs_shape=tuple(obs_pre.shape),
        action_dim=int(action_space_n),
        max_episode_steps=max(1, int(max_episode_steps)),
        _env=env,
        _seed=seed,
        _rng=np.random.default_rng(seed),
    )
