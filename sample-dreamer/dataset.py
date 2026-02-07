from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from worldflux.training import ReplayBuffer
from worldflux.training.data import create_random_buffer

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

PhaseCallback = Callable[[str, str | None], None]
FrameCallback = Callable[[Any, int, int, float, bool], None]


def _emit_phase(callback: PhaseCallback | None, phase: str, detail: str | None = None) -> None:
    if callback is None:
        return
    try:
        callback(phase, detail)
    except Exception:
        # Dashboard callbacks should never interrupt training data creation.
        return


def _emit_frame(
    callback: FrameCallback | None,
    frame: Any,
    episode: int,
    episode_step: int,
    reward: float,
    done: bool,
) -> None:
    if callback is None:
        return
    try:
        callback(frame, episode, episode_step, reward, done)
    except Exception:
        # Dashboard callbacks should never interrupt training data creation.
        return


def _load_project_config(path: str = "worldflux.toml") -> dict[str, Any]:
    with Path(path).open("rb") as f:
        return tomllib.load(f)


def _parse_obs_shape(value: Any) -> tuple[int, ...]:
    if isinstance(value, list | tuple) and value:
        return tuple(int(dim) for dim in value)
    raise ValueError(f"Invalid obs_shape: {value!r}")


def _fit_vector(value: Any, dim: int) -> np.ndarray:
    flat = np.asarray(value, dtype=np.float32).reshape(-1)
    out = np.zeros(dim, dtype=np.float32)
    size = min(dim, flat.size)
    out[:size] = flat[:size]
    return out


def _fit_visual_obs(obs: Any, target_shape: tuple[int, ...]) -> np.ndarray:
    target_c, target_h, target_w = target_shape
    arr = np.asarray(obs)
    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.ndim == 3 and arr.shape[0] not in (1, 3, 4):
        arr = arr.transpose(2, 0, 1)
    if arr.ndim != 3:
        arr = np.resize(arr, target_shape)

    if arr.shape[0] < target_c:
        pad = np.zeros((target_c - arr.shape[0], arr.shape[1], arr.shape[2]), dtype=arr.dtype)
        arr = np.concatenate([arr, pad], axis=0)
    arr = arr[:target_c]
    if arr.shape[1:] != (target_h, target_w):
        arr = np.resize(arr, target_shape)

    arr = arr.astype(np.float32)
    if np.max(arr) > 1.0:
        arr /= 255.0
    return arr


def _one_hot(action: int, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    vec[action % dim] = 1.0
    return vec


def _collect_atari_buffer(
    *,
    gym_env: str,
    obs_shape: tuple[int, ...],
    action_dim: int,
    num_episodes: int,
    episode_length: int,
    capacity: int,
    frame_callback: FrameCallback | None = None,
    phase_callback: PhaseCallback | None = None,
    frame_fps: int = 8,
) -> ReplayBuffer | None:
    try:
        import ale_py
        import gymnasium as gym

        gym.register_envs(ale_py)
    except Exception as exc:  # pragma: no cover - depends on optional extras
        warnings.warn(f"Atari gym collection is unavailable: {exc}")
        _emit_phase(
            phase_callback,
            "unavailable",
            "Install gymnasium + ale-py to enable live gameplay.",
        )
        return None

    try:
        env = gym.make(gym_env, render_mode=None)
    except Exception as exc:  # pragma: no cover - depends on optional extras
        warnings.warn(f"Failed to create Atari env '{gym_env}': {exc}")
        _emit_phase(phase_callback, "unavailable", f"Failed to create Atari env: {gym_env}")
        return None

    _emit_phase(phase_callback, "collecting")
    buffer = ReplayBuffer(capacity=capacity, obs_shape=obs_shape, action_dim=action_dim)
    min_frame_interval = 1.0 / max(1, int(frame_fps))
    last_frame_time = 0.0

    for episode_idx in range(num_episodes):
        obs, _ = env.reset()
        ep_obs: list[np.ndarray] = []
        ep_actions: list[np.ndarray] = []
        ep_rewards: list[float] = []
        ep_dones: list[float] = []
        done = False
        steps = 0

        while not done and steps < episode_length:
            action = int(env.action_space.sample())
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_obs.append(_fit_visual_obs(obs, obs_shape))
            ep_actions.append(_one_hot(action, action_dim))
            ep_rewards.append(float(reward))
            ep_dones.append(float(done))
            now = time.monotonic()
            if now - last_frame_time >= min_frame_interval or done:
                _emit_frame(
                    frame_callback,
                    next_obs,
                    episode_idx + 1,
                    steps + 1,
                    float(reward),
                    done,
                )
                last_frame_time = now
            obs = next_obs
            steps += 1

        if ep_obs:
            buffer.add_episode(
                obs=np.asarray(ep_obs, dtype=np.float32),
                actions=np.asarray(ep_actions, dtype=np.float32),
                rewards=np.asarray(ep_rewards, dtype=np.float32),
                dones=np.asarray(ep_dones, dtype=np.float32),
            )
    env.close()
    return buffer if len(buffer) > 0 else None


def _collect_mujoco_buffer(
    *,
    gym_env: str,
    obs_shape: tuple[int, ...],
    action_dim: int,
    num_episodes: int,
    episode_length: int,
    capacity: int,
    phase_callback: PhaseCallback | None = None,
) -> ReplayBuffer | None:
    try:
        import gymnasium as gym
    except Exception as exc:  # pragma: no cover - depends on optional extras
        warnings.warn(f"MuJoCo gym collection is unavailable: {exc}")
        return None

    try:
        env = gym.make(gym_env)
    except Exception as exc:  # pragma: no cover - depends on optional extras
        warnings.warn(f"Failed to create MuJoCo env '{gym_env}': {exc}")
        return None

    _emit_phase(phase_callback, "collecting")
    obs_dim = int(obs_shape[0])
    buffer = ReplayBuffer(capacity=capacity, obs_shape=(obs_dim,), action_dim=action_dim)
    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_obs: list[np.ndarray] = []
        ep_actions: list[np.ndarray] = []
        ep_rewards: list[float] = []
        ep_dones: list[float] = []
        done = False
        steps = 0

        while not done and steps < episode_length:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_obs.append(_fit_vector(obs, obs_dim))
            ep_actions.append(_fit_vector(action, action_dim))
            ep_rewards.append(float(reward))
            ep_dones.append(float(done))
            obs = next_obs
            steps += 1

        if ep_obs:
            buffer.add_episode(
                obs=np.asarray(ep_obs, dtype=np.float32),
                actions=np.asarray(ep_actions, dtype=np.float32),
                rewards=np.asarray(ep_rewards, dtype=np.float32),
                dones=np.asarray(ep_dones, dtype=np.float32),
            )
    env.close()
    return buffer if len(buffer) > 0 else None


def get_demo_buffer(
    model_config: Any,
    frame_callback: FrameCallback | None = None,
    phase_callback: PhaseCallback | None = None,
) -> ReplayBuffer:
    """Return a replay buffer that works out-of-the-box."""
    cfg = _load_project_config()
    architecture = cfg.get("architecture", {})
    data_cfg = cfg.get("data", {})
    gameplay_cfg = cfg.get("gameplay", {})
    if not isinstance(gameplay_cfg, dict):
        gameplay_cfg = {}

    default_obs = getattr(model_config, "obs_shape", (3, 64, 64))
    default_action_dim = int(getattr(model_config, "action_dim", 6))
    obs_shape = _parse_obs_shape(architecture.get("obs_shape", default_obs))
    action_dim = int(architecture.get("action_dim", default_action_dim))

    source = str(data_cfg.get("source", "random")).strip().lower()
    num_episodes = max(1, int(data_cfg.get("num_episodes", 100)))
    episode_length = max(2, int(data_cfg.get("episode_length", 100)))
    capacity = max(
        int(data_cfg.get("buffer_capacity", 10000)),
        num_episodes * episode_length,
    )
    gym_env = str(data_cfg.get("gym_env", "")).strip()
    environment = str(cfg.get("environment", "custom")).strip().lower()
    gameplay_enabled = bool(gameplay_cfg.get("enabled", True))
    gameplay_fps = max(1, int(gameplay_cfg.get("fps", 8)))
    atari_frame_callback = frame_callback if gameplay_enabled else None

    if source == "gym":
        buffer: ReplayBuffer | None
        if environment == "atari":
            buffer = _collect_atari_buffer(
                gym_env=gym_env or "ALE/Breakout-v5",
                obs_shape=obs_shape,
                action_dim=action_dim,
                num_episodes=num_episodes,
                episode_length=episode_length,
                capacity=capacity,
                frame_callback=atari_frame_callback,
                phase_callback=phase_callback,
                frame_fps=gameplay_fps,
            )
        elif environment == "mujoco":
            if gameplay_enabled:
                _emit_phase(
                    phase_callback,
                    "unavailable",
                    "Live gameplay stream is implemented for Atari in this sample.",
                )
            buffer = _collect_mujoco_buffer(
                gym_env=gym_env or "HalfCheetah-v5",
                obs_shape=obs_shape,
                action_dim=action_dim,
                num_episodes=num_episodes,
                episode_length=episode_length,
                capacity=capacity,
                phase_callback=phase_callback,
            )
        elif len(obs_shape) >= 3:
            buffer = _collect_atari_buffer(
                gym_env=gym_env or "ALE/Breakout-v5",
                obs_shape=obs_shape,
                action_dim=action_dim,
                num_episodes=num_episodes,
                episode_length=episode_length,
                capacity=capacity,
                frame_callback=atari_frame_callback,
                phase_callback=phase_callback,
                frame_fps=gameplay_fps,
            )
        else:
            if gameplay_enabled:
                _emit_phase(
                    phase_callback,
                    "unavailable",
                    "Live gameplay stream is implemented for Atari in this sample.",
                )
            buffer = _collect_mujoco_buffer(
                gym_env=gym_env or "HalfCheetah-v5",
                obs_shape=obs_shape,
                action_dim=action_dim,
                num_episodes=num_episodes,
                episode_length=episode_length,
                capacity=capacity,
                phase_callback=phase_callback,
            )
        if buffer is not None:
            return buffer
        if gameplay_enabled:
            _emit_phase(
                phase_callback,
                "unavailable",
                "Gym collection failed. Falling back to random replay buffer.",
            )
        warnings.warn("Falling back to random buffer because gym collection failed.")
    elif gameplay_enabled:
        _emit_phase(
            phase_callback,
            "unavailable",
            "Set data.source='gym' to enable live gameplay stream.",
        )

    return create_random_buffer(
        capacity=capacity,
        obs_shape=obs_shape,
        action_dim=action_dim,
        num_episodes=num_episodes,
        episode_length=episode_length,
        seed=42,
    )
