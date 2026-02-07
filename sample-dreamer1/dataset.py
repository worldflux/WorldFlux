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
CleanupFn = Callable[[], None]


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


class OnlineAtariBatchProvider:
    """BatchProvider that alternates online Atari rollout and replay sampling."""

    def __init__(
        self,
        *,
        gym_env: str,
        obs_shape: tuple[int, ...],
        action_dim: int,
        capacity: int,
        warmup_transitions: int,
        collect_steps_per_update: int,
        max_episode_steps: int,
        frame_callback: FrameCallback | None = None,
        phase_callback: PhaseCallback | None = None,
        frame_fps: int = 8,
    ):
        self._obs_shape = obs_shape
        self._action_dim = action_dim
        self._collect_steps_per_update = max(1, int(collect_steps_per_update))
        self._warmup_transitions = max(1, int(warmup_transitions))
        self._max_episode_steps = max(8, int(max_episode_steps))
        self._frame_callback = frame_callback
        self._phase_callback = phase_callback

        self._buffer = ReplayBuffer(capacity=capacity, obs_shape=obs_shape, action_dim=action_dim)
        self._episode_obs: list[np.ndarray] = []
        self._episode_actions: list[np.ndarray] = []
        self._episode_rewards: list[float] = []
        self._episode_dones: list[float] = []
        self._episode_step = 0
        self._episode_index = 0

        self._min_frame_interval = 1.0 / max(1, int(frame_fps))
        self._last_frame_time = 0.0
        self._closed = False

        try:
            import ale_py
            import gymnasium as gym

            gym.register_envs(ale_py)
            self._env = gym.make(gym_env, render_mode=None)
        except Exception as exc:  # pragma: no cover - depends on optional extras
            raise RuntimeError(
                "Install gymnasium + ale-py to enable live gameplay. "
                f"(failed to initialize Atari env '{gym_env}': {exc})"
            ) from exc

        self._obs, _ = self._env.reset()
        _emit_phase(self._phase_callback, "collecting")
        self._collect_until(self._warmup_transitions)

    def _flush_episode(self) -> None:
        if not self._episode_obs:
            return
        self._buffer.add_episode(
            obs=np.asarray(self._episode_obs, dtype=np.float32),
            actions=np.asarray(self._episode_actions, dtype=np.float32),
            rewards=np.asarray(self._episode_rewards, dtype=np.float32),
            dones=np.asarray(self._episode_dones, dtype=np.float32),
        )
        self._episode_obs.clear()
        self._episode_actions.clear()
        self._episode_rewards.clear()
        self._episode_dones.clear()

    def _step_once(self) -> None:
        action = int(self._env.action_space.sample())
        next_obs, reward, terminated, truncated, _ = self._env.step(action)
        done = bool(terminated or truncated)

        self._episode_obs.append(_fit_visual_obs(self._obs, self._obs_shape))
        self._episode_actions.append(_one_hot(action, self._action_dim))
        self._episode_rewards.append(float(reward))
        self._episode_dones.append(float(done))

        self._episode_step += 1

        now = time.monotonic()
        if now - self._last_frame_time >= self._min_frame_interval or done:
            _emit_frame(
                self._frame_callback,
                next_obs,
                self._episode_index + 1,
                self._episode_step,
                float(reward),
                done,
            )
            self._last_frame_time = now

        boundary = done or self._episode_step >= self._max_episode_steps
        if boundary:
            self._flush_episode()
            self._episode_index += 1
            self._episode_step = 0
            self._obs, _ = self._env.reset()
        else:
            self._obs = next_obs

    def _collect_steps(self, steps: int) -> None:
        for _ in range(max(0, int(steps))):
            self._step_once()

    def _collect_until(self, min_transitions: int) -> None:
        target = max(1, int(min_transitions))
        guard = 0
        max_guard = max(target * 8, self._max_episode_steps * 8, 2048)
        while len(self._buffer) < target and guard < max_guard:
            self._step_once()
            guard += 1

        if len(self._buffer) < target:
            if self._episode_obs:
                self._flush_episode()
            if len(self._buffer) < target:
                raise RuntimeError(
                    f"Unable to collect enough online transitions: have {len(self._buffer)}, need {target}."
                )

    def sample(self, batch_size: int, seq_len: int, device: str = "cpu"):
        if self._closed:
            raise RuntimeError("OnlineAtariBatchProvider is already closed")

        needed = max(int(seq_len), 1)
        ready_target = max(self._warmup_transitions, needed)
        if len(self._buffer) < ready_target:
            self._collect_until(ready_target)

        self._collect_steps(self._collect_steps_per_update)

        if len(self._buffer) < needed:
            self._collect_until(needed)

        return self._buffer.sample(batch_size=batch_size, seq_len=seq_len, device=device)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._env.close()
        except Exception:
            return


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


def _resolve_data_settings(model_config: Any) -> dict[str, Any]:
    cfg = _load_project_config()
    architecture = cfg.get("architecture", {})
    data_cfg = cfg.get("data", {})
    gameplay_cfg = cfg.get("gameplay", {})
    if not isinstance(gameplay_cfg, dict):
        gameplay_cfg = {}
    online_cfg = cfg.get("online_collection", {})
    if not isinstance(online_cfg, dict):
        online_cfg = {}

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

    model_type = str(cfg.get("model_type", "")).strip().lower()
    online_default = environment == "atari" and model_type.startswith("dreamer")

    return {
        "obs_shape": obs_shape,
        "action_dim": action_dim,
        "source": source,
        "num_episodes": num_episodes,
        "episode_length": episode_length,
        "capacity": capacity,
        "gym_env": gym_env,
        "environment": environment,
        "gameplay_enabled": bool(gameplay_cfg.get("enabled", True)),
        "gameplay_fps": max(1, int(gameplay_cfg.get("fps", 8))),
        "online_enabled": bool(online_cfg.get("enabled", online_default)),
        "warmup_transitions": max(32, int(online_cfg.get("warmup_transitions", 512))),
        "collect_steps_per_update": max(1, int(online_cfg.get("collect_steps_per_update", 64))),
        "max_episode_steps": max(8, int(online_cfg.get("max_episode_steps", episode_length))),
    }


def get_demo_buffer(
    model_config: Any,
    frame_callback: FrameCallback | None = None,
    phase_callback: PhaseCallback | None = None,
) -> ReplayBuffer:
    """Return a replay buffer that works out-of-the-box."""
    settings = _resolve_data_settings(model_config)

    source = settings["source"]
    obs_shape = settings["obs_shape"]
    action_dim = settings["action_dim"]
    num_episodes = settings["num_episodes"]
    episode_length = settings["episode_length"]
    capacity = settings["capacity"]
    gym_env = settings["gym_env"]
    environment = settings["environment"]
    gameplay_enabled = settings["gameplay_enabled"]
    gameplay_fps = settings["gameplay_fps"]

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


def build_training_data(
    model_config: Any,
    frame_callback: FrameCallback | None = None,
    phase_callback: PhaseCallback | None = None,
) -> tuple[Any, CleanupFn, str]:
    """Build either online provider or offline replay buffer for Trainer.train()."""
    settings = _resolve_data_settings(model_config)

    source = settings["source"]
    environment = settings["environment"]
    gameplay_enabled = settings["gameplay_enabled"]
    online_enabled = settings["online_enabled"]

    if (
        source == "gym"
        and online_enabled
        and (environment == "atari" or len(settings["obs_shape"]) >= 3)
    ):
        if gameplay_enabled:
            _emit_phase(phase_callback, "collecting")
        try:
            provider = OnlineAtariBatchProvider(
                gym_env=settings["gym_env"] or "ALE/Breakout-v5",
                obs_shape=settings["obs_shape"],
                action_dim=settings["action_dim"],
                capacity=settings["capacity"],
                warmup_transitions=settings["warmup_transitions"],
                collect_steps_per_update=settings["collect_steps_per_update"],
                max_episode_steps=settings["max_episode_steps"],
                frame_callback=frame_callback if gameplay_enabled else None,
                phase_callback=phase_callback,
                frame_fps=settings["gameplay_fps"],
            )
            return provider, provider.close, "online"
        except Exception as exc:
            warnings.warn(f"Online Atari collection is unavailable: {exc}")
            _emit_phase(phase_callback, "unavailable", str(exc))

    if source == "gym" and online_enabled and environment == "mujoco":
        _emit_phase(
            phase_callback,
            "unavailable",
            "Online collection loop is currently implemented for Atari only.",
        )

    buffer = get_demo_buffer(
        model_config,
        frame_callback=frame_callback,
        phase_callback=phase_callback,
    )

    mode = "offline" if source == "gym" else "random"
    return buffer, (lambda: None), mode
