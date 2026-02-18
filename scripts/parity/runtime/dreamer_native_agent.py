"""WorldFlux native Dreamer parity agent using online environment interaction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from worldflux import create_world_model
from worldflux.training import ReplayBuffer, Trainer, TrainingConfig

from .atari_env import AtariEnvError, build_atari_env


@dataclass(frozen=True)
class DreamerNativeRunConfig:
    task_id: str
    seed: int
    steps: int
    eval_interval: int
    eval_episodes: int
    eval_window: int
    env_backend: str
    device: str
    run_dir: Path
    buffer_capacity: int = 200_000
    warmup_steps: int = 1024
    train_steps_per_eval: int = 64
    sequence_length: int = 32
    batch_size: int = 16
    max_episode_steps: int = 27_000
    policy_mode: str = "diagnostic_random"


def _evaluate_random_policy(
    *,
    task_id: str,
    backend: str,
    seed: int,
    num_episodes: int,
    max_episode_steps: int,
) -> float:
    """Evaluate a deterministic random policy on real environment episodes."""
    rng = np.random.default_rng(seed)
    returns: list[float] = []

    for episode in range(max(1, num_episodes)):
        env = build_atari_env(
            task_id=task_id,
            seed=seed + episode,
            backend=backend,
            max_episode_steps=max_episode_steps,
        )
        try:
            obs = env.reset(seed=seed + episode)
            _ = obs
            done = False
            ep_return = 0.0
            steps = 0
            while not done and steps < max_episode_steps:
                action = env.sample_action(rng)
                _next_obs, reward, terminated, truncated, _info = env.step(action)
                ep_return += float(reward)
                done = bool(terminated or truncated)
                steps += 1
            returns.append(ep_return)
        finally:
            env.close()

    return float(np.mean(returns)) if returns else 0.0


def run_dreamer_native(
    config: DreamerNativeRunConfig,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    """Run native online interaction + training loop and return evaluation curve."""
    env = build_atari_env(
        task_id=config.task_id,
        seed=config.seed,
        backend=config.env_backend,
        max_episode_steps=config.max_episode_steps,
    )

    model_id = "dreamerv3:ci"
    model = create_world_model(model_id, obs_shape=env.obs_shape, action_dim=env.action_dim)

    trainer = Trainer(
        model,
        TrainingConfig(
            total_steps=max(1, int(config.train_steps_per_eval)),
            batch_size=max(1, int(config.batch_size)),
            sequence_length=max(2, int(config.sequence_length)),
            output_dir=str((config.run_dir / "trainer").resolve()),
            device=config.device,
            seed=int(config.seed),
            log_interval=max(1, int(config.train_steps_per_eval) // 4),
            save_interval=max(2, int(config.train_steps_per_eval)),
        ),
    )

    buffer = ReplayBuffer(
        capacity=max(1, int(config.buffer_capacity)),
        obs_shape=tuple(env.obs_shape),
        action_dim=int(env.action_dim),
    )

    rng = np.random.default_rng(config.seed)
    curve: list[tuple[float, float]] = []

    env_steps = 0
    train_target_steps = 0
    train_updates = 0
    episodes = 0
    next_eval = max(1, int(config.eval_interval))

    try:
        while env_steps < int(config.steps):
            obs = env.reset(seed=config.seed + episodes)
            ep_obs: list[np.ndarray] = []
            ep_actions: list[np.ndarray] = []
            ep_rewards: list[float] = []
            ep_dones: list[float] = []

            done = False
            ep_len = 0
            while not done and env_steps < int(config.steps):
                action = env.sample_action(rng)
                model_action = env.to_model_action(action)
                next_obs, reward, terminated, truncated, _ = env.step(action)

                ep_obs.append(np.asarray(obs, dtype=np.float32))
                ep_actions.append(np.asarray(model_action, dtype=np.float32))
                ep_rewards.append(float(reward))
                ep_dones.append(float(terminated or truncated))

                obs = next_obs
                env_steps += 1
                ep_len += 1
                done = bool(terminated or truncated)

                while env_steps >= next_eval:
                    if len(buffer) >= max(2, int(config.sequence_length) + 1) and env_steps >= int(
                        config.warmup_steps
                    ):
                        train_target_steps += max(1, int(config.train_steps_per_eval))
                        trainer.train(buffer, num_steps=train_target_steps)
                        train_updates += max(1, int(config.train_steps_per_eval))

                    eval_return = _evaluate_random_policy(
                        task_id=config.task_id,
                        backend=config.env_backend,
                        seed=config.seed + 10_000 + int(next_eval),
                        num_episodes=config.eval_episodes,
                        max_episode_steps=config.max_episode_steps,
                    )
                    curve.append((float(env_steps), float(eval_return)))
                    next_eval += max(1, int(config.eval_interval))

                if ep_len >= int(config.max_episode_steps):
                    break

            if ep_obs:
                dones = np.asarray(ep_dones, dtype=np.float32)
                dones[-1] = 1.0
                buffer.add_episode(
                    obs=np.stack(ep_obs, axis=0).astype(np.float32),
                    actions=np.stack(ep_actions, axis=0).astype(np.float32),
                    rewards=np.asarray(ep_rewards, dtype=np.float32),
                    dones=dones,
                )
                episodes += 1
    finally:
        env.close()

    if not curve:
        eval_return = _evaluate_random_policy(
            task_id=config.task_id,
            backend=config.env_backend,
            seed=config.seed + 20_000,
            num_episodes=config.eval_episodes,
            max_episode_steps=config.max_episode_steps,
        )
        curve.append((float(config.steps), float(eval_return)))

    metadata: dict[str, Any] = {
        "mode": "native_real_env",
        "family": "dreamerv3",
        "task_id": config.task_id,
        "model_id": model_id,
        "policy": "random",
        "policy_mode": config.policy_mode,
        "policy_impl": "random_env_sampler",
        "env_backend": config.env_backend,
        "obs_shape": list(env.obs_shape),
        "action_dim": int(env.action_dim),
        "buffer_capacity": int(config.buffer_capacity),
        "warmup_steps": int(config.warmup_steps),
        "train_steps_per_eval": int(config.train_steps_per_eval),
        "train_updates_executed": int(train_updates),
        "env_steps_collected": int(env_steps),
        "episodes_collected": int(episodes),
        "eval_interval": int(config.eval_interval),
        "eval_episodes": int(config.eval_episodes),
        "eval_window": int(config.eval_window),
    }

    return curve, metadata


__all__ = ["DreamerNativeRunConfig", "run_dreamer_native", "AtariEnvError"]
