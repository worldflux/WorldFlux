"""WorldFlux native TD-MPC2 parity agent using online environment interaction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from worldflux import create_world_model
from worldflux.planners import CEMPlanner
from worldflux.training import ReplayBuffer, Trainer, TrainingConfig

from .dmcontrol_env import DMControlEnvError, build_dmcontrol_env


@dataclass(frozen=True)
class TDMPC2NativeRunConfig:
    task_id: str
    seed: int
    steps: int
    eval_interval: int
    eval_episodes: int
    eval_window: int
    env_backend: str
    device: str
    run_dir: Path
    buffer_capacity: int = 300_000
    warmup_steps: int = 2048
    train_steps_per_eval: int = 96
    sequence_length: int = 10
    batch_size: int = 64
    max_episode_steps: int = 1000
    policy_mode: str = "diagnostic_random"
    cem_horizon: int = 5
    cem_num_samples: int = 128
    cem_num_elites: int = 16
    cem_iterations: int = 2


def _evaluate_policy(
    *,
    task_id: str,
    backend: str,
    seed: int,
    num_episodes: int,
    max_episode_steps: int,
    policy_mode: str,
    model: Any,
    planner: CEMPlanner | None,
    device: str,
) -> float:
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    normalized_policy_mode = str(policy_mode).strip().lower()
    if normalized_policy_mode == "parity_candidate" and planner is None:
        raise DMControlEnvError("CEM planner is required for parity_candidate evaluation.")
    was_training = bool(getattr(model, "training", False))
    if was_training:
        model.eval()

    try:
        for episode in range(max(1, num_episodes)):
            env = build_dmcontrol_env(
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
                    if normalized_policy_mode == "parity_candidate":
                        assert planner is not None
                        action = _select_tdmpc2_cem_action(
                            model=model,
                            planner=planner,
                            obs=obs,
                            action_low=env.action_low,
                            action_high=env.action_high,
                            device=device,
                            torch_seed=seed + episode * 100_000 + steps,
                        )
                    else:
                        action = env.sample_action(rng)
                    _next_obs, reward, terminated, truncated, _info = env.step(action)
                    obs = _next_obs
                    ep_return += float(reward)
                    done = bool(terminated or truncated)
                    steps += 1
                returns.append(ep_return)
            finally:
                env.close()
    finally:
        if was_training:
            model.train()

    return float(np.mean(returns)) if returns else 0.0


def _obs_to_tensor(obs: np.ndarray, *, device: str) -> torch.Tensor:
    return torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device=device)


def _select_tdmpc2_cem_action(
    *,
    model: Any,
    planner: CEMPlanner,
    obs: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    device: str,
    torch_seed: int,
) -> np.ndarray:
    with torch.no_grad():
        torch.manual_seed(int(torch_seed))
        obs_tensor = _obs_to_tensor(obs, device=device)
        state = model.encode(obs_tensor)
        action_payload = planner.plan(model, state)
        if action_payload.tensor is None:
            raise DMControlEnvError("CEM planner returned empty action tensor.")
        action_tensor = action_payload.tensor
        if action_tensor.ndim == 3:
            selected = action_tensor[0, 0]
        elif action_tensor.ndim == 2:
            selected = action_tensor[0]
        elif action_tensor.ndim == 1:
            selected = action_tensor
        else:
            raise DMControlEnvError(
                f"Unexpected CEM action tensor shape: {tuple(action_tensor.shape)}"
            )
        action = selected.detach().cpu().numpy().astype(np.float32)
    return np.clip(action, action_low, action_high).astype(np.float32)


def run_tdmpc2_native(
    config: TDMPC2NativeRunConfig,
) -> tuple[list[tuple[float, float]], dict[str, Any]]:
    env = build_dmcontrol_env(
        task_id=config.task_id,
        seed=config.seed,
        backend=config.env_backend,
        max_episode_steps=config.max_episode_steps,
    )

    model_id = "tdmpc2:ci"
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
    policy_mode = str(config.policy_mode).strip().lower()
    if policy_mode not in {"diagnostic_random", "parity_candidate"}:
        raise DMControlEnvError(
            "policy_mode must be either 'diagnostic_random' or 'parity_candidate', "
            f"got {config.policy_mode!r}"
        )
    planner: CEMPlanner | None = None
    if policy_mode == "parity_candidate":
        planner = CEMPlanner(
            horizon=max(1, int(config.cem_horizon)),
            action_dim=env.action_dim,
            num_samples=max(2, int(config.cem_num_samples)),
            num_elites=max(1, int(config.cem_num_elites)),
            iterations=max(1, int(config.cem_iterations)),
            action_low=env.action_low,
            action_high=env.action_high,
        )

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
                if policy_mode == "parity_candidate":
                    assert planner is not None
                    action = _select_tdmpc2_cem_action(
                        model=model,
                        planner=planner,
                        obs=obs,
                        action_low=env.action_low,
                        action_high=env.action_high,
                        device=config.device,
                        torch_seed=config.seed + env_steps,
                    )
                else:
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

                    eval_return = _evaluate_policy(
                        task_id=config.task_id,
                        backend=config.env_backend,
                        seed=config.seed + 10_000 + int(next_eval),
                        num_episodes=config.eval_episodes,
                        max_episode_steps=config.max_episode_steps,
                        policy_mode=policy_mode,
                        model=model,
                        planner=planner,
                        device=config.device,
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
        eval_return = _evaluate_policy(
            task_id=config.task_id,
            backend=config.env_backend,
            seed=config.seed + 20_000,
            num_episodes=config.eval_episodes,
            max_episode_steps=config.max_episode_steps,
            policy_mode=policy_mode,
            model=model,
            planner=planner,
            device=config.device,
        )
        curve.append((float(config.steps), float(eval_return)))

    metadata: dict[str, Any] = {
        "mode": "native_real_env",
        "family": "tdmpc2",
        "task_id": config.task_id,
        "model_id": model_id,
        "policy": "learned" if policy_mode == "parity_candidate" else "random",
        "policy_mode": config.policy_mode,
        "policy_impl": "cem_planner" if policy_mode == "parity_candidate" else "random_env_sampler",
        "eval_policy": "learned" if policy_mode == "parity_candidate" else "random",
        "eval_policy_impl": "cem_planner_eval"
        if policy_mode == "parity_candidate"
        else "random_env_sampler_eval",
        "env_backend": config.env_backend,
        "obs_shape": list(env.obs_shape),
        "action_dim": int(env.action_dim),
        "cem_horizon": int(config.cem_horizon),
        "cem_num_samples": int(config.cem_num_samples),
        "cem_num_elites": int(config.cem_num_elites),
        "cem_iterations": int(config.cem_iterations),
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


__all__ = ["TDMPC2NativeRunConfig", "run_tdmpc2_native", "DMControlEnvError"]
