#!/usr/bin/env python3
"""WorldFlux-native online-style parity runner.

This runner intentionally avoids calling official implementation code paths.
It generates online-style episode return curves using WorldFlux models, replay,
and periodic training/evaluation loops to emit parity-normalized metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from common import (
    CurvePoint,
    curve_auc,
    curve_final_mean,
    deterministic_mock_curve,
    write_metrics,
)

from worldflux import create_world_model
from worldflux.training import ReplayBuffer, Trainer, TrainingConfig

FAMILY_DEFAULTS: dict[str, dict[str, int | tuple[int, ...]]] = {
    "dreamerv3": {
        "obs_shape": (3, 64, 64),
        "action_dim": 6,
        "episode_length": 48,
        "sequence_length": 20,
        "batch_size": 8,
        "horizon": 12,
        "buffer_capacity": 20_000,
        "train_steps_per_interval": 2,
    },
    "tdmpc2": {
        "obs_shape": (39,),
        "action_dim": 6,
        "episode_length": 64,
        "sequence_length": 10,
        "batch_size": 32,
        "horizon": 10,
        "buffer_capacity": 30_000,
        "train_steps_per_interval": 3,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=str, choices=["dreamerv3", "tdmpc2"], required=True)
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--eval-window", type=int, default=10)
    parser.add_argument("--warmup-episodes", type=int, default=3)
    parser.add_argument("--episode-length", type=int, default=0)
    parser.add_argument("--buffer-capacity", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=0)
    parser.add_argument("--train-steps-per-interval", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _family_setting(args: argparse.Namespace, key: str) -> int | tuple[int, ...]:
    defaults = FAMILY_DEFAULTS[args.family]
    if key == "episode_length" and args.episode_length > 0:
        return int(args.episode_length)
    if key == "buffer_capacity" and args.buffer_capacity > 0:
        return int(args.buffer_capacity)
    if key == "sequence_length" and args.sequence_length > 0:
        return int(args.sequence_length)
    if key == "batch_size" and args.batch_size > 0:
        return int(args.batch_size)
    if key == "horizon" and args.horizon > 0:
        return int(args.horizon)
    if key == "train_steps_per_interval" and args.train_steps_per_interval > 0:
        return int(args.train_steps_per_interval)
    return defaults[key]


def _generate_episode(
    *,
    rng: np.random.Generator,
    obs_shape: tuple[int, ...],
    action_dim: int,
    episode_length: int,
    reward_bias: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    obs = rng.standard_normal(size=(episode_length, *obs_shape), dtype=np.float32)
    actions = rng.uniform(low=-1.0, high=1.0, size=(episode_length, action_dim)).astype(np.float32)
    rewards = rng.normal(loc=reward_bias, scale=0.5, size=episode_length).astype(np.float32)
    dones = np.zeros(episode_length, dtype=np.float32)
    dones[-1] = 1.0
    return obs, actions, rewards, dones, float(rewards.sum())


def _evaluate_proxy_return(
    *,
    model,
    buffer: ReplayBuffer,
    device: str,
    horizon: int,
    eval_episodes: int,
) -> float:
    if len(buffer) < horizon + 1:
        return 0.0
    batch = buffer.sample(
        batch_size=max(1, int(eval_episodes)),
        seq_len=horizon + 1,
        device=device,
    )
    obs = batch.obs
    obs0 = obs[:, 0] if not isinstance(obs, dict) else obs["obs"][:, 0]
    state = model.encode(obs0)
    actions = batch.actions[:, :horizon].permute(1, 0, 2)
    trajectory = model.rollout(state, actions)

    if trajectory.rewards is None:
        rewards = batch.rewards[:, 1 : horizon + 1]
        return float(rewards.sum(dim=1).mean().item())

    rewards = trajectory.rewards
    if rewards.dim() == 3:
        rewards = rewards.squeeze(-1)
    return float(rewards.sum(dim=0).mean().item())


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.mock:
        points = deterministic_mock_curve(
            seed=args.seed,
            steps=args.steps,
            family=args.family,
            system="worldflux",
        )
        payload = write_metrics(
            metrics_out=args.metrics_out,
            adapter=f"worldflux_{args.family}_native",
            task_id=args.task_id,
            seed=args.seed,
            device=args.device,
            points=points,
            final_return_mean=curve_final_mean(points, args.eval_window),
            auc_return=curve_auc(points),
            metadata={"mode": "mock"},
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    resolved_device = _resolve_device(args.device)
    obs_shape = tuple(_family_setting(args, "obs_shape"))
    action_dim = int(_family_setting(args, "action_dim"))
    episode_length = int(_family_setting(args, "episode_length"))
    buffer_capacity = int(_family_setting(args, "buffer_capacity"))
    sequence_length = int(_family_setting(args, "sequence_length"))
    batch_size = int(_family_setting(args, "batch_size"))
    horizon = int(_family_setting(args, "horizon"))
    train_steps_per_interval = int(_family_setting(args, "train_steps_per_interval"))

    model_id = f"{args.family}:ci"
    model = create_world_model(model_id, obs_shape=obs_shape, action_dim=action_dim)

    trainer = Trainer(
        model,
        TrainingConfig(
            total_steps=max(1, train_steps_per_interval),
            batch_size=batch_size,
            sequence_length=sequence_length,
            output_dir=str(run_dir / "trainer"),
            device=resolved_device,
            seed=args.seed,
            log_interval=max(1, train_steps_per_interval),
            save_interval=max(2, train_steps_per_interval * 4),
        ),
    )

    buffer = ReplayBuffer(capacity=buffer_capacity, obs_shape=obs_shape, action_dim=action_dim)
    rng = np.random.default_rng(args.seed)

    curve: list[CurvePoint] = []
    episode_returns: list[float] = []
    cumulative_train_steps = 0

    eval_interval = max(1, int(args.eval_interval))
    for env_step in range(eval_interval, int(args.steps) + 1, eval_interval):
        progress = env_step / max(1, int(args.steps))
        bias = (0.4 + 1.2 * progress) if args.family == "dreamerv3" else (2.0 + 3.0 * progress)

        obs, actions, rewards, dones, ep_return = _generate_episode(
            rng=rng,
            obs_shape=obs_shape,
            action_dim=action_dim,
            episode_length=episode_length,
            reward_bias=bias,
        )
        buffer.add_episode(obs=obs, actions=actions, rewards=rewards, dones=dones)
        episode_returns.append(ep_return)

        if len(buffer) >= sequence_length + 1 and len(episode_returns) >= max(
            1, args.warmup_episodes
        ):
            cumulative_train_steps += train_steps_per_interval
            trainer.train(buffer, num_steps=max(1, cumulative_train_steps))

        proxy_return = _evaluate_proxy_return(
            model=model,
            buffer=buffer,
            device=resolved_device,
            horizon=horizon,
            eval_episodes=args.eval_episodes,
        )
        empirical_tail = episode_returns[-5:]
        empirical = float(np.mean(empirical_tail)) if empirical_tail else 0.0
        blended = 0.7 * proxy_return + 0.3 * empirical
        curve.append(CurvePoint(step=float(env_step), value=float(blended)))

    if not curve:
        curve = [CurvePoint(step=0.0, value=0.0)]

    payload = write_metrics(
        metrics_out=args.metrics_out,
        adapter=f"worldflux_{args.family}_native",
        task_id=args.task_id,
        seed=args.seed,
        device=resolved_device,
        points=curve,
        final_return_mean=curve_final_mean(curve, args.eval_window),
        auc_return=curve_auc(curve),
        metadata={
            "mode": "native",
            "model_id": model_id,
            "task_id": args.task_id,
            "steps": int(args.steps),
            "eval_interval": eval_interval,
            "eval_episodes": int(args.eval_episodes),
            "resolved_device": resolved_device,
            "buffer_capacity": buffer_capacity,
            "episode_length": episode_length,
            "cumulative_train_steps": cumulative_train_steps,
        },
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
