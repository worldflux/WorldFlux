#!/usr/bin/env python3
"""TD-MPC2 benchmark with quick/full modes."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from common import (
    add_common_cli,
    build_run_context,
    emit_failure,
    emit_success,
    resolve_mode,
    write_reward_heatmap_ppm,
    write_summary,
)

from worldflux import create_world_model
from worldflux.training import ReplayBuffer, Trainer, TrainingConfig
from worldflux.training.data import create_random_buffer


def _load_mujoco_buffer(path: str) -> ReplayBuffer:
    data = np.load(path, allow_pickle=False)
    obs = data["obs"]
    actions = data["actions"]
    rewards = data["rewards"]
    dones = data["dones"]
    obs_dim = int(data["obs_dim"])
    action_dim = int(data["action_dim"])

    buffer = ReplayBuffer(capacity=len(obs), obs_shape=(obs_dim,), action_dim=action_dim)

    starts = [0] + list(np.where(dones[:-1] == 1.0)[0] + 1)
    ends = list(np.where(dones == 1.0)[0] + 1)
    if len(ends) < len(starts):
        ends.append(len(obs))

    for start, end in zip(starts, ends, strict=False):
        if start >= end:
            continue
        buffer.add_episode(
            obs=obs[start:end],
            actions=actions[start:end],
            rewards=rewards[start:end],
            dones=dones[start:end],
        )
    return buffer


def main() -> int:
    parser = argparse.ArgumentParser(description="TD-MPC2 benchmark")
    add_common_cli(parser, default_output_dir="outputs/benchmarks/tdmpc2-mujoco")
    parser.add_argument("--data", type=str, default="mujoco_data.npz")
    args = parser.parse_args()

    mode = resolve_mode(args)
    context = build_run_context(scenario="benchmark_tdmpc2_mujoco", mode=mode)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    heatmap_path = output_dir / "imagination.ppm"

    try:
        if mode == "quick":
            buffer = create_random_buffer(
                capacity=2500,
                obs_shape=(39,),
                action_dim=6,
                num_episodes=30,
                episode_length=40,
                seed=args.seed,
            )
            total_steps = 8
            batch_size = 64
            seq_len = 10
        else:
            buffer = _load_mujoco_buffer(args.data)
            total_steps = 60
            batch_size = 128
            seq_len = 10

        model = create_world_model(
            "tdmpc2:ci", obs_shape=buffer.obs_shape, action_dim=buffer.action_dim
        )
        trainer = Trainer(
            model,
            TrainingConfig(
                total_steps=total_steps,
                batch_size=batch_size,
                sequence_length=seq_len,
                output_dir=str(output_dir),
                device="cpu",
                seed=args.seed,
                log_interval=max(1, total_steps // 4),
                save_interval=total_steps + 1,
            ),
        )

        initial_loss = float(trainer.evaluate(buffer, num_batches=2)["loss"])
        trainer.train(buffer)
        final_loss = float(trainer.evaluate(buffer, num_batches=2)["loss"])

        horizon = 12 if mode == "quick" else 30
        batch = buffer.sample(batch_size=1, seq_len=horizon + 1, device="cpu")
        state = model.encode(batch.obs[:, 0])
        actions = batch.actions[:, :horizon].permute(1, 0, 2)
        rollout = model.rollout(state, actions)
        rewards = rollout.rewards.detach().cpu().numpy().reshape(-1)
        write_reward_heatmap_ppm(rewards, heatmap_path)

        success = bool(
            np.isfinite(initial_loss) and np.isfinite(final_loss) and rollout.actions.shape[0] > 0
        )
        summary = {
            "benchmark": "tdmpc2-mujoco",
            "mode": mode,
            "seed": int(args.seed),
            "model": "tdmpc2:ci",
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_improved": bool(final_loss < initial_loss),
            "runtime_sec": float(time.time() - float(context["started_at"])),
            "success": success,
            "artifacts": {
                "summary": str(summary_path),
                "imagination": str(heatmap_path),
            },
        }
        write_summary(summary_path, summary)

        if success:
            emit_success(
                context,
                ttfi_sec=float(time.time() - float(context["started_at"])),
                artifacts={"summary": str(summary_path), "imagination": str(heatmap_path)},
            )
            return 0

        emit_failure(
            context, error="benchmark_validation_failed", artifacts={"summary": str(summary_path)}
        )
        print(summary)
        return 1
    except Exception as exc:  # pragma: no cover - runtime guard
        emit_failure(context, error=str(exc), artifacts={"summary": str(summary_path)})
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
