#!/usr/bin/env python3
"""
Measure reproducibility gates and loss drop thresholds for WorldFlux.

Runs DreamerV3 (Atari) and TD-MPC2 (MuJoCo) on bundled datasets with multiple
seeds, then writes per-run results and summary stats to JSON.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from worldflux import create_world_model
from worldflux.training import ReplayBuffer, Trainer, TrainingConfig


def _parse_seeds(seeds: str) -> list[int]:
    return [int(s.strip()) for s in seeds.split(",") if s.strip()]


def _load_atari_buffer(path: str) -> ReplayBuffer:
    data = np.load(path, allow_pickle=False)
    obs = data["obs"]
    actions_onehot = data["actions_onehot"]
    rewards = data["rewards"]
    dones = data["dones"]

    capacity = len(obs)
    obs_shape = obs.shape[1:]
    action_dim = actions_onehot.shape[1]

    buffer = ReplayBuffer(
        capacity=capacity,
        obs_shape=obs_shape,
        action_dim=action_dim,
    )

    episode_starts = [0] + list(np.where(dones[:-1] == 1.0)[0] + 1)
    episode_ends = list(np.where(dones == 1.0)[0] + 1)
    if len(episode_ends) < len(episode_starts):
        episode_ends.append(len(obs))

    num_episodes = min(len(episode_starts), len(episode_ends))
    for i in range(num_episodes):
        start = episode_starts[i]
        end = episode_ends[i]
        if start >= end:
            continue
        buffer.add_episode(
            obs=obs[start:end],
            actions=actions_onehot[start:end],
            rewards=rewards[start:end],
            dones=dones[start:end],
        )

    return buffer


def _load_mujoco_buffer(path: str) -> ReplayBuffer:
    data = np.load(path, allow_pickle=False)
    obs = data["obs"]
    actions = data["actions"]
    rewards = data["rewards"]
    dones = data["dones"]
    obs_dim = int(data["obs_dim"])
    action_dim = int(data["action_dim"])

    capacity = len(obs)
    buffer = ReplayBuffer(
        capacity=capacity,
        obs_shape=(obs_dim,),
        action_dim=action_dim,
    )

    episode_starts = [0] + list(np.where(dones[:-1] == 1.0)[0] + 1)
    episode_ends = list(np.where(dones == 1.0)[0] + 1)
    if len(episode_ends) < len(episode_starts):
        episode_ends.append(len(obs))

    num_episodes = min(len(episode_starts), len(episode_ends))
    for i in range(num_episodes):
        start = episode_starts[i]
        end = episode_ends[i]
        if start >= end:
            continue
        buffer.add_episode(
            obs=obs[start:end],
            actions=actions[start:end],
            rewards=rewards[start:end],
            dones=dones[start:end],
        )

    return buffer


def _loss_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": mean(values), "std": pstdev(values)}


def _tolerance(mean_val: float, std_val: float) -> float:
    return max(2.0 * std_val, abs(mean_val) * 0.10)


def _summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_model.setdefault(run["model"], []).append(run)

    summary: dict[str, Any] = {}
    for model, items in by_model.items():
        final_losses = [item["final_loss"] for item in items]
        loss_drops = [item["loss_drop"] for item in items]
        final_stats = _loss_stats(final_losses)
        drop_stats = _loss_stats(loss_drops)
        tolerance = _tolerance(final_stats["mean"], final_stats["std"])
        drop_threshold = drop_stats["mean"] - 2.0 * drop_stats["std"]
        summary[model] = {
            "seeds": [item["seed"] for item in items],
            "steps": items[0]["steps"] if items else 0,
            "final_loss": {
                "mean": final_stats["mean"],
                "std": final_stats["std"],
                "tolerance": tolerance,
                "range": [final_stats["mean"] - tolerance, final_stats["mean"] + tolerance],
            },
            "loss_drop": {
                "mean": drop_stats["mean"],
                "std": drop_stats["std"],
                "threshold": drop_threshold,
            },
        }
    return summary


def _run_model(
    model_id: str,
    buffer: ReplayBuffer,
    dataset_name: str,
    steps: int,
    batch_size: int,
    seq_len: int,
    seed: int,
    device: str,
    output_dir: Path,
    eval_batches: int,
    log_interval: int,
) -> dict[str, Any]:
    model = create_world_model(
        model_id,
        obs_shape=buffer.obs_shape,
        action_dim=buffer.action_dim,
    )

    config = TrainingConfig(
        total_steps=steps,
        batch_size=batch_size,
        sequence_length=seq_len,
        output_dir=str(output_dir),
        device=device,
        seed=seed,
        log_interval=max(1, log_interval),
        save_interval=steps + 1,
    )

    trainer = Trainer(model, config)

    start_time = time.time()
    initial_metrics = trainer.evaluate(buffer, num_batches=eval_batches)
    trainer.train(buffer)
    final_metrics = trainer.evaluate(buffer, num_batches=eval_batches)
    duration = time.time() - start_time

    initial_loss = float(initial_metrics.get("loss", 0.0))
    final_loss = float(final_metrics.get("loss", 0.0))
    loss_drop = (initial_loss - final_loss) / initial_loss if initial_loss != 0 else 0.0

    return {
        "model": model_id,
        "dataset": dataset_name,
        "obs_shape": buffer.obs_shape,
        "seed": seed,
        "steps": steps,
        "device": device,
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_drop": loss_drop,
        "duration_sec": duration,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure quality gates with multiple seeds")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval-batches", type=int, default=2)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Training log interval (steps). Lower = more frequent progress.",
    )
    parser.add_argument("--output", type=str, default="reports/quality-gates/seed_runs.json")
    parser.add_argument(
        "--models",
        type=str,
        default="dreamer,tdmpc2",
        help="Comma-separated list: dreamer,tdmpc2",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing JSON if present",
    )

    parser.add_argument("--atari-data", type=str, default="atari_data.npz")
    parser.add_argument("--mujoco-data", type=str, default="mujoco_data.npz")

    parser.add_argument("--dreamer-size", type=str, default="ci")
    parser.add_argument("--dreamer-batch-size", type=int, default=4)
    parser.add_argument("--dreamer-seq-len", type=int, default=10)

    parser.add_argument("--tdmpc2-size", type=str, default="ci")
    parser.add_argument("--tdmpc2-batch-size", type=int, default=16)
    parser.add_argument("--tdmpc2-seq-len", type=int, default=10)

    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    atari_buffer = _load_atari_buffer(args.atari_data)
    mujoco_buffer = _load_mujoco_buffer(args.mujoco_data)

    existing: dict[str, Any] | None = None
    if args.append and output_path.exists():
        existing = json.loads(output_path.read_text())
    runs: list[dict[str, Any]] = list(existing.get("runs", [])) if existing else []
    seen = {(r["model"], r["seed"]) for r in runs if "model" in r and "seed" in r}

    if "dreamer" in models:
        for seed in seeds:
            model_id = f"dreamerv3:{args.dreamer_size}"
            if (model_id, seed) in seen:
                continue
            dreamer_out = output_path.parent / f"dreamer_seed_{seed}"
            runs.append(
                _run_model(
                    model_id=model_id,
                    buffer=atari_buffer,
                    dataset_name=args.atari_data,
                    steps=args.steps,
                    batch_size=args.dreamer_batch_size,
                    seq_len=args.dreamer_seq_len,
                    seed=seed,
                    device=args.device,
                    output_dir=dreamer_out,
                    eval_batches=args.eval_batches,
                    log_interval=args.log_interval,
                )
            )
            summary = _summary(runs)
            payload = {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": args.device,
                "steps": args.steps,
                "seeds": seeds,
                "runs": runs,
                "summary": summary,
            }
            output_path.write_text(json.dumps(payload, indent=2))
            print(f"Updated results at {output_path}")

    if "tdmpc2" in models:
        for seed in seeds:
            model_id = f"tdmpc2:{args.tdmpc2_size}"
            if (model_id, seed) in seen:
                continue
            tdmpc_out = output_path.parent / f"tdmpc2_seed_{seed}"
            runs.append(
                _run_model(
                    model_id=model_id,
                    buffer=mujoco_buffer,
                    dataset_name=args.mujoco_data,
                    steps=args.steps,
                    batch_size=args.tdmpc2_batch_size,
                    seq_len=args.tdmpc2_seq_len,
                    seed=seed,
                    device=args.device,
                    output_dir=tdmpc_out,
                    eval_batches=args.eval_batches,
                    log_interval=args.log_interval,
                )
            )
            summary = _summary(runs)
            payload = {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "device": args.device,
                "steps": args.steps,
                "seeds": seeds,
                "runs": runs,
                "summary": summary,
            }
            output_path.write_text(json.dumps(payload, indent=2))
            print(f"Updated results at {output_path}")

    summary = _summary(runs)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": args.device,
        "steps": args.steps,
        "seeds": seeds,
        "runs": runs,
        "summary": summary,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
