#!/usr/bin/env python3
"""Diffusion world-model benchmark with one-command imagination artifact generation."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
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
from worldflux.core.batch import Batch
from worldflux.training import Trainer, TrainingConfig


class RandomDiffusionProvider:
    """Generate random diffusion batches for lightweight benchmarking."""

    def __init__(self, *, obs_dim: int, action_dim: int, seed: int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def sample(
        self,
        batch_size: int,
        seq_len: int | None = None,
        device: str | torch.device = "cpu",
    ) -> Batch:
        del seq_len
        obs = torch.randn(batch_size, self.obs_dim, generator=self.generator, device=device)
        target = torch.randn(batch_size, self.obs_dim, generator=self.generator, device=device)
        actions = torch.randn(batch_size, self.action_dim, generator=self.generator, device=device)
        return Batch(obs=obs, actions=actions, target=target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diffusion imagination benchmark")
    add_common_cli(parser, default_output_dir="outputs/benchmarks/diffusion-imagination")
    parser.add_argument("--obs-dim", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=2)
    args = parser.parse_args()

    mode = resolve_mode(args)
    context = build_run_context(scenario="benchmark_diffusion_imagination", mode=mode)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    image_path = output_dir / "imagination.ppm"

    try:
        total_steps = 12 if mode == "quick" else 120
        model = create_world_model(
            "diffusion:base",
            obs_shape=(args.obs_dim,),
            action_dim=args.action_dim,
            hidden_dim=64,
            diffusion_steps=2,
        )

        trainer = Trainer(
            model,
            TrainingConfig(
                total_steps=total_steps,
                batch_size=32,
                sequence_length=1,
                output_dir=str(output_dir),
                device="cpu",
                seed=args.seed,
                log_interval=max(1, total_steps // 4),
                save_interval=total_steps + 1,
            ),
        )
        provider = RandomDiffusionProvider(
            obs_dim=args.obs_dim, action_dim=args.action_dim, seed=args.seed
        )

        initial_loss = float(trainer.evaluate(provider, num_batches=2)["loss"])
        trainer.train(provider)
        final_loss = float(trainer.evaluate(provider, num_batches=2)["loss"])

        horizon = 16 if mode == "quick" else 64
        obs = torch.zeros(1, args.obs_dim)
        state = model.encode(obs)
        rewards = []
        for _ in range(horizon):
            action = torch.randn(1, args.action_dim)
            state = model.transition(state, action)
            decoded = model.decode(state)
            pred = decoded.preds["obs"]
            score = -float(torch.mean(torch.square(pred)).item())
            rewards.append(score)

        write_reward_heatmap_ppm(np.asarray(rewards, dtype=np.float32), image_path)
        success = bool(
            np.isfinite(initial_loss) and np.isfinite(final_loss) and len(rewards) == horizon
        )

        summary = {
            "benchmark": "diffusion-imagination",
            "mode": mode,
            "seed": int(args.seed),
            "model": "diffusion:base",
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_improved": bool(final_loss < initial_loss),
            "runtime_sec": float(time.time() - float(context["started_at"])),
            "success": success,
            "artifacts": {
                "summary": str(summary_path),
                "imagination": str(image_path),
            },
        }
        write_summary(summary_path, summary)

        if success:
            emit_success(
                context,
                ttfi_sec=float(time.time() - float(context["started_at"])),
                artifacts={"summary": str(summary_path), "imagination": str(image_path)},
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
