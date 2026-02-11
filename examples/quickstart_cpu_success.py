#!/usr/bin/env python3
"""Official CPU-first quickstart with machine-checkable success criteria."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from _shared.viz import write_reward_heatmap_ppm

from worldflux import create_world_model
from worldflux.telemetry.wasr import make_run_id, write_event
from worldflux.training import Trainer, TrainingConfig
from worldflux.training.data import create_random_buffer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CPU-first WorldFlux success path")
    parser.add_argument("--quick", action="store_true", help="Run short CI-safe config")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/quickstart_cpu",
        help="Output directory for summary and image artifacts",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    run_id = make_run_id()
    started_at = time.time()
    scenario = "quickstart_cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    image_path = output_dir / "imagination.ppm"

    try:
        total_steps = 8 if args.quick else 48
        horizon = 10 if args.quick else 20

        obs_shape = (8,)
        action_dim = 2
        buffer = create_random_buffer(
            capacity=2000 if args.quick else 8000,
            obs_shape=obs_shape,
            action_dim=action_dim,
            num_episodes=40 if args.quick else 80,
            episode_length=40,
            seed=args.seed,
        )

        model = create_world_model(
            "dreamerv3:ci",
            obs_shape=obs_shape,
            action_dim=action_dim,
            encoder_type="mlp",
            decoder_type="mlp",
            device="cpu",
        )

        trainer = Trainer(
            model,
            TrainingConfig(
                total_steps=total_steps,
                batch_size=16,
                sequence_length=10,
                learning_rate=3e-4,
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

        sample = buffer.sample(batch_size=1, seq_len=horizon + 1, device="cpu")
        state = model.encode(sample.obs[:, 0])
        action_seq = sample.actions[:, :horizon].permute(1, 0, 2)
        rollout = model.rollout(state, action_seq)
        rewards = rollout.rewards.detach().cpu().numpy().reshape(-1)

        ttfi_sec = float(time.time() - started_at)
        write_reward_heatmap_ppm(rewards, image_path)

        finite_ok = bool(np.isfinite(initial_loss) and np.isfinite(final_loss))
        horizon_ok = int(rollout.actions.shape[0]) == horizon
        improved = final_loss < initial_loss
        success = bool(finite_ok and horizon_ok and improved)

        summary = {
            "scenario": scenario,
            "run_id": run_id,
            "seed": int(args.seed),
            "quick": bool(args.quick),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_improved": improved,
            "finite": finite_ok,
            "horizon": horizon,
            "rollout_horizon": int(rollout.actions.shape[0]),
            "artifacts": {
                "summary": str(summary_path),
                "imagination": str(image_path),
            },
            "success": success,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        write_event(
            event="run_complete",
            scenario=scenario,
            run_id=run_id,
            success=success,
            duration_sec=float(time.time() - started_at),
            ttfi_sec=ttfi_sec,
            artifacts={"summary": str(summary_path), "imagination": str(image_path)},
            error="" if success else "validation_failed",
        )

        if not success:
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 1
        return 0

    except Exception as exc:  # pragma: no cover - runtime guard
        write_event(
            event="run_complete",
            scenario=scenario,
            run_id=run_id,
            success=False,
            duration_sec=float(time.time() - started_at),
            ttfi_sec=0.0,
            artifacts={"summary": str(summary_path), "imagination": str(image_path)},
            error=str(exc),
        )
        print(f"quickstart failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
