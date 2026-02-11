#!/usr/bin/env python3
"""Compare DreamerV3 and TD-MPC2 with the same Trainer/data contract."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from _shared.viz import write_reward_heatmap_ppm

from worldflux import create_world_model
from worldflux.telemetry.wasr import make_run_id, write_event
from worldflux.training import Trainer, TrainingConfig
from worldflux.training.data import create_random_buffer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified training comparison demo")
    parser.add_argument("--quick", action="store_true", help="CI-safe runtime settings")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/comparison")
    return parser


def _run_family(
    family: str,
    *,
    buffer,
    output_dir: Path,
    total_steps: int,
    seed: int,
) -> dict[str, Any]:
    obs_shape = buffer.obs_shape
    action_dim = buffer.action_dim

    kwargs: dict[str, Any] = {}
    if family == "dreamerv3":
        kwargs["encoder_type"] = "mlp"
        kwargs["decoder_type"] = "mlp"

    model = create_world_model(f"{family}:ci", obs_shape=obs_shape, action_dim=action_dim, **kwargs)
    trainer = Trainer(
        model,
        TrainingConfig(
            total_steps=total_steps,
            batch_size=16,
            sequence_length=10,
            output_dir=str(output_dir / family),
            device="cpu",
            seed=seed,
            log_interval=max(1, total_steps // 4),
            save_interval=total_steps + 1,
        ),
    )

    initial_loss = float(trainer.evaluate(buffer, num_batches=2)["loss"])
    trainer.train(buffer)
    final_loss = float(trainer.evaluate(buffer, num_batches=2)["loss"])

    sample = buffer.sample(batch_size=1, seq_len=11, device="cpu")
    state = model.encode(sample.obs[:, 0])
    action_seq = sample.actions[:, :10].permute(1, 0, 2)
    rollout = model.rollout(state, action_seq)
    rewards = rollout.rewards.detach().cpu().numpy().reshape(-1)

    image_path = output_dir / f"{family.split(':')[0]}.ppm"
    write_reward_heatmap_ppm(rewards, image_path)

    return {
        "family": family,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "finite": bool(np.isfinite(initial_loss) and np.isfinite(final_loss)),
        "loss_improved": bool(final_loss < initial_loss),
        "rollout_horizon": int(rollout.actions.shape[0]),
        "artifacts": {"imagination": str(image_path)},
    }


def main() -> int:
    args = _parser().parse_args()
    run_id = make_run_id()
    started = time.time()
    scenario = "comparison_unified_training"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"

    try:
        total_steps = 8 if args.quick else 48
        buffer = create_random_buffer(
            capacity=2000 if args.quick else 8000,
            obs_shape=(8,),
            action_dim=2,
            num_episodes=40 if args.quick else 100,
            episode_length=40,
            seed=args.seed,
        )

        dreamer_result = _run_family(
            "dreamerv3",
            buffer=buffer,
            output_dir=output_dir,
            total_steps=total_steps,
            seed=args.seed,
        )
        tdmpc2_result = _run_family(
            "tdmpc2",
            buffer=buffer,
            output_dir=output_dir,
            total_steps=total_steps,
            seed=args.seed,
        )

        success = bool(
            dreamer_result["finite"]
            and tdmpc2_result["finite"]
            and dreamer_result["rollout_horizon"] > 0
            and tdmpc2_result["rollout_horizon"] > 0
        )
        summary = {
            "scenario": scenario,
            "run_id": run_id,
            "quick": bool(args.quick),
            "seed": int(args.seed),
            "success": success,
            "models": {
                "dreamerv3": dreamer_result,
                "tdmpc2": tdmpc2_result,
            },
            "artifacts": {
                "summary": str(summary_path),
                "dreamer": dreamer_result["artifacts"]["imagination"],
                "tdmpc2": tdmpc2_result["artifacts"]["imagination"],
            },
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        write_event(
            event="run_complete",
            scenario=scenario,
            run_id=run_id,
            success=success,
            duration_sec=float(time.time() - started),
            ttfi_sec=float(time.time() - started),
            artifacts={
                "summary": str(summary_path),
                "dreamer": dreamer_result["artifacts"]["imagination"],
                "tdmpc2": tdmpc2_result["artifacts"]["imagination"],
            },
            error="" if success else "comparison_validation_failed",
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
            duration_sec=float(time.time() - started),
            ttfi_sec=0.0,
            artifacts={"summary": str(summary_path)},
            error=str(exc),
        )
        print(f"comparison failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
