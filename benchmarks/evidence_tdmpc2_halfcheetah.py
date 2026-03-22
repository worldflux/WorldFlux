#!/usr/bin/env python3
"""Evidence-oriented TD-MPC2 HalfCheetah benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from common import (  # noqa: E402
    add_common_cli,
    build_run_context,
    emit_failure,
    emit_success,
    resolve_mode,
    write_summary,
)

from worldflux import create_world_model  # noqa: E402
from worldflux.evals.env_policy import collect_env_policy_rollout  # noqa: E402
from worldflux.training import Trainer, TrainingConfig  # noqa: E402
from worldflux.training.mujoco_collection import (  # noqa: E402
    collect_mujoco_dataset,
    load_buffer_from_dataset_manifest,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TD-MPC2 HalfCheetah evidence benchmark")
    add_common_cli(parser, default_output_dir="outputs/benchmarks/tdmpc2-halfcheetah-evidence")
    parser.add_argument("--env", type=str, default="HalfCheetah-v5")
    parser.add_argument("--dataset-manifest", type=str, default="")
    parser.add_argument(
        "--collector-policy",
        type=str,
        default="policy_checkpoint",
        choices=("policy_checkpoint", "noisy", "random"),
    )
    parser.add_argument("--policy-checkpoint", type=str, default="")
    parser.add_argument("--action-noise", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def _milestones_for_mode(mode: str) -> list[int]:
    return [0, 4, 8] if mode == "quick" else [0, 20, 40, 60]


def _prepare_buffer_and_manifest(
    args: argparse.Namespace,
) -> tuple[object, Path, dict[str, object]]:
    if str(args.dataset_manifest).strip():
        manifest_path = Path(args.dataset_manifest).expanduser()
        buffer, manifest = load_buffer_from_dataset_manifest(manifest_path)
        return buffer, manifest_path, manifest

    dataset_root = Path(args.output_dir) / "dataset"
    buffer, manifest_path, manifest = collect_mujoco_dataset(
        env_name=str(args.env),
        output_dir=dataset_root,
        num_episodes=8 if resolve_mode(args) == "quick" else 32,
        max_steps_per_episode=64 if resolve_mode(args) == "quick" else 256,
        collector_policy=str(args.collector_policy),
        policy_checkpoint=str(args.policy_checkpoint).strip() or None,
        action_noise=float(args.action_noise),
        seed=int(args.seed),
        device=str(args.device),
    )
    return buffer, manifest_path, manifest


def _write_returns_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_learning_curve(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step", "loss", "mean_return", "std_return", "episodes"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_checkpoint_index(path: Path, checkpoints: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps({"checkpoints": checkpoints}, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_report(
    path: Path,
    *,
    env_name: str,
    manifest_path: Path,
    curve_rows: list[dict[str, object]],
    provenance: dict[str, object],
) -> None:
    latest = curve_rows[-1]
    lines = [
        "# TD-MPC2 HalfCheetah Evidence Report",
        "",
        f"- env: `{env_name}`",
        f"- dataset_manifest: `{manifest_path.resolve()}`",
        f"- final_step: `{latest['step']}`",
        f"- final_loss: `{latest['loss']}`",
        f"- final_mean_return: `{latest['mean_return']}`",
        f"- final_std_return: `{latest['std_return']}`",
        f"- eval_mode: `{provenance.get('eval_mode', 'env_policy')}`",
        f"- policy_impl: `{provenance.get('policy_impl', 'cem_planner_eval')}`",
        "",
        "This benchmark is an evidence lane artifact bundle, not a public SOTA claim.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    mode = resolve_mode(args)
    context = build_run_context(scenario="tdmpc2_halfcheetah_evidence", mode=mode)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    returns_path = output_dir / "returns.jsonl"
    curve_path = output_dir / "learning_curve.csv"
    checkpoint_index_path = output_dir / "checkpoint_index.json"
    report_path = output_dir / "report.md"

    try:
        buffer, manifest_path, manifest = _prepare_buffer_and_manifest(args)
        model = create_world_model(
            "tdmpc2:5m",
            obs_shape=buffer.obs_shape,
            action_dim=buffer.action_dim,
            device=str(args.device),
        )
        trainer = Trainer(
            model,
            TrainingConfig(
                total_steps=max(_milestones_for_mode(mode)),
                batch_size=8 if mode == "quick" else 32,
                sequence_length=5,
                output_dir=str(output_dir),
                device=str(args.device),
                log_interval=1,
                save_interval=max(_milestones_for_mode(mode)) + 1,
                auto_quality_check=False,
            ),
            callbacks=[],
        )
        setattr(buffer, "data_provenance", getattr(buffer, "data_provenance", {}))

        curve_rows: list[dict[str, object]] = []
        return_rows: list[dict[str, object]] = []
        checkpoints: list[dict[str, object]] = []
        best_mean_return = -float("inf")
        best_checkpoint_path = output_dir / "checkpoint_best.pt"
        latest_rollout_provenance: dict[str, object] = {
            "policy_impl": "cem_planner_eval",
            "eval_mode": "env_policy",
            "seed_schedule": [],
        }

        eval_episodes = 3 if mode == "quick" else 5
        for step_target in _milestones_for_mode(mode):
            if step_target > trainer.state.global_step:
                trainer.train(buffer, num_steps=step_target)
            loss = float(trainer.evaluate(buffer, num_batches=2)["loss"])
            rollout = collect_env_policy_rollout(
                model,
                env_id=str(args.env),
                family="tdmpc2",
                episodes=eval_episodes,
                seed=int(args.seed) + step_target,
                device=str(args.device),
            )
            episode_returns = rollout.episode_returns
            latest_rollout_provenance = dict(rollout.provenance)
            mean_return = float(np.mean(episode_returns))
            std_return = float(np.std(episode_returns))
            curve_rows.append(
                {
                    "step": step_target,
                    "loss": loss,
                    "mean_return": mean_return,
                    "std_return": std_return,
                    "episodes": len(episode_returns),
                }
            )
            for index, value in enumerate(episode_returns):
                return_rows.append({"step": step_target, "episode": index, "return": value})

            checkpoint_path = output_dir / f"checkpoint_step_{step_target}.pt"
            trainer.save_checkpoint(str(checkpoint_path))
            checkpoints.append({"step": step_target, "path": str(checkpoint_path.resolve())})
            if mean_return >= best_mean_return:
                best_mean_return = mean_return
                shutil.copy2(checkpoint_path, best_checkpoint_path)
                checkpoints.append(
                    {
                        "step": step_target,
                        "path": str(best_checkpoint_path.resolve()),
                        "tag": "best",
                    }
                )
                latest_rollout_provenance["checkpoint_path"] = str(best_checkpoint_path.resolve())

        _write_returns_jsonl(returns_path, return_rows)
        _write_learning_curve(curve_path, curve_rows)
        _write_checkpoint_index(checkpoint_index_path, checkpoints)
        _write_report(
            report_path,
            env_name=str(args.env),
            manifest_path=manifest_path,
            curve_rows=curve_rows,
            provenance=latest_rollout_provenance,
        )

        summary = {
            "benchmark": "tdmpc2-halfcheetah-evidence",
            "mode": mode,
            "model": "tdmpc2:5m",
            "env": args.env,
            "eval_mode": latest_rollout_provenance.get("eval_mode", "env_policy"),
            "policy_impl": latest_rollout_provenance.get("policy_impl", "cem_planner_eval"),
            "seed_schedule": latest_rollout_provenance.get("seed_schedule", []),
            "checkpoint_path": latest_rollout_provenance.get(
                "checkpoint_path", str(best_checkpoint_path.resolve())
            ),
            "dataset_manifest": str(manifest_path.resolve()),
            "collector_policy": manifest.get("collector_policy"),
            "final_step": curve_rows[-1]["step"],
            "final_loss": curve_rows[-1]["loss"],
            "final_mean_return": curve_rows[-1]["mean_return"],
            "final_std_return": curve_rows[-1]["std_return"],
            "success": True,
            "artifacts": {
                "summary": str(summary_path.resolve()),
                "dataset_manifest": str(manifest_path.resolve()),
                "returns": str(returns_path.resolve()),
                "learning_curve": str(curve_path.resolve()),
                "checkpoint_index": str(checkpoint_index_path.resolve()),
                "report": str(report_path.resolve()),
            },
        }
        write_summary(summary_path, summary)
        emit_success(
            context,
            ttfi_sec=float(curve_rows[0]["step"]),
            artifacts={"summary": str(summary_path.resolve())},
        )
        return 0
    except Exception as exc:  # pragma: no cover - runtime guard
        emit_failure(context, error=str(exc), artifacts={"summary": str(summary_path.resolve())})
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
