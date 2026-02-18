#!/usr/bin/env python3
"""WorldFlux-native online parity runner using real environment interactions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from common import CurvePoint, curve_auc, curve_final_mean, deterministic_mock_curve, write_metrics

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from runtime.atari_env import AtariEnvError  # noqa: E402
from runtime.dmcontrol_env import DMControlEnvError  # noqa: E402
from runtime.dreamer_native_agent import DreamerNativeRunConfig, run_dreamer_native  # noqa: E402
from runtime.tdmpc2_native_agent import TDMPC2NativeRunConfig, run_tdmpc2_native  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=str, choices=["dreamerv3", "tdmpc2"], required=True)
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--eval-window", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--env-backend", type=str, default="auto")
    parser.add_argument(
        "--policy-mode",
        type=str,
        default="diagnostic_random",
        choices=["diagnostic_random", "parity_candidate"],
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--mock", action="store_true")

    parser.add_argument("--buffer-capacity", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--train-steps-per-eval", type=int, default=0)
    parser.add_argument("--sequence-length", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    return parser.parse_args()


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _resolve_backend(family: str, backend: str) -> str:
    normalized = backend.strip().lower()
    if normalized != "auto":
        return normalized
    return "gymnasium" if family == "dreamerv3" else "dmcontrol"


def _override_int(value: int, default: int) -> int:
    return int(value) if int(value) > 0 else int(default)


def _eval_protocol_hash(
    *,
    family: str,
    task_id: str,
    eval_interval: int,
    eval_episodes: int,
    eval_window: int,
) -> str:
    payload = {
        "family": str(family),
        "task_id": str(task_id),
        "eval_interval": int(eval_interval),
        "eval_episodes": int(eval_episodes),
        "eval_window": int(eval_window),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = _resolve_device(args.device)
    resolved_backend = _resolve_backend(args.family, args.env_backend)
    protocol_hash = _eval_protocol_hash(
        family=args.family,
        task_id=args.task_id,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eval_window=args.eval_window,
    )

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
            device=resolved_device,
            points=points,
            final_return_mean=curve_final_mean(points, args.eval_window),
            auc_return=curve_auc(points),
            metadata={
                "mode": "mock",
                "env_backend": resolved_backend,
                "resolved_device": resolved_device,
                "policy_mode": args.policy_mode,
                "policy_impl": "deterministic_mock_curve",
                "eval_protocol_hash": protocol_hash,
            },
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    try:
        if args.family == "dreamerv3":
            curve_raw, metadata = run_dreamer_native(
                DreamerNativeRunConfig(
                    task_id=args.task_id,
                    seed=args.seed,
                    steps=args.steps,
                    eval_interval=args.eval_interval,
                    eval_episodes=args.eval_episodes,
                    eval_window=args.eval_window,
                    env_backend=resolved_backend,
                    device=resolved_device,
                    run_dir=run_dir,
                    buffer_capacity=_override_int(args.buffer_capacity, 200_000),
                    warmup_steps=_override_int(args.warmup_steps, 1_024),
                    train_steps_per_eval=_override_int(args.train_steps_per_eval, 64),
                    sequence_length=_override_int(args.sequence_length, 32),
                    batch_size=_override_int(args.batch_size, 16),
                    max_episode_steps=_override_int(args.max_episode_steps, 27_000),
                    policy_mode=args.policy_mode,
                )
            )
        else:
            curve_raw, metadata = run_tdmpc2_native(
                TDMPC2NativeRunConfig(
                    task_id=args.task_id,
                    seed=args.seed,
                    steps=args.steps,
                    eval_interval=args.eval_interval,
                    eval_episodes=args.eval_episodes,
                    eval_window=args.eval_window,
                    env_backend=resolved_backend,
                    device=resolved_device,
                    run_dir=run_dir,
                    buffer_capacity=_override_int(args.buffer_capacity, 300_000),
                    warmup_steps=_override_int(args.warmup_steps, 2_048),
                    train_steps_per_eval=_override_int(args.train_steps_per_eval, 96),
                    sequence_length=_override_int(args.sequence_length, 10),
                    batch_size=_override_int(args.batch_size, 64),
                    max_episode_steps=_override_int(args.max_episode_steps, 1_000),
                    policy_mode=args.policy_mode,
                )
            )
    except (AtariEnvError, DMControlEnvError) as exc:
        raise SystemExit(str(exc)) from exc

    points = [CurvePoint(step=float(step), value=float(value)) for step, value in curve_raw]

    payload = write_metrics(
        metrics_out=args.metrics_out,
        adapter=f"worldflux_{args.family}_native",
        task_id=args.task_id,
        seed=args.seed,
        device=resolved_device,
        points=points,
        final_return_mean=curve_final_mean(points, args.eval_window),
        auc_return=curve_auc(points),
        metadata={
            **metadata,
            "env_backend": resolved_backend,
            "resolved_device": resolved_device,
            "policy_mode": args.policy_mode,
            "eval_protocol_hash": protocol_hash,
        },
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
