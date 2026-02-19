#!/usr/bin/env python3
"""Parity adapter for official DreamerV3 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from common import (
    curve_auc,
    curve_final_mean,
    deterministic_mock_curve,
    find_latest_file,
    load_jsonl_curve,
    run_command,
    write_metrics,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=110_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--eval-window", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument("--scores-file", type=Path, default=None)
    parser.add_argument("--python-executable", type=str, default="python3")
    parser.add_argument("--train-command", type=str, default="")
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


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


def _format_template(template: str, values: dict[str, Any]) -> str:
    return template.format_map(values)


def _default_command(args: argparse.Namespace, *, repo_root: Path, logdir: Path) -> list[str]:
    # JAX expects explicit backend names (e.g. "cuda"), and "gpu" can route
    # through unsupported backend probes on some CUDA hosts.
    jax_platform = "cpu" if args.device.lower() == "cpu" else "cuda"
    return [
        args.python_executable,
        "dreamerv3/main.py",
        "--logdir",
        str(logdir),
        "--configs",
        "atari100k",
        "--task",
        args.task_id,
        "--seed",
        str(args.seed),
        "--run.steps",
        str(args.steps),
        "--jax.platform",
        jax_platform,
        "--logger.outputs",
        "jsonl",
    ]


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.mock:
        points = deterministic_mock_curve(
            seed=args.seed,
            steps=args.steps,
            family="dreamerv3",
            system="official",
        )
        payload = write_metrics(
            metrics_out=args.metrics_out,
            adapter="official_dreamerv3",
            task_id=args.task_id,
            seed=args.seed,
            device=args.device,
            points=points,
            final_return_mean=curve_final_mean(points, args.eval_window),
            auc_return=curve_auc(points),
            metadata={
                "mode": "mock",
                "policy_mode": "official_reference",
                "policy_impl": "official_dreamerv3_reference",
                "eval_protocol_hash": _eval_protocol_hash(
                    family="dreamerv3",
                    task_id=args.task_id,
                    eval_interval=args.eval_interval,
                    eval_episodes=args.eval_episodes,
                    eval_window=args.eval_window,
                ),
            },
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.repo_root is None:
        raise SystemExit("--repo-root is required unless --mock is set")
    repo_root = args.repo_root.resolve()
    if not repo_root.exists():
        raise SystemExit(f"repo root not found: {repo_root}")

    logdir = run_dir / "dreamerv3_logdir"
    template_values = {
        "repo_root": str(repo_root),
        "logdir": str(logdir),
        "task_id": args.task_id,
        "seed": args.seed,
        "steps": args.steps,
        "device": args.device,
        "run_dir": str(run_dir),
        "metrics_out": str(args.metrics_out),
        "python_executable": args.python_executable,
    }

    command: str | list[str]
    if args.train_command.strip():
        command = _format_template(args.train_command, template_values)
    else:
        command = _default_command(args, repo_root=repo_root, logdir=logdir)

    completed = run_command(
        command,
        cwd=repo_root,
        timeout_sec=args.timeout_sec if args.timeout_sec > 0 else None,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        return int(completed.returncode)

    scores_file = args.scores_file
    if scores_file is None:
        scores_file = find_latest_file(
            [run_dir, repo_root],
            ["scores.jsonl", "metrics.jsonl"],
        )
    if scores_file is None or not scores_file.exists():
        raise SystemExit("could not locate DreamerV3 score logs (scores.jsonl/metrics.jsonl)")

    points = load_jsonl_curve(
        scores_file,
        value_keys=["episode/score", "score", "episode_reward", "episode/return", "return"],
    )
    if not points:
        raise SystemExit(f"no score curve points found in {scores_file}")

    payload = write_metrics(
        metrics_out=args.metrics_out,
        adapter="official_dreamerv3",
        task_id=args.task_id,
        seed=args.seed,
        device=args.device,
        points=points,
        final_return_mean=curve_final_mean(points, args.eval_window),
        auc_return=curve_auc(points),
        metadata={
            "mode": "official",
            "repo_root": str(repo_root),
            "scores_file": str(scores_file),
            "command": command,
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
            "policy_mode": "official_reference",
            "policy_impl": "official_dreamerv3_reference",
            "eval_protocol_hash": _eval_protocol_hash(
                family="dreamerv3",
                task_id=args.task_id,
                eval_interval=args.eval_interval,
                eval_episodes=args.eval_episodes,
                eval_window=args.eval_window,
            ),
        },
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
