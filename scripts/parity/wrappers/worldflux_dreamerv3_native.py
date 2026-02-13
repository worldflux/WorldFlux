#!/usr/bin/env python3
"""Parity adapter for WorldFlux-native DreamerV3 runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from common import run_command


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
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
    parser.add_argument("--python-executable", type=str, default="python3")
    parser.add_argument("--train-command", type=str, default="")
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def _format_template(template: str, values: dict[str, Any]) -> str:
    return template.format_map(values)


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    values = {
        "repo_root": str(repo_root),
        "task_id": args.task_id,
        "seed": args.seed,
        "steps": args.steps,
        "device": args.device,
        "run_dir": str(run_dir),
        "metrics_out": str(args.metrics_out),
        "python_executable": args.python_executable,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "eval_window": args.eval_window,
    }

    command: str | list[str]
    if args.train_command.strip():
        command = _format_template(args.train_command, values)
    else:
        command = [
            args.python_executable,
            "scripts/parity/wrappers/worldflux_native_online_runner.py",
            "--family",
            "dreamerv3",
            "--task-id",
            args.task_id,
            "--seed",
            str(args.seed),
            "--steps",
            str(args.steps),
            "--device",
            args.device,
            "--run-dir",
            str(run_dir),
            "--metrics-out",
            str(args.metrics_out),
            "--eval-interval",
            str(args.eval_interval),
            "--eval-episodes",
            str(args.eval_episodes),
            "--eval-window",
            str(args.eval_window),
        ]
        if args.mock:
            command.append("--mock")

    completed = run_command(
        command,
        cwd=repo_root,
        timeout_sec=args.timeout_sec if args.timeout_sec > 0 else None,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        return int(completed.returncode)

    if not args.metrics_out.exists():
        raise SystemExit(f"metrics file missing after run: {args.metrics_out}")

    payload = json.loads(args.metrics_out.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
