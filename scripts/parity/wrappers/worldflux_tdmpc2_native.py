#!/usr/bin/env python3
"""Parity adapter for WorldFlux-native TD-MPC2 runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from common import run_command

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--eval-window", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument(
        "--policy-mode",
        type=str,
        default="parity_candidate",
        choices=["diagnostic_random", "parity_candidate"],
    )
    parser.add_argument(
        "--tdmpc2-model-profile",
        type=str,
        default="5m",
        choices=["ci", "5m", "proof_5m", "5m_legacy", "19m", "48m", "317m"],
    )
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--train-command", type=str, default="")
    parser.add_argument("--alignment-report", type=Path, default=None)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def _format_template(template: str, values: dict[str, Any]) -> str:
    return template.format_map(values)


def _resolve_alignment_report(path: Path | None) -> tuple[str, str]:
    candidate = path
    if candidate is None:
        raw = str(os.environ.get("WORLDFLUX_TDMPC2_ALIGNMENT_REPORT", "")).strip()
        candidate = Path(raw).expanduser() if raw else None
    if candidate is None or not candidate.exists():
        return "", ""
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return "", ""
    return str(candidate.resolve()), str(payload.get("status", "")).strip().lower()


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
        "policy_mode": args.policy_mode,
        "tdmpc2_model_profile": args.tdmpc2_model_profile,
    }

    command: str | list[str]
    if args.train_command.strip():
        command = _format_template(args.train_command, values)
    else:
        command = [
            args.python_executable,
            "scripts/parity/wrappers/worldflux_native_online_runner.py",
            "--family",
            "tdmpc2",
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
            "--policy-mode",
            str(args.policy_mode),
            "--tdmpc2-model-profile",
            str(args.tdmpc2_model_profile),
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
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        alignment_report_path, alignment_status = _resolve_alignment_report(args.alignment_report)
        if alignment_report_path:
            metadata["alignment_report_path"] = alignment_report_path
            metadata["alignment_status"] = alignment_status
            args.metrics_out.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
