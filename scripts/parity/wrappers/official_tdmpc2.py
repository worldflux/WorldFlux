#!/usr/bin/env python3
"""Parity adapter for official TD-MPC2 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from common import (
    curve_auc,
    curve_final_mean,
    deterministic_mock_curve,
    find_latest_file,
    load_csv_curve,
    run_command,
    write_metrics,
)

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from worldflux.parity import get_backend_adapter_registry, stable_recipe_hash  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--task-id", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--eval-window", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument("--eval-csv", type=Path, default=None)
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--train-command", type=str, default="")
    parser.add_argument("--model-size", type=int, default=5)
    parser.add_argument("--alignment-report", type=Path, default=None)
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def _format_template(template: str, values: dict[str, Any]) -> str:
    return template.format_map(values)


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


def _default_command(args: argparse.Namespace, *, exp_name: str) -> list[str]:
    adapter = get_backend_adapter_registry().require("official_tdmpc2_torch_subprocess")
    spec = adapter.prepare_run(
        recipe={"steps": int(args.steps), "model_size": int(args.model_size)},
        env_spec={
            "task_id": args.task_id,
            "eval_protocol": {
                "eval_episodes": int(args.eval_episodes),
                "eval_interval": int(args.eval_freq),
            },
        },
        seed=int(args.seed),
        run_dir=Path(exp_name),
        repo_root=Path("."),
        python_executable=args.python_executable,
        device=args.device,
    )
    return list(spec.command)


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.mock:
        recipe = {"steps": int(args.steps), "model_size": int(args.model_size)}
        points = deterministic_mock_curve(
            seed=args.seed,
            steps=args.steps,
            family="tdmpc2",
            system="official",
        )
        alignment_report_path, alignment_status = _resolve_alignment_report(args.alignment_report)
        payload = write_metrics(
            metrics_out=args.metrics_out,
            adapter="official_tdmpc2",
            task_id=args.task_id,
            seed=args.seed,
            device=args.device,
            points=points,
            final_return_mean=curve_final_mean(points, args.eval_window),
            auc_return=curve_auc(points),
            metadata={
                "mode": "mock",
                "env_backend": "dmcontrol",
                "backend_kind": "torch_subprocess",
                "adapter_id": "official_tdmpc2_torch_subprocess",
                "recipe_hash": stable_recipe_hash(recipe),
                "model_id": f"tdmpc2:{args.model_size}m",
                "model_profile": f"{args.model_size}m",
                "canonical_compare_profile": "proof_5m",
                "official_model_size": int(args.model_size),
                "artifact_manifest": {
                    "adapter_id": "official_tdmpc2_torch_subprocess",
                    "backend_kind": "torch_subprocess",
                    "recipe_hash": stable_recipe_hash(recipe),
                    "checkpoint_paths": [],
                    "score_paths": [],
                    "metrics_paths": [str(args.metrics_out)],
                },
                "policy_mode": "official_reference",
                "policy_impl": "official_tdmpc2_reference",
                "train_budget": {
                    "steps": int(args.steps),
                    "model_size": int(args.model_size),
                },
                "eval_protocol": {
                    "eval_interval": int(args.eval_freq),
                    "eval_episodes": int(args.eval_episodes),
                    "eval_window": int(args.eval_window),
                    "environment_backend": "dmcontrol",
                },
                "eval_protocol_hash": _eval_protocol_hash(
                    family="tdmpc2",
                    task_id=args.task_id,
                    eval_interval=args.eval_freq,
                    eval_episodes=args.eval_episodes,
                    eval_window=args.eval_window,
                ),
                "alignment_report_path": alignment_report_path,
                "alignment_status": alignment_status,
            },
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.repo_root is None:
        raise SystemExit("--repo-root is required unless --mock is set")
    repo_root = args.repo_root.resolve()
    if not repo_root.exists():
        raise SystemExit(f"repo root not found: {repo_root}")

    exp_name = run_dir.name
    recipe = {"steps": int(args.steps), "model_size": int(args.model_size)}
    adapter = get_backend_adapter_registry().require("official_tdmpc2_torch_subprocess")
    template_values = {
        "repo_root": str(repo_root),
        "task_id": args.task_id,
        "seed": args.seed,
        "steps": args.steps,
        "device": args.device,
        "run_dir": str(run_dir),
        "metrics_out": str(args.metrics_out),
        "python_executable": args.python_executable,
        "exp_name": exp_name,
        "eval_episodes": args.eval_episodes,
        "eval_freq": args.eval_freq,
        "model_size": args.model_size,
    }

    command: str | list[str]
    if args.train_command.strip():
        command = _format_template(args.train_command, template_values)
    else:
        command = _default_command(args, exp_name=exp_name)

    completed = run_command(
        command,
        cwd=repo_root,
        timeout_sec=args.timeout_sec if args.timeout_sec > 0 else None,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        return int(completed.returncode)

    eval_csv = args.eval_csv
    if eval_csv is None:
        eval_csv = find_latest_file(
            [run_dir, repo_root],
            ["eval.csv"],
        )
    if eval_csv is None or not eval_csv.exists():
        raise SystemExit("could not locate TD-MPC2 eval.csv")

    points = load_csv_curve(
        eval_csv,
        value_keys=["episode_reward", "reward", "episode_return", "return"],
    )
    if not points:
        raise SystemExit(f"no evaluation points found in {eval_csv}")
    artifact_manifest = adapter.collect_artifacts(
        run_dir=run_dir,
        source_commit=None,
        eval_protocol_hash=_eval_protocol_hash(
            family="tdmpc2",
            task_id=args.task_id,
            eval_interval=args.eval_freq,
            eval_episodes=args.eval_episodes,
            eval_window=args.eval_window,
        ),
        command_argv=list(command) if isinstance(command, list) else [],
        recipe=recipe,
    )

    alignment_report_path, alignment_status = _resolve_alignment_report(args.alignment_report)
    payload = write_metrics(
        metrics_out=args.metrics_out,
        adapter="official_tdmpc2",
        task_id=args.task_id,
        seed=args.seed,
        device=args.device,
        points=points,
        final_return_mean=curve_final_mean(points, args.eval_window),
        auc_return=curve_auc(points),
        metadata={
            "mode": "official",
            "env_backend": "dmcontrol",
            "backend_kind": artifact_manifest.backend_kind,
            "adapter_id": artifact_manifest.adapter_id,
            "recipe_hash": artifact_manifest.recipe_hash,
            "model_id": f"tdmpc2:{args.model_size}m",
            "model_profile": f"{args.model_size}m",
            "canonical_compare_profile": "proof_5m",
            "official_model_size": int(args.model_size),
            "artifact_manifest": artifact_manifest.to_dict(),
            "repo_root": str(repo_root),
            "eval_csv": str(eval_csv),
            "command": command,
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
            "policy_mode": "official_reference",
            "policy_impl": "official_tdmpc2_reference",
            "train_budget": {
                "steps": int(args.steps),
                "model_size": int(args.model_size),
            },
            "eval_protocol": {
                "eval_interval": int(args.eval_freq),
                "eval_episodes": int(args.eval_episodes),
                "eval_window": int(args.eval_window),
                "environment_backend": "dmcontrol",
            },
            "eval_protocol_hash": _eval_protocol_hash(
                family="tdmpc2",
                task_id=args.task_id,
                eval_interval=args.eval_freq,
                eval_episodes=args.eval_episodes,
                eval_window=args.eval_window,
            ),
            "alignment_report_path": alignment_report_path,
            "alignment_status": alignment_status,
        },
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
