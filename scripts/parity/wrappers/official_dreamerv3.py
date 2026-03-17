#!/usr/bin/env python3
"""Parity adapter for official DreamerV3 runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import sysconfig
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

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from runtime.dreamer_official_recipe import OFFICIAL_DREAMER_ATARI100K_RECIPE  # noqa: E402

from worldflux.parity import get_backend_adapter_registry, stable_recipe_hash  # noqa: E402


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
    parser.add_argument(
        "--eval-episodes", type=int, default=OFFICIAL_DREAMER_ATARI100K_RECIPE.eval_eps
    )
    parser.add_argument("--timeout-sec", type=int, default=0)
    parser.add_argument("--scores-file", type=Path, default=None)
    parser.add_argument("--python-executable", type=str, default=sys.executable)
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
    adapter = get_backend_adapter_registry().require("official_dreamerv3_jax_subprocess")
    spec = adapter.prepare_run(
        recipe={**OFFICIAL_DREAMER_ATARI100K_RECIPE.to_metadata(), "steps": int(args.steps)},
        env_spec={"task_id": args.task_id},
        seed=int(args.seed),
        run_dir=logdir.parent,
        repo_root=repo_root,
        python_executable=args.python_executable,
        device=args.device,
    )
    return list(spec.command)


def _official_env(*, repo_root: Path) -> dict[str, str]:
    purelib = Path(sysconfig.get_paths().get("purelib", "")).resolve()
    pythonpath_parts = [
        str(purelib),
        str(repo_root.resolve()),
        str((repo_root / "embodied").resolve()),
    ]
    env_parts = []
    seen: set[str] = set()
    for part in pythonpath_parts:
        if part and part not in seen:
            env_parts.append(part)
            seen.add(part)
    if os.environ.get("PYTHONPATH"):
        for part in os.environ["PYTHONPATH"].split(os.pathsep):
            part = part.strip()
            if part and part not in seen:
                env_parts.append(part)
                seen.add(part)
    return {"PYTHONPATH": os.pathsep.join(env_parts)}


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.mock:
        recipe = {**OFFICIAL_DREAMER_ATARI100K_RECIPE.to_metadata(), "steps": int(args.steps)}
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
                "env_backend": "gymnasium",
                "backend_kind": "jax_subprocess",
                "adapter_id": "official_dreamerv3_jax_subprocess",
                "recipe_hash": stable_recipe_hash(recipe),
                "model_id": "dreamerv3:official_xl",
                "model_profile": "official_xl",
                "official_recipe": OFFICIAL_DREAMER_ATARI100K_RECIPE.to_metadata(),
                "effective_recipe": recipe,
                "artifact_manifest": {
                    "adapter_id": "official_dreamerv3_jax_subprocess",
                    "backend_kind": "jax_subprocess",
                    "recipe_hash": stable_recipe_hash(recipe),
                    "checkpoint_paths": [],
                    "score_paths": [],
                    "metrics_paths": [str(args.metrics_out)],
                },
                "policy_mode": "official_reference",
                "policy_impl": "official_dreamerv3_reference",
                "train_budget": {
                    "steps": int(args.steps),
                    "train_ratio": OFFICIAL_DREAMER_ATARI100K_RECIPE.train_ratio,
                    "batch_size": OFFICIAL_DREAMER_ATARI100K_RECIPE.batch_size,
                    "batch_length": OFFICIAL_DREAMER_ATARI100K_RECIPE.batch_length,
                    "report_length": OFFICIAL_DREAMER_ATARI100K_RECIPE.report_length,
                    "replay_size": OFFICIAL_DREAMER_ATARI100K_RECIPE.replay_size,
                    "replay_chunksize": OFFICIAL_DREAMER_ATARI100K_RECIPE.replay_chunksize,
                    "envs": OFFICIAL_DREAMER_ATARI100K_RECIPE.envs,
                    "max_episode_steps": OFFICIAL_DREAMER_ATARI100K_RECIPE.max_episode_steps,
                },
                "eval_protocol": {
                    "eval_interval": int(args.eval_interval),
                    "eval_episodes": int(args.eval_episodes),
                    "eval_window": int(args.eval_window),
                    "environment_backend": "gymnasium",
                    "log_every": OFFICIAL_DREAMER_ATARI100K_RECIPE.log_every,
                    "report_every": OFFICIAL_DREAMER_ATARI100K_RECIPE.report_every,
                    "save_every": OFFICIAL_DREAMER_ATARI100K_RECIPE.save_every,
                },
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

    recipe = {**OFFICIAL_DREAMER_ATARI100K_RECIPE.to_metadata(), "steps": int(args.steps)}
    adapter = get_backend_adapter_registry().require("official_dreamerv3_jax_subprocess")
    command: str | list[str]
    if args.train_command.strip():
        command = _format_template(args.train_command, template_values)
    else:
        command = _default_command(args, repo_root=repo_root, logdir=logdir)

    stdout_log = run_dir / "train_stdout.log"
    stderr_log = run_dir / "train_stderr.log"
    completed = run_command(
        command,
        cwd=repo_root,
        timeout_sec=args.timeout_sec if args.timeout_sec > 0 else None,
        env=_official_env(repo_root=repo_root),
        stdout_path=stdout_log,
        stderr_path=stderr_log,
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
    artifact_manifest = adapter.collect_artifacts(
        run_dir=run_dir,
        source_commit=None,
        eval_protocol_hash=_eval_protocol_hash(
            family="dreamerv3",
            task_id=args.task_id,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            eval_window=args.eval_window,
        ),
        command_argv=list(command) if isinstance(command, list) else [],
        recipe=recipe,
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
            "mode": "official",
            "env_backend": "gymnasium",
            "backend_kind": artifact_manifest.backend_kind,
            "adapter_id": artifact_manifest.adapter_id,
            "recipe_hash": artifact_manifest.recipe_hash,
            "model_id": "dreamerv3:official_xl",
            "model_profile": "official_xl",
            "official_recipe": OFFICIAL_DREAMER_ATARI100K_RECIPE.to_metadata(),
            "effective_recipe": recipe,
            "artifact_manifest": artifact_manifest.to_dict(),
            "repo_root": str(repo_root),
            "scores_file": str(scores_file),
            "command": command,
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
            "policy_mode": "official_reference",
            "policy_impl": "official_dreamerv3_reference",
            "train_budget": {
                "steps": int(args.steps),
                "train_ratio": OFFICIAL_DREAMER_ATARI100K_RECIPE.train_ratio,
                "batch_size": OFFICIAL_DREAMER_ATARI100K_RECIPE.batch_size,
                "batch_length": OFFICIAL_DREAMER_ATARI100K_RECIPE.batch_length,
                "report_length": OFFICIAL_DREAMER_ATARI100K_RECIPE.report_length,
                "replay_size": OFFICIAL_DREAMER_ATARI100K_RECIPE.replay_size,
                "replay_chunksize": OFFICIAL_DREAMER_ATARI100K_RECIPE.replay_chunksize,
                "envs": OFFICIAL_DREAMER_ATARI100K_RECIPE.envs,
                "max_episode_steps": OFFICIAL_DREAMER_ATARI100K_RECIPE.max_episode_steps,
            },
            "eval_protocol": {
                "eval_interval": int(args.eval_interval),
                "eval_episodes": int(args.eval_episodes),
                "eval_window": int(args.eval_window),
                "environment_backend": "gymnasium",
                "log_every": OFFICIAL_DREAMER_ATARI100K_RECIPE.log_every,
                "report_every": OFFICIAL_DREAMER_ATARI100K_RECIPE.report_every,
                "save_every": OFFICIAL_DREAMER_ATARI100K_RECIPE.save_every,
            },
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
