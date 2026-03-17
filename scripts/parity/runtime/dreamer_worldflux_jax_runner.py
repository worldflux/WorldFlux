#!/usr/bin/env python3
"""Repo-local runner for WorldFlux Dreamer JAX proof executions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import sysconfig
from pathlib import Path

RUNTIME_ROOT = Path(__file__).resolve().parent
WRAPPERS_ROOT = RUNTIME_ROOT.parent / "wrappers"
for path in (WRAPPERS_ROOT, RUNTIME_ROOT, Path(__file__).resolve().parents[3] / "src"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from common import (  # noqa: E402
    curve_auc,
    curve_final_mean,
    find_latest_file,
    load_jsonl_curve,
    run_command,
    write_metrics,
)
from dreamer_official_recipe import OFFICIAL_DREAMER_ATARI100K_RECIPE  # noqa: E402

from worldflux.parity import get_backend_adapter_registry  # noqa: E402

_ADAPTER_ID = "worldflux_dreamerv3_jax_subprocess"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--official-repo-root", type=Path, default=None)
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


def _official_env(*, repo_root: Path) -> dict[str, str]:
    purelib = Path(sysconfig.get_paths().get("purelib", "")).resolve()
    vendor_root = _resolve_official_repo_root(
        argparse.Namespace(repo_root=repo_root, official_repo_root=None)
    )
    pythonpath_parts = [
        str(purelib),
        str(vendor_root.resolve()),
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


def _resolve_official_repo_root(args: argparse.Namespace) -> Path:
    if args.official_repo_root is not None:
        return args.official_repo_root.resolve()
    return (args.repo_root.resolve() / "third_party" / "dreamerv3_official").resolve()


def _build_command(args: argparse.Namespace) -> list[str]:
    official_repo_root = _resolve_official_repo_root(args)
    python_executable = str(getattr(args, "python_executable", sys.executable))
    logdir = args.run_dir.resolve() / "dreamerv3_logdir"
    return [
        python_executable,
        str((official_repo_root / "dreamerv3" / "main.py").resolve()),
        "--logdir",
        str(logdir.resolve()),
        "--configs",
        "atari100k",
        "--task",
        str(args.task_id),
        "--seed",
        str(int(args.seed)),
        "--run.steps",
        str(int(args.steps)),
        "--jax.platform",
        "cpu" if str(args.device).lower() == "cpu" else "cuda",
        "--logger.outputs",
        "jsonl",
    ]


def main() -> int:
    args = _parse_args()
    repo_root = args.repo_root.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    official_repo_root = _resolve_official_repo_root(args)
    if not official_repo_root.exists():
        raise SystemExit(f"official Dreamer repo not found: {official_repo_root}")

    recipe = {**OFFICIAL_DREAMER_ATARI100K_RECIPE.to_metadata(), "steps": int(args.steps)}
    command = _build_command(args)
    completed = run_command(
        command,
        cwd=official_repo_root,
        timeout_sec=args.timeout_sec if args.timeout_sec > 0 else None,
        env=_official_env(repo_root=official_repo_root),
    )
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        return int(completed.returncode)

    scores_file = args.scores_file
    if scores_file is None:
        scores_file = find_latest_file(
            [run_dir, official_repo_root],
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

    eval_protocol_hash = _eval_protocol_hash(
        family="dreamerv3",
        task_id=args.task_id,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eval_window=args.eval_window,
    )
    adapter = get_backend_adapter_registry().require(_ADAPTER_ID)
    artifact_manifest = adapter.collect_artifacts(
        run_dir=run_dir,
        source_commit=None,
        eval_protocol_hash=eval_protocol_hash,
        command_argv=list(command),
        recipe=recipe,
    )

    payload = write_metrics(
        metrics_out=args.metrics_out,
        adapter="worldflux_dreamerv3_jax",
        task_id=args.task_id,
        seed=args.seed,
        device=args.device,
        points=points,
        final_return_mean=curve_final_mean(points, args.eval_window),
        auc_return=curve_auc(points),
        metadata={
            "mode": "worldflux_jax",
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
            "official_repo_root": str(official_repo_root),
            "scores_file": str(scores_file),
            "command": command,
            "command_source": "repo_local_runner",
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
            "policy_mode": "parity_candidate",
            "policy_impl": "worldflux_dreamerv3_jax_candidate",
            "framework_mode": "shared_jax_subprocess",
            "eval_protocol_hash": eval_protocol_hash,
            "implementation_source": "repo_local_worldflux_jax_runner",
        },
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
