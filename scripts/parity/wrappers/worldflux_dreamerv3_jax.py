#!/usr/bin/env python3
"""Parity adapter for WorldFlux DreamerV3 runs on the shared JAX subprocess path."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from common import (
    curve_auc,
    curve_final_mean,
    deterministic_mock_curve,
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

from worldflux.parity import stable_recipe_hash  # noqa: E402

_ADAPTER_ID = "worldflux_dreamerv3_jax_subprocess"


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


def _runner_command(args: argparse.Namespace, *, repo_root: Path) -> list[str]:
    return [
        args.python_executable,
        str((RUNTIME_ROOT / "runtime" / "dreamer_worldflux_jax_runner.py").resolve()),
        "--repo-root",
        str(repo_root),
        "--task-id",
        str(args.task_id),
        "--seed",
        str(int(args.seed)),
        "--steps",
        str(int(args.steps)),
        "--device",
        str(args.device),
        "--run-dir",
        str(args.run_dir.resolve()),
        "--metrics-out",
        str(args.metrics_out.resolve()),
        "--eval-window",
        str(int(args.eval_window)),
        "--eval-interval",
        str(int(args.eval_interval)),
        "--eval-episodes",
        str(int(args.eval_episodes)),
        "--python-executable",
        str(args.python_executable),
    ]


def main() -> int:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    recipe = {**OFFICIAL_DREAMER_ATARI100K_RECIPE.to_metadata(), "steps": int(args.steps)}
    eval_protocol_hash = _eval_protocol_hash(
        family="dreamerv3",
        task_id=args.task_id,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        eval_window=args.eval_window,
    )

    if args.mock:
        points = deterministic_mock_curve(
            seed=args.seed,
            steps=args.steps,
            family="dreamerv3",
            system="worldflux",
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
                "mode": "mock",
                "env_backend": "gymnasium",
                "backend_kind": "jax_subprocess",
                "adapter_id": _ADAPTER_ID,
                "recipe_hash": stable_recipe_hash(recipe),
                "model_id": "dreamerv3:official_xl",
                "model_profile": "official_xl",
                "official_recipe": OFFICIAL_DREAMER_ATARI100K_RECIPE.to_metadata(),
                "effective_recipe": recipe,
                "artifact_manifest": {
                    "adapter_id": _ADAPTER_ID,
                    "backend_kind": "jax_subprocess",
                    "recipe_hash": stable_recipe_hash(recipe),
                    "checkpoint_paths": [],
                    "score_paths": [],
                    "metrics_paths": [str(args.metrics_out)],
                },
                "policy_mode": "parity_candidate",
                "policy_impl": "worldflux_dreamerv3_jax_candidate",
                "framework_mode": "shared_jax_subprocess",
                "command_source": "mock",
                "eval_protocol_hash": eval_protocol_hash,
            },
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    if args.repo_root is None:
        raise SystemExit("--repo-root is required unless --mock is set")
    repo_root = args.repo_root.resolve()
    if not repo_root.exists():
        raise SystemExit(f"repo root not found: {repo_root}")

    command = _runner_command(args, repo_root=repo_root)
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
