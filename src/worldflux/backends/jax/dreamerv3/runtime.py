# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Runtime helpers for WorldFlux-managed DreamerV3 JAX subprocess execution."""

from __future__ import annotations

import os
import sysconfig
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from worldflux.parity.backend_contract import resolve_latest_checkpoint_dir

LAUNCHER_MODULE = "worldflux.backends.jax.dreamerv3.launcher"
_OFFICIAL_RECIPE_BASE = {
    "recipe_id": "official_dreamerv3_atari100k",
    "recipe_source": "dreamerv3/configs.yaml#atari100k",
    "envs": 1,
    "train_ratio": 256.0,
    "batch_size": 16,
    "batch_length": 64,
    "report_length": 32,
    "replay_size": 5_000_000,
    "replay_chunksize": 1024,
    "log_every": 120,
    "report_every": 300,
    "save_every": 900,
    "eval_eps": 1,
    "learning_rate": 4e-5,
    "model_profile": "official_xl",
    "action_type": "discrete",
    "max_episode_steps": 27_000,
}


@dataclass(frozen=True)
class DreamerJAXRuntimeConfig:
    repo_root: Path
    official_repo_root: Path
    run_dir: Path
    task_id: str
    seed: int
    steps: int
    device: str
    python_executable: str

    @property
    def logdir(self) -> Path:
        return self.run_dir / "dreamerv3_logdir"


def resolve_official_repo_root(
    repo_root: Path,
    official_repo_root: Path | None = None,
) -> Path:
    if official_repo_root is not None:
        return official_repo_root.resolve()
    return (repo_root.resolve() / "third_party" / "dreamerv3_official").resolve()


def _jax_platform(device: str) -> str:
    return "cpu" if str(device).lower() == "cpu" else "cuda"


def build_launcher_command(config: DreamerJAXRuntimeConfig) -> list[str]:
    return [
        str(config.python_executable),
        "-m",
        LAUNCHER_MODULE,
        "--official-repo-root",
        str(config.official_repo_root.resolve()),
        "--logdir",
        str(config.logdir.resolve()),
        "--configs",
        "atari100k",
        "--task",
        str(config.task_id),
        "--seed",
        str(int(config.seed)),
        "--run.steps",
        str(int(config.steps)),
        "--jax.platform",
        _jax_platform(config.device),
        "--logger.outputs",
        "jsonl",
    ]


def official_runtime_env(*, repo_root: Path, official_repo_root: Path) -> dict[str, str]:
    purelib = Path(sysconfig.get_paths().get("purelib", "")).resolve()
    pythonpath_parts = [
        str(purelib),
        str((repo_root.resolve() / "src").resolve()),
        str(official_repo_root.resolve()),
        str(official_repo_root.resolve().parent),
    ]
    merged_parts: list[str] = []
    seen: set[str] = set()
    for part in pythonpath_parts:
        if part and part not in seen:
            merged_parts.append(part)
            seen.add(part)
    existing = os.environ.get("PYTHONPATH", "")
    for part in existing.split(os.pathsep):
        part = part.strip()
        if part and part not in seen:
            merged_parts.append(part)
            seen.add(part)
    return {"PYTHONPATH": os.pathsep.join(merged_parts)}


def required_artifact_paths(run_dir: Path) -> dict[str, Path]:
    run_root = run_dir.resolve()
    logdir = run_root / "dreamerv3_logdir"
    ckpt_root = logdir / "ckpt"
    latest_dir = resolve_latest_checkpoint_dir(ckpt_root)
    if latest_dir is None:
        latest_dir = ckpt_root
    return {
        "config_yaml": logdir / "config.yaml",
        "scores_jsonl": logdir / "scores.jsonl",
        "metrics_jsonl": logdir / "metrics.jsonl",
        "latest_pointer": ckpt_root / "latest",
        "agent_pkl": latest_dir / "agent.pkl",
        "replay_pkl": latest_dir / "replay.pkl",
        "step_pkl": latest_dir / "step.pkl",
        "done_marker": latest_dir / "done",
    }


def missing_required_artifacts(run_dir: Path) -> list[str]:
    paths = required_artifact_paths(run_dir)
    return [name for name, path in paths.items() if not path.exists()]


def validate_required_artifacts(run_dir: Path) -> dict[str, Path]:
    paths = required_artifact_paths(run_dir)
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise RuntimeError(f"missing required Dreamer artifacts: {', '.join(missing)}")
    return paths


def _eval_protocol_hash(
    *,
    family: str,
    task_id: str,
    eval_interval: int,
    eval_episodes: int,
    eval_window: int,
) -> str:
    import hashlib
    import json

    payload = {
        "family": str(family),
        "task_id": str(task_id),
        "eval_interval": int(eval_interval),
        "eval_episodes": int(eval_episodes),
        "eval_window": int(eval_window),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_train_budget(*, steps: int) -> dict[str, Any]:
    return {
        "steps": int(steps),
        "train_ratio": _OFFICIAL_RECIPE_BASE["train_ratio"],
        "batch_size": _OFFICIAL_RECIPE_BASE["batch_size"],
        "batch_length": _OFFICIAL_RECIPE_BASE["batch_length"],
        "report_length": _OFFICIAL_RECIPE_BASE["report_length"],
        "replay_size": _OFFICIAL_RECIPE_BASE["replay_size"],
        "replay_chunksize": _OFFICIAL_RECIPE_BASE["replay_chunksize"],
        "envs": _OFFICIAL_RECIPE_BASE["envs"],
        "max_episode_steps": _OFFICIAL_RECIPE_BASE["max_episode_steps"],
    }


def build_eval_protocol(
    *,
    eval_interval: int,
    eval_episodes: int,
    eval_window: int,
) -> dict[str, Any]:
    return {
        "eval_interval": int(eval_interval),
        "eval_episodes": int(eval_episodes),
        "eval_window": int(eval_window),
        "environment_backend": "gymnasium",
        "log_every": _OFFICIAL_RECIPE_BASE["log_every"],
        "report_every": _OFFICIAL_RECIPE_BASE["report_every"],
        "save_every": _OFFICIAL_RECIPE_BASE["save_every"],
    }


def build_proof_metadata(
    *,
    task_id: str,
    seed: int,
    device: str,
    steps: int,
    eval_interval: int,
    eval_episodes: int,
    eval_window: int,
    recipe_hash: str,
    backend_kind: str,
    adapter_id: str,
    artifact_manifest: dict[str, Any],
    command: list[str],
    repo_root: Path,
    scores_file: Path,
    stdout_tail: str,
    stderr_tail: str,
) -> dict[str, Any]:
    recipe = {**_OFFICIAL_RECIPE_BASE, "steps": int(steps)}
    return {
        "mode": "worldflux_jax",
        "env_backend": "gymnasium",
        "backend_kind": str(backend_kind),
        "adapter_id": str(adapter_id),
        "recipe_hash": str(recipe_hash),
        "model_id": "dreamerv3:official_xl",
        "model_profile": "official_xl",
        "official_recipe": dict(_OFFICIAL_RECIPE_BASE),
        "effective_recipe": recipe,
        "artifact_manifest": dict(artifact_manifest),
        "repo_root": str(repo_root),
        "scores_file": str(scores_file),
        "command": list(command),
        "command_source": "worldflux_launcher_module",
        "stdout_tail": str(stdout_tail),
        "stderr_tail": str(stderr_tail),
        "strict_official_semantics": True,
        "policy_mode": "parity_candidate",
        "policy_impl": "candidate_actor_stateful",
        "framework_mode": "shared_jax_subprocess",
        "train_budget": build_train_budget(steps=steps),
        "eval_protocol": build_eval_protocol(
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            eval_window=eval_window,
        ),
        "eval_protocol_hash": _eval_protocol_hash(
            family="dreamerv3",
            task_id=task_id,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
            eval_window=eval_window,
        ),
        "implementation_source": "worldflux_backends_jax_dreamerv3",
        "task_id": str(task_id),
        "seed": int(seed),
        "device": str(device),
    }


__all__ = [
    "LAUNCHER_MODULE",
    "DreamerJAXRuntimeConfig",
    "build_eval_protocol",
    "build_launcher_command",
    "build_proof_metadata",
    "build_train_budget",
    "missing_required_artifacts",
    "official_runtime_env",
    "required_artifact_paths",
    "resolve_official_repo_root",
    "validate_required_artifacts",
]
