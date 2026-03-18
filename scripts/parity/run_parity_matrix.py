#!/usr/bin/env python3
# ruff: noqa: E402
"""Run official-vs-WorldFlux parity experiments from a manifest."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist, pstdev
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (SCRIPT_DIR, SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from contract_schema import load_suite_contract
from suite_registry import build_default_registry

from worldflux.parity import discover_artifacts, get_backend_adapter_registry, stable_recipe_hash

_LEGACY_SUPPORTED_ADAPTERS: set[str] = {
    "official_dreamerv3",
    "official_tdmpc2",
    "worldflux_dreamerv3_jax",
    "worldflux_dreamerv3_native",
    "worldflux_tdmpc2_native",
}


def _supported_adapters() -> set[str]:
    registry = get_backend_adapter_registry()
    return set(_LEGACY_SUPPORTED_ADAPTERS) | set(registry.list_ids())


class _SafeFormat(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(frozen=True)
class CommandSpec:
    adapter: str
    backend_kind: str | None
    adapter_id: str | None
    cwd: str
    command: list[str]
    env: dict[str, str]
    timeout_sec: int | None
    source_commit: str | None
    source_artifact_path: str | None
    artifact_requirements: dict[str, Any]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    family: str
    required_metrics: tuple[str, ...]
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    higher_is_better: bool
    effect_transform: str
    equivalence_margin: float
    noninferiority_margin: float
    alpha: float
    holm_scope: str
    train_budget: dict[str, Any]
    eval_protocol: dict[str, Any]
    validity_requirements: dict[str, Any]
    official: CommandSpec
    worldflux: CommandSpec


@dataclass(frozen=True)
class SeedPolicy:
    mode: str
    values: tuple[int, ...]
    pilot_seeds: int
    min_seeds: int
    max_seeds: int
    power_target: float


@dataclass(frozen=True)
class Manifest:
    schema_version: str
    suite_id: str
    family: str
    defaults: dict[str, Any]
    train_budget: dict[str, Any]
    eval_protocol: dict[str, Any]
    validity_requirements: dict[str, Any]
    primary_metric: str
    secondary_metrics: tuple[str, ...]
    higher_is_better: bool
    effect_transform: str
    equivalence_margin: float
    noninferiority_margin: float
    alpha: float
    holm_scope: str
    seed_policy: SeedPolicy
    tasks: tuple[TaskSpec, ...]


@dataclass(frozen=True)
class RunContext:
    manifest_path: Path
    run_root: Path
    run_id: str
    device: str
    worldflux_sha: str
    dry_run: bool
    max_retries: int
    task_filter: tuple[str, ...]
    systems: tuple[str, ...]
    shard_index: int
    num_shards: int
    artifact_retention: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/parity"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed-list", type=str, default="")
    parser.add_argument(
        "--pair-plan",
        type=Path,
        default=None,
        help="Optional JSON/JSONL file listing explicit (task_id, seed, system) pairs.",
    )
    parser.add_argument(
        "--systems",
        type=str,
        default="official,worldflux",
        help="Comma-separated systems to execute (official, worldflux).",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--pilot-seeds", type=int, default=None)
    parser.add_argument("--min-seeds", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--power-target", type=float, default=None)
    parser.add_argument("--equivalence-margin", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument(
        "--task-filter",
        type=str,
        default="",
        help="Comma-separated task filters. Supports exact IDs and fnmatch patterns.",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Shard index for distributed execution (0-based).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total shard count for distributed execution.",
    )
    parser.add_argument(
        "--plot-curves",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate learning-curve plots after the run.",
    )
    parser.add_argument(
        "--artifact-retention",
        type=str,
        choices=["full", "minimal"],
        default="minimal",
        help="Retention policy for per-run artifacts after successful execution.",
    )
    return parser.parse_args()


def _load_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Manifest must be valid JSON (YAML optional parser unavailable: install pyyaml)."
            ) from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise RuntimeError("Manifest root must be an object.")
        data = loaded
    if not isinstance(data, dict):
        raise RuntimeError("Manifest root must be an object.")
    return data


def _resolve_source_value(value: str | None, variables: dict[str, Any]) -> str | None:
    if value is None:
        return None
    rendered = _format_recursive(value, variables)
    if rendered is None:
        return None
    return str(rendered)


def _parse_manifest(raw: dict[str, Any]) -> Manifest:
    contract = load_suite_contract(raw, supported_adapters=_supported_adapters())

    tasks: list[TaskSpec] = []
    for task in contract.tasks:
        tasks.append(
            TaskSpec(
                task_id=task.task_id,
                family=task.family,
                required_metrics=task.required_metrics,
                primary_metric=task.primary_metric,
                secondary_metrics=task.secondary_metrics,
                higher_is_better=task.higher_is_better,
                effect_transform=task.effect_transform,
                equivalence_margin=task.equivalence_margin,
                noninferiority_margin=task.noninferiority_margin,
                alpha=task.alpha,
                holm_scope=task.holm_scope,
                train_budget=dict(task.train_budget),
                eval_protocol=dict(task.eval_protocol),
                validity_requirements=dict(task.validity_requirements),
                official=CommandSpec(
                    adapter=task.official.adapter,
                    backend_kind=task.official.backend_kind,
                    adapter_id=task.official.adapter_id,
                    cwd=task.official.cwd,
                    command=task.official.command,
                    env=dict(task.official.env),
                    timeout_sec=task.official.timeout_sec,
                    source_commit=(
                        task.official.source.commit if task.official.source is not None else None
                    ),
                    source_artifact_path=(
                        task.official.source.artifact_path
                        if task.official.source is not None
                        else None
                    ),
                    artifact_requirements=dict(task.official.artifact_requirements),
                ),
                worldflux=CommandSpec(
                    adapter=task.worldflux.adapter,
                    backend_kind=task.worldflux.backend_kind,
                    adapter_id=task.worldflux.adapter_id,
                    cwd=task.worldflux.cwd,
                    command=task.worldflux.command,
                    env=dict(task.worldflux.env),
                    timeout_sec=task.worldflux.timeout_sec,
                    source_commit=(
                        task.worldflux.source.commit if task.worldflux.source is not None else None
                    ),
                    source_artifact_path=(
                        task.worldflux.source.artifact_path
                        if task.worldflux.source is not None
                        else None
                    ),
                    artifact_requirements=dict(task.worldflux.artifact_requirements),
                ),
            )
        )

    return Manifest(
        schema_version=contract.schema_version,
        suite_id=contract.suite_id,
        family=contract.family,
        defaults=dict(contract.defaults),
        train_budget=dict(contract.train_budget),
        eval_protocol=dict(contract.eval_protocol),
        validity_requirements=dict(contract.validity_requirements),
        primary_metric=contract.primary_metric,
        secondary_metrics=contract.secondary_metrics,
        higher_is_better=contract.higher_is_better,
        effect_transform=contract.effect_transform,
        equivalence_margin=contract.equivalence_margin,
        noninferiority_margin=contract.noninferiority_margin,
        alpha=contract.alpha,
        holm_scope=contract.holm_scope,
        seed_policy=SeedPolicy(
            mode=contract.seed_policy.mode,
            values=contract.seed_policy.values,
            pilot_seeds=contract.seed_policy.pilot_seeds,
            min_seeds=contract.seed_policy.min_seeds,
            max_seeds=contract.seed_policy.max_seeds,
            power_target=contract.seed_policy.power_target,
        ),
        tasks=tuple(tasks),
    )


def _format_recursive(value: Any, variables: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format_map(_SafeFormat(variables))
    if isinstance(value, list):
        return [_format_recursive(v, variables) for v in value]
    if isinstance(value, dict):
        return {str(k): _format_recursive(v, variables) for k, v in value.items()}
    return value


def _load_existing_success(run_jsonl: Path) -> set[tuple[str, int, str]]:
    if not run_jsonl.exists():
        return set()
    out: set[tuple[str, int, str]] = set()
    with run_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("status") == "success":
                out.add(
                    (
                        str(entry.get("task_id", "")),
                        int(entry.get("seed", -1)),
                        str(entry.get("system", "")),
                    )
                )
    return out


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


_RETENTION_KEEP_FILENAMES = {"metrics.json", "stdout.log", "stderr.log"}
_RETENTION_HEAVY_DIR_NAMES = {
    "checkpoint",
    "checkpoints",
    "replay",
    "replay_buffer",
    "rollouts",
    "tb",
    "tensorboard",
    "videos",
}
_RETENTION_HEAVY_FILE_SUFFIXES = (
    ".pt",
    ".pth",
    ".ckpt",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
    ".pkl",
    ".zst",
)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _protected_artifact_paths(system_dir: Path, artifact_manifest: Any) -> set[Path]:
    if not isinstance(artifact_manifest, dict):
        return set()

    protected: set[Path] = set()
    for key in ("config_snapshot", "component_match_path"):
        raw = artifact_manifest.get(key)
        if isinstance(raw, str) and raw.strip():
            candidate = Path(raw).expanduser()
            resolved = (
                candidate.resolve()
                if candidate.is_absolute()
                else (system_dir / candidate).resolve()
            )
            if _is_within(resolved, system_dir.resolve()):
                protected.add(resolved)

    for key in ("score_paths", "metrics_paths"):
        raw = artifact_manifest.get(key, [])
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                continue
            candidate = Path(item).expanduser()
            resolved = (
                candidate.resolve()
                if candidate.is_absolute()
                else (system_dir / candidate).resolve()
            )
            if _is_within(resolved, system_dir.resolve()):
                protected.add(resolved)
    return protected


def _path_protected(path: Path, *, protected_paths: set[Path]) -> bool:
    resolved = path.resolve()
    for protected in protected_paths:
        if resolved == protected:
            return True
        if path.is_dir() and _is_within(protected, resolved):
            return True
    return False


def _prune_system_artifacts(system_dir: Path, *, protected_paths: set[Path] | None = None) -> None:
    if not system_dir.exists():
        return
    protected = protected_paths or set()
    for path in sorted(system_dir.rglob("*"), reverse=True):
        if not path.exists():
            continue
        if _path_protected(path, protected_paths=protected):
            continue
        if path.is_dir() and path.name.lower() in _RETENTION_HEAVY_DIR_NAMES:
            shutil.rmtree(path, ignore_errors=True)
            continue
        if not path.is_file():
            continue
        if path.name in _RETENTION_KEEP_FILENAMES:
            continue
        name = path.name.lower()
        if name.endswith(_RETENTION_HEAVY_FILE_SUFFIXES) or name.endswith((".tar", ".tar.gz")):
            path.unlink(missing_ok=True)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _infer_worldflux_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _shell_quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(v) for v in command)


def _load_metrics(path: Path, required_metrics: tuple[str, ...]) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"Metrics file not found: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Metrics file must contain an object: {path}")

    missing = [k for k in required_metrics if k not in loaded]
    if missing:
        raise RuntimeError(f"Missing required metrics {missing} in {path}")

    for key in required_metrics:
        value = loaded[key]
        if not isinstance(value, int | float):
            raise RuntimeError(f"Metric {key!r} must be numeric in {path}, got {type(value)}")
    return loaded


def _run_one(
    *,
    context: RunContext,
    task: TaskSpec,
    system: str,
    seed: int,
    spec: CommandSpec,
    run_jsonl: Path,
    command_manifest: Path,
) -> dict[str, Any]:
    system_dir = context.run_root / "executions" / task.task_id / f"seed_{seed}" / system
    system_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = system_dir / "metrics.json"
    stdout_path = system_dir / "stdout.log"
    stderr_path = system_dir / "stderr.log"

    variables = {
        "run_id": context.run_id,
        "task_id": task.task_id,
        "family": task.family,
        "seed": seed,
        "device": context.device,
        "metrics_out": str(metrics_path),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "run_root": str(context.run_root),
        "worldflux_sha": context.worldflux_sha,
    }
    backend_registry = get_backend_adapter_registry()
    backend_adapter = backend_registry.get(spec.adapter)

    formatted_command: list[str]
    prepared_backend_kind = spec.backend_kind
    prepared_adapter_id = spec.adapter_id or spec.adapter
    prepared_recipe_hash: str | None = None
    if backend_adapter is not None:
        env_backend = str(task.eval_protocol.get("environment_backend", "")).strip().lower()
        env_spec = {
            "task_id": task.task_id,
            "family": task.family,
            "environment_backend": env_backend,
            "eval_protocol": dict(task.eval_protocol),
        }
        backend_spec = backend_adapter.prepare_run(
            recipe=dict(task.train_budget),
            env_spec=env_spec,
            seed=seed,
            run_dir=system_dir,
            repo_root=(
                Path(str(spec.cwd)).expanduser()
                if Path(str(spec.cwd)).is_absolute()
                else (context.manifest_path.parent / spec.cwd).resolve()
            ),
            python_executable=sys.executable,
            device=context.device,
        )
        formatted_command = list(backend_spec.command)
        prepared_backend_kind = backend_spec.backend_kind
        prepared_adapter_id = backend_spec.adapter_id
        prepared_recipe_hash = backend_spec.recipe_hash
    else:
        formatted_command_raw = _format_recursive(spec.command, variables)
        if not isinstance(formatted_command_raw, list) or not all(
            isinstance(item, str) for item in formatted_command_raw
        ):
            raise RuntimeError(
                f"Rendered command must be list[str] for {task.task_id}/{system}/seed={seed}, "
                f"got {type(formatted_command_raw)!r}"
            )
        formatted_command = [str(item) for item in formatted_command_raw]
    formatted_env = _format_recursive(spec.env, variables)
    formatted_cwd_raw = _format_recursive(spec.cwd, variables)
    formatted_source_commit = _resolve_source_value(spec.source_commit, variables)
    formatted_source_artifact_path = _resolve_source_value(spec.source_artifact_path, variables)
    formatted_cwd = Path(str(formatted_cwd_raw)).expanduser()
    if not formatted_cwd.is_absolute():
        formatted_cwd = (context.manifest_path.parent / formatted_cwd).resolve()

    command_str = _shell_quote_command(formatted_command)
    with command_manifest.open("a", encoding="utf-8") as f:
        f.write(
            f"{datetime.now(timezone.utc).isoformat()}\t{task.task_id}\t{seed}\t{system}\t{command_str}\n"
        )

    if context.dry_run:
        record = {
            "schema_version": "parity.v1",
            "run_id": context.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task.task_id,
            "family": task.family,
            "seed": seed,
            "system": system,
            "adapter": spec.adapter,
            "expected_backend_kind": spec.backend_kind,
            "expected_adapter_id": spec.adapter_id or spec.adapter,
            "status": "planned",
            "backend_kind": prepared_backend_kind,
            "adapter_id": prepared_adapter_id,
            "recipe_hash": prepared_recipe_hash,
            "artifact_manifest": None,
            "return_code": None,
            "duration_sec": 0.0,
            "attempt": 0,
            "max_retries": context.max_retries,
            "metrics": {},
            "primary_metric": task.primary_metric,
            "secondary_metrics": list(task.secondary_metrics),
            "effect_transform": task.effect_transform,
            "higher_is_better": task.higher_is_better,
            "equivalence_margin": task.equivalence_margin,
            "noninferiority_margin": task.noninferiority_margin,
            "alpha": task.alpha,
            "holm_scope": task.holm_scope,
            "train_budget": task.train_budget,
            "eval_protocol": task.eval_protocol,
            "validity_requirements": task.validity_requirements,
            "source_commit": formatted_source_commit,
            "source_artifact_path": formatted_source_artifact_path,
            "artifact_requirements": dict(spec.artifact_requirements),
            "success": False,
            "command": command_str,
            "cwd": str(formatted_cwd),
            "artifacts": {
                "metrics": str(metrics_path),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            },
            "error": "",
        }
        _append_jsonl(run_jsonl, record)
        return record

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in dict(formatted_env).items()})

    last_error = ""
    start_total = time.time()
    for attempt in range(context.max_retries + 1):
        attempt_started = time.time()
        try:
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stderr_path.parent.mkdir(parents=True, exist_ok=True)
            with (
                open(stdout_path, "w", encoding="utf-8") as f_out,
                open(stderr_path, "w", encoding="utf-8") as f_err,
            ):
                proc = subprocess.run(
                    formatted_command,
                    cwd=str(formatted_cwd),
                    env=env,
                    stdout=f_out,
                    stderr=f_err,
                    text=True,
                    timeout=spec.timeout_sec,
                    check=False,
                )

            if proc.returncode == 0:
                metrics = _load_metrics(metrics_path, task.required_metrics)
                metadata = metrics.get("metadata", {}) if isinstance(metrics, dict) else {}
                if isinstance(metadata, dict):
                    fallback_recipe_hash = prepared_recipe_hash or stable_recipe_hash(
                        dict(task.train_budget)
                    )
                    metadata.setdefault(
                        "backend_kind",
                        prepared_backend_kind or "legacy_wrapper",
                    )
                    metadata.setdefault("adapter_id", prepared_adapter_id)
                    metadata.setdefault("recipe_hash", fallback_recipe_hash)
                    if backend_adapter is not None:
                        artifact_manifest = backend_adapter.collect_artifacts(
                            run_dir=system_dir,
                            source_commit=formatted_source_commit,
                            eval_protocol_hash=str(metadata.get("eval_protocol_hash", "")) or None,
                            command_argv=formatted_command,
                            recipe=dict(task.train_budget),
                        )
                    else:
                        artifact_manifest = discover_artifacts(
                            run_root=system_dir,
                            backend_kind=str(metadata["backend_kind"]),
                            adapter_id=str(metadata["adapter_id"]),
                            recipe_hash=str(metadata["recipe_hash"]),
                            command_argv=formatted_command,
                            source_commit=formatted_source_commit,
                            eval_protocol_hash=str(metadata.get("eval_protocol_hash", "")) or None,
                        )
                    metadata.setdefault("artifact_manifest", artifact_manifest.to_dict())
                policy_mode = (
                    str(metadata.get("policy_mode", "")) if isinstance(metadata, dict) else ""
                )
                policy_impl = (
                    str(metadata.get("policy_impl", "")) if isinstance(metadata, dict) else ""
                )
                eval_protocol_hash = (
                    str(metadata.get("eval_protocol_hash", ""))
                    if isinstance(metadata, dict)
                    else ""
                )
                record = {
                    "schema_version": "parity.v1",
                    "run_id": context.run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "task_id": task.task_id,
                    "family": task.family,
                    "seed": seed,
                    "system": system,
                    "adapter": spec.adapter,
                    "expected_backend_kind": spec.backend_kind,
                    "expected_adapter_id": spec.adapter_id or spec.adapter,
                    "status": "success",
                    "backend_kind": (
                        str(metadata.get("backend_kind"))
                        if isinstance(metadata, dict) and metadata.get("backend_kind") is not None
                        else prepared_backend_kind
                    ),
                    "adapter_id": (
                        str(metadata.get("adapter_id"))
                        if isinstance(metadata, dict) and metadata.get("adapter_id") is not None
                        else prepared_adapter_id
                    ),
                    "recipe_hash": (
                        str(metadata.get("recipe_hash"))
                        if isinstance(metadata, dict) and metadata.get("recipe_hash") is not None
                        else prepared_recipe_hash
                    ),
                    "artifact_manifest": (
                        metadata.get("artifact_manifest")
                        if isinstance(metadata, dict)
                        and isinstance(metadata.get("artifact_manifest"), dict)
                        else None
                    ),
                    "return_code": proc.returncode,
                    "duration_sec": float(time.time() - attempt_started),
                    "duration_total_sec": float(time.time() - start_total),
                    "attempt": attempt,
                    "max_retries": context.max_retries,
                    "metrics": metrics,
                    "primary_metric": task.primary_metric,
                    "secondary_metrics": list(task.secondary_metrics),
                    "effect_transform": task.effect_transform,
                    "higher_is_better": task.higher_is_better,
                    "equivalence_margin": task.equivalence_margin,
                    "noninferiority_margin": task.noninferiority_margin,
                    "alpha": task.alpha,
                    "holm_scope": task.holm_scope,
                    "train_budget": task.train_budget,
                    "eval_protocol": task.eval_protocol,
                    "validity_requirements": task.validity_requirements,
                    "source_commit": formatted_source_commit,
                    "source_artifact_path": formatted_source_artifact_path,
                    "artifact_requirements": dict(spec.artifact_requirements),
                    "policy_mode": policy_mode,
                    "policy_impl": policy_impl,
                    "eval_protocol_hash": eval_protocol_hash,
                    "success": bool(metrics.get("success", True)),
                    "command": command_str,
                    "cwd": str(formatted_cwd),
                    "artifacts": {
                        "metrics": str(metrics_path),
                        "stdout": str(stdout_path),
                        "stderr": str(stderr_path),
                    },
                    "error": "",
                }
                _append_jsonl(run_jsonl, record)
                if context.artifact_retention == "minimal":
                    _prune_system_artifacts(
                        system_dir,
                        protected_paths=_protected_artifact_paths(
                            system_dir,
                            record.get("artifact_manifest"),
                        ),
                    )
                return record

            stderr_tail = ""
            if stderr_path.exists():
                raw = stderr_path.read_bytes()
                stderr_tail = raw[-2000:].decode("utf-8", errors="replace")
            stderr_preview = stderr_tail[-500:] if stderr_tail else "<empty>"
            last_error = f"non-zero exit code ({proc.returncode}); stderr tail: {stderr_preview}"
            # SIGKILL detection: when the Linux OOM-killer terminates a process,
            # it sends SIGKILL (signal 9) which leaves no stderr trace.
            # returncode == -9  : direct child received SIGKILL
            # returncode == 137 : shell-wrapped 128+9
            # returncode == 247 : Python SystemExit(-9) wraps to 256-9
            is_signal_kill = proc.returncode in (-9, 137, 247)
            is_oom = is_signal_kill or any(
                marker in stderr_tail
                for marker in (
                    "RESOURCE_EXHAUSTED",
                    "Out of memory",
                    "OutOfMemoryError",
                    "CUDA out of memory",
                    "oom-kill",
                )
            )
            if is_oom:
                if is_signal_kill:
                    last_error = f"OOM (SIGKILL, rc={proc.returncode}): {last_error}"
                else:
                    last_error = f"OOM: {last_error}"
                if attempt < context.max_retries:
                    time.sleep(5)  # GPU memory reclaim grace period
        except Exception as exc:  # pragma: no cover - runtime guard
            last_error = str(exc)

    record = {
        "schema_version": "parity.v1",
        "run_id": context.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_id": task.task_id,
        "family": task.family,
        "seed": seed,
        "system": system,
        "adapter": spec.adapter,
        "expected_backend_kind": spec.backend_kind,
        "expected_adapter_id": spec.adapter_id or spec.adapter,
        "status": "failed",
        "backend_kind": prepared_backend_kind,
        "adapter_id": prepared_adapter_id,
        "recipe_hash": prepared_recipe_hash,
        "artifact_manifest": None,
        "return_code": None,
        "duration_sec": float(time.time() - start_total),
        "attempt": context.max_retries,
        "max_retries": context.max_retries,
        "metrics": {},
        "primary_metric": task.primary_metric,
        "secondary_metrics": list(task.secondary_metrics),
        "effect_transform": task.effect_transform,
        "higher_is_better": task.higher_is_better,
        "equivalence_margin": task.equivalence_margin,
        "noninferiority_margin": task.noninferiority_margin,
        "alpha": task.alpha,
        "holm_scope": task.holm_scope,
        "train_budget": task.train_budget,
        "eval_protocol": task.eval_protocol,
        "validity_requirements": task.validity_requirements,
        "source_commit": formatted_source_commit,
        "source_artifact_path": formatted_source_artifact_path,
        "artifact_requirements": dict(spec.artifact_requirements),
        "success": False,
        "command": command_str,
        "cwd": str(formatted_cwd),
        "artifacts": {
            "metrics": str(metrics_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
        "error": last_error,
    }
    _append_jsonl(run_jsonl, record)
    return record


def _parse_seed_list(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        return []
    return [int(v) for v in values]


def _parse_task_filter(raw: str) -> list[str]:
    patterns = [part.strip() for part in raw.split(",") if part.strip()]
    return patterns


def _parse_systems(raw: str) -> tuple[str, ...]:
    values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not values:
        raise RuntimeError("--systems must include at least one value.")
    allowed = {"official", "worldflux"}
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise RuntimeError(
            f"--systems contains unsupported values {sorted(set(unknown))}; "
            f"allowed={sorted(allowed)}."
        )
    return tuple(dict.fromkeys(values))


def _load_pair_plan(
    path: Path,
    *,
    allowed_tasks: set[str],
    allowed_systems: set[str],
) -> list[tuple[str, int, str]]:
    if not path.exists():
        raise RuntimeError(f"--pair-plan not found: {path}")

    rows: list[dict[str, Any]] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    rows.append(parsed)
    else:
        parsed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            pairs = parsed.get("pairs", [])
            if isinstance(pairs, list):
                rows = [item for item in pairs if isinstance(item, dict)]
        elif isinstance(parsed, list):
            rows = [item for item in parsed if isinstance(item, dict)]

    if not rows:
        raise RuntimeError(f"--pair-plan has no entries: {path}")

    out: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for idx, row in enumerate(rows):
        task_id = str(row.get("task_id", "")).strip()
        system = str(row.get("system", "")).strip().lower()
        try:
            seed = int(row.get("seed", -1))
        except Exception as exc:
            raise RuntimeError(f"--pair-plan entry[{idx}] has invalid seed: {row}") from exc
        if not task_id:
            raise RuntimeError(f"--pair-plan entry[{idx}] missing task_id")
        if task_id not in allowed_tasks:
            raise RuntimeError(f"--pair-plan entry[{idx}] unknown task_id: {task_id}")
        if system not in allowed_systems:
            raise RuntimeError(
                f"--pair-plan entry[{idx}] invalid system {system!r}; allowed={sorted(allowed_systems)}"
            )
        if seed < 0:
            raise RuntimeError(f"--pair-plan entry[{idx}] seed must be >= 0")
        key = (task_id, seed, system)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _select_tasks(
    tasks: tuple[TaskSpec, ...],
    *,
    patterns: list[str],
    shard_index: int,
    num_shards: int,
) -> tuple[TaskSpec, ...]:
    if num_shards < 1:
        raise RuntimeError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise RuntimeError("--shard-index must satisfy 0 <= shard-index < num-shards")

    selected = list(tasks)
    if patterns:
        filtered: list[TaskSpec] = []
        for task in selected:
            if any(fnmatch.fnmatch(task.task_id, pattern) for pattern in patterns):
                filtered.append(task)
        selected = filtered

    selected = sorted(selected, key=lambda task: task.task_id)
    sharded = [task for idx, task in enumerate(selected) if idx % num_shards == shard_index]
    if not sharded:
        raise RuntimeError(
            "No tasks selected after applying --task-filter/--shard-index/--num-shards."
        )
    return tuple(sharded)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                out.append(parsed)
    return out


def _collect_paired_metric(
    entries: list[dict[str, Any]], metric: str
) -> dict[str, list[tuple[float, float]]]:
    by_key: dict[tuple[str, int], dict[str, float]] = {}
    for entry in entries:
        if entry.get("status") != "success":
            continue
        task_id = str(entry.get("task_id", ""))
        seed = int(entry.get("seed", -1))
        system = str(entry.get("system", ""))
        metrics = entry.get("metrics", {})
        if not isinstance(metrics, dict) or metric not in metrics:
            continue
        key = (task_id, seed)
        pair = by_key.setdefault(key, {})
        pair[system] = float(metrics[metric])

    out: dict[str, list[tuple[float, float]]] = {}
    for (task_id, _seed), pair in by_key.items():
        if "official" in pair and "worldflux" in pair:
            out.setdefault(task_id, []).append((pair["official"], pair["worldflux"]))
    return out


def _estimate_seed_count(
    *,
    entries: list[dict[str, Any]],
    alpha: float,
    equivalence_margin: float,
    power_target: float,
    min_seeds: int,
    max_seeds: int,
) -> int:
    paired = _collect_paired_metric(entries, "final_return_mean")
    sigmas: list[float] = []
    for task_pairs in paired.values():
        if len(task_pairs) < 2:
            continue
        ratios: list[float] = []
        for off, wf in task_pairs:
            shift = 0.0
            floor = min(off, wf)
            if floor <= 0.0:
                shift = -floor + 1e-8
            ratios.append(math.log((wf + shift + 1e-8) / (off + shift + 1e-8)))
        if len(ratios) >= 2:
            sigmas.append(pstdev(ratios))

    if not sigmas:
        return max_seeds

    sigma = max(sigmas)
    delta = abs(math.log(1.0 + equivalence_margin))
    if delta <= 0:
        return max_seeds
    if sigma == 0:
        return min_seeds

    z_alpha = NormalDist().inv_cdf(1.0 - alpha)
    z_beta = NormalDist().inv_cdf(power_target)
    n_required = math.ceil(((z_alpha + z_beta) * sigma / delta) ** 2)
    return max(min_seeds, min(max_seeds, int(n_required)))


def _write_run_context(
    *,
    context: RunContext,
    manifest: Manifest,
    manifest_hash: str,
    run_jsonl: Path,
    seed_values: list[int],
    selected_tasks: tuple[TaskSpec, ...],
) -> None:
    payload = {
        "schema_version": "parity.v1",
        "run_id": context.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(context.manifest_path),
        "manifest_sha256": manifest_hash,
        "manifest_schema": manifest.schema_version,
        "suite_id": manifest.suite_id,
        "suite_family": manifest.family,
        "worldflux_sha": context.worldflux_sha,
        "device": context.device,
        "dry_run": context.dry_run,
        "max_retries": context.max_retries,
        "artifact_retention": context.artifact_retention,
        "task_filter": list(context.task_filter),
        "systems": list(context.systems),
        "shard_index": context.shard_index,
        "num_shards": context.num_shards,
        "seeds": seed_values,
        "selected_tasks": [task.task_id for task in selected_tasks],
        "suite_contract": {
            "primary_metric": manifest.primary_metric,
            "secondary_metrics": list(manifest.secondary_metrics),
            "higher_is_better": manifest.higher_is_better,
            "effect_transform": manifest.effect_transform,
            "equivalence_margin": manifest.equivalence_margin,
            "noninferiority_margin": manifest.noninferiority_margin,
            "alpha": manifest.alpha,
            "holm_scope": manifest.holm_scope,
            "train_budget": manifest.train_budget,
            "eval_protocol": manifest.eval_protocol,
            "validity_requirements": manifest.validity_requirements,
        },
        "tasks": [
            {
                "task_id": t.task_id,
                "family": t.family,
                "required_metrics": list(t.required_metrics),
                "primary_metric": t.primary_metric,
                "secondary_metrics": list(t.secondary_metrics),
                "higher_is_better": t.higher_is_better,
                "effect_transform": t.effect_transform,
                "equivalence_margin": t.equivalence_margin,
                "noninferiority_margin": t.noninferiority_margin,
                "alpha": t.alpha,
                "holm_scope": t.holm_scope,
                "train_budget": t.train_budget,
                "eval_protocol": t.eval_protocol,
                "validity_requirements": t.validity_requirements,
                "official_adapter": t.official.adapter,
                "worldflux_adapter": t.worldflux.adapter,
                "official_source_commit": t.official.source_commit,
                "worldflux_source_commit": t.worldflux.source_commit,
            }
            for t in selected_tasks
        ],
        "artifacts": {
            "runs_jsonl": str(run_jsonl),
        },
    }
    _atomic_write_text(
        context.run_root / "run_context.json",
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def main() -> int:
    args = _parse_args()
    raw_manifest = _load_manifest(args.manifest)
    manifest = _parse_manifest(raw_manifest)
    registry = build_default_registry()
    for task in manifest.tasks:
        registry.require(task.family)

    run_id = (
        args.run_id.strip() or f"parity_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    run_root = (args.output_dir / run_id).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    context = RunContext(
        manifest_path=args.manifest.resolve(),
        run_root=run_root,
        run_id=run_id,
        device=args.device,
        worldflux_sha=_infer_worldflux_sha(),
        dry_run=bool(args.dry_run),
        max_retries=max(0, int(args.max_retries)),
        task_filter=tuple(_parse_task_filter(args.task_filter)),
        systems=_parse_systems(args.systems),
        shard_index=int(args.shard_index),
        num_shards=int(args.num_shards),
        artifact_retention=str(args.artifact_retention),
    )

    run_jsonl = run_root / "parity_runs.jsonl"
    command_manifest = run_root / "command_manifest.txt"

    if args.resume:
        done_success = _load_existing_success(run_jsonl)
    else:
        done_success = set()
        if run_jsonl.exists():
            run_jsonl.unlink()
        if command_manifest.exists():
            command_manifest.unlink()

    seed_override = _parse_seed_list(args.seed_list)
    selected_tasks = _select_tasks(
        manifest.tasks,
        patterns=list(context.task_filter),
        shard_index=context.shard_index,
        num_shards=context.num_shards,
    )
    task_map = {task.task_id: task for task in selected_tasks}
    pair_plan: list[tuple[str, int, str]] = []
    if args.pair_plan is not None:
        pair_plan = _load_pair_plan(
            args.pair_plan.resolve(),
            allowed_tasks=set(task_map),
            allowed_systems=set(context.systems),
        )

    defaults = dict(manifest.defaults)
    alpha = float(
        args.alpha
        if args.alpha is not None
        else defaults.get("alpha", manifest.alpha if manifest.alpha > 0 else 0.05)
    )
    equivalence_margin = float(
        args.equivalence_margin
        if args.equivalence_margin is not None
        else defaults.get(
            "equivalence_margin",
            manifest.equivalence_margin if manifest.equivalence_margin > 0 else 0.05,
        )
    )

    seed_policy = manifest.seed_policy
    pilot_seeds = int(args.pilot_seeds if args.pilot_seeds is not None else seed_policy.pilot_seeds)
    min_seeds = int(args.min_seeds if args.min_seeds is not None else seed_policy.min_seeds)
    max_seeds = int(args.max_seeds if args.max_seeds is not None else seed_policy.max_seeds)
    power_target = float(
        args.power_target if args.power_target is not None else seed_policy.power_target
    )

    if pair_plan:
        seed_values = sorted({seed for _task_id, seed, _system in pair_plan})
        seed_plan = {
            "mode": "pair_plan",
            "pair_plan_path": str(args.pair_plan.resolve()) if args.pair_plan is not None else "",
            "pair_count": len(pair_plan),
            "seed_values": seed_values,
            "pairs": [
                {"task_id": task_id, "seed": seed, "system": system}
                for task_id, seed, system in pair_plan
            ],
        }
    elif seed_override:
        seed_values = sorted(set(seed_override))
        seed_plan = {
            "mode": "override",
            "seed_values": seed_values,
        }
    elif seed_policy.mode == "fixed":
        seed_values = sorted(set(seed_policy.values))
        if not seed_values:
            seed_values = [0]
        seed_plan = {
            "mode": "fixed",
            "seed_values": seed_values,
        }
    else:
        pilot_values = list(range(pilot_seeds))
        seed_values = pilot_values
        seed_plan = {
            "mode": "auto_power",
            "pilot_seed_values": pilot_values,
            "alpha": alpha,
            "equivalence_margin": equivalence_margin,
            "power_target": power_target,
            "min_seeds": min_seeds,
            "max_seeds": max_seeds,
        }

    manifest_hash = _hash_file(args.manifest)
    _write_run_context(
        context=context,
        manifest=manifest,
        manifest_hash=manifest_hash,
        run_jsonl=run_jsonl,
        seed_values=seed_values,
        selected_tasks=selected_tasks,
    )

    def run_one_pair(task: TaskSpec, *, seed: int, system: str) -> None:
        key = (task.task_id, int(seed), system)
        if key in done_success:
            return
        spec = task.official if system == "official" else task.worldflux
        _run_one(
            context=context,
            task=task,
            system=system,
            seed=int(seed),
            spec=spec,
            run_jsonl=run_jsonl,
            command_manifest=command_manifest,
        )

    def run_seed_set(values: list[int]) -> None:
        for task in selected_tasks:
            for seed in values:
                for system in ("official", "worldflux"):
                    if system not in context.systems:
                        continue
                    run_one_pair(task, seed=int(seed), system=system)

    if pair_plan:
        for task_id, seed, system in pair_plan:
            run_one_pair(task_map[task_id], seed=seed, system=system)
    else:
        run_seed_set(seed_values)

    if not pair_plan and not seed_override and seed_policy.mode == "auto_power":
        entries = _load_jsonl(run_jsonl)
        n_required = _estimate_seed_count(
            entries=entries,
            alpha=alpha,
            equivalence_margin=equivalence_margin,
            power_target=power_target,
            min_seeds=min_seeds,
            max_seeds=max_seeds,
        )
        if n_required > len(seed_values):
            extra = list(range(len(seed_values), n_required))
            run_seed_set(extra)
            seed_values.extend(extra)
        seed_plan["n_required"] = int(n_required)
        seed_plan["seed_values"] = sorted(seed_values)

    _atomic_write_text(
        run_root / "seed_plan.json",
        json.dumps(seed_plan, indent=2, sort_keys=True) + "\n",
    )

    entries = _load_jsonl(run_jsonl)
    success = sum(1 for e in entries if e.get("status") == "success")
    failed = sum(1 for e in entries if e.get("status") == "failed")
    planned = sum(1 for e in entries if e.get("status") == "planned")

    summary = {
        "schema_version": "parity.v1",
        "run_id": context.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "suite_id": manifest.suite_id,
        "suite_family": manifest.family,
        "plugin_families": list(registry.families()),
        "total_records": len(entries),
        "success_records": success,
        "failed_records": failed,
        "planned_records": planned,
        "tasks_selected": [task.task_id for task in selected_tasks],
        "task_filter": list(context.task_filter),
        "systems": list(context.systems),
        "shard_index": context.shard_index,
        "num_shards": context.num_shards,
        "artifact_retention": context.artifact_retention,
        "artifacts": {
            "run_context": str(run_root / "run_context.json"),
            "seed_plan": str(run_root / "seed_plan.json"),
            "parity_runs": str(run_jsonl),
            "command_manifest": str(command_manifest),
        },
    }
    _atomic_write_text(
        run_root / "run_summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
