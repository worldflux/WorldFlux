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
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist, pstdev
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from contract_schema import load_suite_contract
from suite_registry import build_default_registry

SUPPORTED_ADAPTERS: set[str] = {
    "official_dreamerv3",
    "official_tdmpc2",
    "worldflux_dreamerv3_native",
    "worldflux_tdmpc2_native",
}


class _SafeFormat(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(frozen=True)
class CommandSpec:
    adapter: str
    cwd: str
    command: list[str]
    env: dict[str, str]
    timeout_sec: int | None
    source_commit: str | None
    source_artifact_path: str | None


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/parity"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed-list", type=str, default="")
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


def _require_object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be an object.")
    return value


def _require_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{name} must be a non-empty string.")
    return value


def _coerce_command(value: Any, *, name: str) -> list[str]:
    if isinstance(value, str):
        tokens = shlex.split(value, posix=True)
        if not tokens:
            raise RuntimeError(f"{name} must not be an empty command string.")
        dangerous_tokens = {";", "&&", "||", "|", "&"}
        if any(token in dangerous_tokens for token in tokens):
            raise RuntimeError(
                f"{name} contains shell control operators {sorted(dangerous_tokens)}; "
                "use list[str] without shell operators."
            )
        return [str(token) for token in tokens]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        out = [str(v) for v in value if str(v)]
        if not out:
            raise RuntimeError(f"{name} must not be an empty command list.")
        return out
    raise RuntimeError(f"{name} must be string or list[str].")


def _parse_command_spec(raw: Any, *, name: str) -> CommandSpec:
    obj = _require_object(raw, name=name)
    adapter = _require_string(obj.get("adapter"), name=f"{name}.adapter")
    if adapter not in SUPPORTED_ADAPTERS:
        raise RuntimeError(
            f"Unsupported adapter '{adapter}' in {name}.adapter. Supported: {sorted(SUPPORTED_ADAPTERS)}"
        )
    cwd = _require_string(obj.get("cwd", "."), name=f"{name}.cwd")
    command = _coerce_command(obj.get("command"), name=f"{name}.command")

    env = obj.get("env", {})
    if not isinstance(env, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env.items()
    ):
        raise RuntimeError(f"{name}.env must be a mapping of string keys and values.")

    timeout = obj.get("timeout_sec", None)
    if timeout is not None:
        if not isinstance(timeout, int) or timeout <= 0:
            raise RuntimeError(f"{name}.timeout_sec must be a positive integer when provided.")

    source = obj.get("source", None)
    source_commit: str | None = None
    source_artifact_path: str | None = None
    if source is not None:
        source_obj = _require_object(source, name=f"{name}.source")
        source_commit = _require_string(source_obj.get("commit"), name=f"{name}.source.commit")
        source_artifact_path = _require_string(
            source_obj.get("artifact_path"),
            name=f"{name}.source.artifact_path",
        )

    return CommandSpec(
        adapter=adapter,
        cwd=cwd,
        command=command,
        env=dict(env),
        timeout_sec=timeout,
        source_commit=source_commit,
        source_artifact_path=source_artifact_path,
    )


def _parse_seed_policy(raw: Any) -> SeedPolicy:
    obj = _require_object(raw, name="seed_policy")
    mode = _require_string(obj.get("mode", "fixed"), name="seed_policy.mode")
    if mode not in {"fixed", "auto_power"}:
        raise RuntimeError("seed_policy.mode must be either 'fixed' or 'auto_power'.")

    values_raw = obj.get("values", [])
    if not isinstance(values_raw, list) or not all(isinstance(v, int) for v in values_raw):
        raise RuntimeError("seed_policy.values must be a list[int].")

    pilot_seeds = int(obj.get("pilot_seeds", 10))
    min_seeds = int(obj.get("min_seeds", 20))
    max_seeds = int(obj.get("max_seeds", 50))
    power_target = float(obj.get("power_target", 0.80))
    if pilot_seeds < 1:
        raise RuntimeError("seed_policy.pilot_seeds must be >= 1")
    if not (1 <= min_seeds <= max_seeds):
        raise RuntimeError("seed_policy must satisfy 1 <= min_seeds <= max_seeds")
    if not (0.5 <= power_target < 1.0):
        raise RuntimeError("seed_policy.power_target must be in [0.5, 1.0)")

    return SeedPolicy(
        mode=mode,
        values=tuple(int(v) for v in values_raw),
        pilot_seeds=pilot_seeds,
        min_seeds=min_seeds,
        max_seeds=max_seeds,
        power_target=power_target,
    )


def _parse_manifest(raw: dict[str, Any]) -> Manifest:
    contract = load_suite_contract(raw, supported_adapters=SUPPORTED_ADAPTERS)

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
                ),
                worldflux=CommandSpec(
                    adapter=task.worldflux.adapter,
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
            "status": "planned",
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
            "eval_protocol": task.eval_protocol,
            "validity_requirements": task.validity_requirements,
            "source_commit": spec.source_commit,
            "source_artifact_path": spec.source_artifact_path,
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
            proc = subprocess.run(
                formatted_command,
                cwd=str(formatted_cwd),
                env=env,
                capture_output=True,
                text=True,
                timeout=spec.timeout_sec,
                check=False,
            )
            stdout_path.write_text(proc.stdout, encoding="utf-8")
            stderr_path.write_text(proc.stderr, encoding="utf-8")

            if proc.returncode == 0:
                metrics = _load_metrics(metrics_path, task.required_metrics)
                metadata = metrics.get("metadata", {}) if isinstance(metrics, dict) else {}
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
                    "status": "success",
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
                    "eval_protocol": task.eval_protocol,
                    "validity_requirements": task.validity_requirements,
                    "source_commit": spec.source_commit,
                    "source_artifact_path": spec.source_artifact_path,
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
                return record

            last_error = (
                f"non-zero exit code ({proc.returncode}); stderr tail: "
                f"{proc.stderr[-500:] if proc.stderr else '<empty>'}"
            )
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
        "status": "failed",
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
        "eval_protocol": task.eval_protocol,
        "validity_requirements": task.validity_requirements,
        "source_commit": spec.source_commit,
        "source_artifact_path": spec.source_artifact_path,
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
    (context.run_root / "run_context.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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

    if seed_override:
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

    def run_seed_set(values: list[int]) -> None:
        for task in selected_tasks:
            for seed in values:
                for system, spec in (("official", task.official), ("worldflux", task.worldflux)):
                    if system not in context.systems:
                        continue
                    key = (task.task_id, int(seed), system)
                    if key in done_success:
                        continue
                    _run_one(
                        context=context,
                        task=task,
                        system=system,
                        seed=int(seed),
                        spec=spec,
                        run_jsonl=run_jsonl,
                        command_manifest=command_manifest,
                    )

    run_seed_set(seed_values)

    if not seed_override and seed_policy.mode == "auto_power":
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

    (run_root / "seed_plan.json").write_text(
        json.dumps(seed_plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
        "artifacts": {
            "run_context": str(run_root / "run_context.json"),
            "seed_plan": str(run_root / "seed_plan.json"),
            "parity_runs": str(run_jsonl),
            "command_manifest": str(command_manifest),
        },
    }
    (run_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
