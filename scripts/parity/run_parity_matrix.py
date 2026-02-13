#!/usr/bin/env python3
"""Run official-vs-WorldFlux parity experiments from a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist, pstdev
from typing import Any

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
    command: str | list[str]
    env: dict[str, str]
    timeout_sec: int | None


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    family: str
    required_metrics: tuple[str, ...]
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
    defaults: dict[str, Any]
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/parity"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed-list", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--pilot-seeds", type=int, default=None)
    parser.add_argument("--min-seeds", type=int, default=None)
    parser.add_argument("--max-seeds", type=int, default=None)
    parser.add_argument("--power-target", type=float, default=None)
    parser.add_argument("--equivalence-margin", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
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


def _coerce_command(value: Any, *, name: str) -> str | list[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return [str(v) for v in value]
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

    return CommandSpec(
        adapter=adapter,
        cwd=cwd,
        command=command,
        env=dict(env),
        timeout_sec=timeout,
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
    schema = _require_string(raw.get("schema_version"), name="schema_version")
    if schema != "parity.manifest.v1":
        raise RuntimeError(f"Unsupported schema_version '{schema}'. Expected 'parity.manifest.v1'.")

    defaults = _require_object(raw.get("defaults", {}), name="defaults")
    seed_policy = _parse_seed_policy(raw.get("seed_policy", {}))

    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        raise RuntimeError("tasks must be a non-empty list.")

    tasks: list[TaskSpec] = []
    seen_ids: set[str] = set()
    for idx, task_raw in enumerate(tasks_raw):
        name = f"tasks[{idx}]"
        task_obj = _require_object(task_raw, name=name)
        task_id = _require_string(task_obj.get("task_id"), name=f"{name}.task_id")
        if task_id in seen_ids:
            raise RuntimeError(f"Duplicate task_id: {task_id}")
        seen_ids.add(task_id)

        family = _require_string(task_obj.get("family"), name=f"{name}.family")
        metrics_raw = task_obj.get("required_metrics", ["final_return_mean", "auc_return"])
        if not isinstance(metrics_raw, list) or not all(isinstance(v, str) for v in metrics_raw):
            raise RuntimeError(f"{name}.required_metrics must be list[str]")

        official = _parse_command_spec(task_obj.get("official"), name=f"{name}.official")
        worldflux = _parse_command_spec(task_obj.get("worldflux"), name=f"{name}.worldflux")
        tasks.append(
            TaskSpec(
                task_id=task_id,
                family=family,
                required_metrics=tuple(metrics_raw),
                official=official,
                worldflux=worldflux,
            )
        )

    return Manifest(
        schema_version=schema,
        defaults=dict(defaults),
        seed_policy=seed_policy,
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


def _shell_quote_command(command: str | list[str]) -> str:
    if isinstance(command, str):
        return command
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

    formatted_command = _format_recursive(spec.command, variables)
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
                shell=isinstance(formatted_command, str),
                timeout=spec.timeout_sec,
                check=False,
            )
            stdout_path.write_text(proc.stdout, encoding="utf-8")
            stderr_path.write_text(proc.stderr, encoding="utf-8")

            if proc.returncode == 0:
                metrics = _load_metrics(metrics_path, task.required_metrics)
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
) -> None:
    payload = {
        "schema_version": "parity.v1",
        "run_id": context.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(context.manifest_path),
        "manifest_sha256": manifest_hash,
        "manifest_schema": manifest.schema_version,
        "worldflux_sha": context.worldflux_sha,
        "device": context.device,
        "dry_run": context.dry_run,
        "max_retries": context.max_retries,
        "seeds": seed_values,
        "tasks": [
            {
                "task_id": t.task_id,
                "family": t.family,
                "required_metrics": list(t.required_metrics),
                "official_adapter": t.official.adapter,
                "worldflux_adapter": t.worldflux.adapter,
            }
            for t in manifest.tasks
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

    defaults = dict(manifest.defaults)
    alpha = float(args.alpha if args.alpha is not None else defaults.get("alpha", 0.05))
    equivalence_margin = float(
        args.equivalence_margin
        if args.equivalence_margin is not None
        else defaults.get("equivalence_margin", 0.05)
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
    )

    def run_seed_set(values: list[int]) -> None:
        for task in manifest.tasks:
            for seed in values:
                for system, spec in (("official", task.official), ("worldflux", task.worldflux)):
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
        "total_records": len(entries),
        "success_records": success,
        "failed_records": failed,
        "planned_records": planned,
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
