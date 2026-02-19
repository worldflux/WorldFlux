#!/usr/bin/env python3
"""Run official-vs-WorldFlux parity matrix across AWS SSM shard workers."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist, pstdev
from typing import Any


@dataclass(frozen=True)
class TaskCost:
    task_id: str
    family: str
    estimated_steps: int
    estimated_seeds: int
    estimated_cost: float
    estimated_duration_sec: float


@dataclass(frozen=True)
class ShardPlan:
    shard_id: int
    instance_id: str
    task_ids: tuple[str, ...]
    shard_run_id: str
    estimated_cost: float
    estimated_duration_sec: float
    seed_values: tuple[int, ...]
    systems: tuple[str, ...]
    gpu_slot: int | None


@dataclass(frozen=True)
class PhaseResult:
    run_id: str
    return_code: int
    summary_path: Path
    summary: dict[str, Any]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--instance-ids", type=str, default="")
    parser.add_argument("--official-instance-ids", type=str, default="")
    parser.add_argument("--worldflux-instance-ids", type=str, default="")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--full-manifest",
        type=Path,
        default=Path("scripts/parity/manifests/official_vs_worldflux_full_v1.yaml"),
    )
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--s3-prefix", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--systems", type=str, default="official,worldflux")
    parser.add_argument("--seed-list", type=str, default="")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=172800)
    parser.add_argument("--poll-interval-sec", type=int, default=30)
    parser.add_argument("--sync-interval-sec", type=int, default=300)
    parser.add_argument(
        "--resume-from-s3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch prior shard artifacts from S3 before running remote shard command.",
    )
    parser.add_argument("--worldflux-sha", type=str, default="")
    parser.add_argument(
        "--dreamer-sha",
        type=str,
        default="b65cf81a6fb13625af8722127459283f899a35d9",
    )
    parser.add_argument(
        "--tdmpc2-sha",
        type=str,
        default="8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe",
    )
    parser.add_argument("--workspace-root", type=str, default="/opt/parity")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/parity"))
    parser.add_argument(
        "--wait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wait for remote shard completion before merging results.",
    )

    parser.add_argument(
        "--sharding-mode",
        type=str,
        choices=["task", "seed_system"],
        default="task",
    )
    parser.add_argument(
        "--phase-plan",
        type=str,
        choices=["single", "two_stage_proof"],
        default="single",
    )
    parser.add_argument(
        "--phase-gate",
        type=str,
        choices=["strict_pass", "always_continue"],
        default="strict_pass",
    )
    parser.add_argument("--pilot-seed-list", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--seed-min", type=int, default=20)
    parser.add_argument("--seed-max", type=int, default=50)
    parser.add_argument("--power-target", type=float, default=0.80)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--equivalence-margin", type=float, default=0.05)

    parser.add_argument(
        "--thread-limit-profile",
        type=str,
        choices=["off", "strict1"],
        default="off",
    )
    parser.add_argument(
        "--cpu-affinity-policy",
        type=str,
        choices=["none", "p4d_8gpu_12vcpu"],
        default="none",
    )
    parser.add_argument("--gpu-slots-per-instance", type=int, default=8)

    parser.add_argument(
        "--auto-provision",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--auto-terminate",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--fleet-size", type=int, default=4)
    parser.add_argument("--fleet-split", type=str, default="2,2")
    parser.add_argument("--instance-type", type=str, default="p4d.24xlarge")
    parser.add_argument("--image-id", type=str, default="")
    parser.add_argument("--subnet-id", type=str, default="")
    parser.add_argument("--security-group-ids", type=str, default="")
    parser.add_argument("--iam-instance-profile", type=str, default="")
    parser.add_argument("--key-name", type=str, default="")
    parser.add_argument("--volume-size-gb", type=int, default=200)
    parser.add_argument("--provision-timeout-sec", type=int, default=1800)
    return parser.parse_args()


def _run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, capture_output=True)


def _load_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Manifest must be JSON, or install pyyaml for YAML support."
            ) from exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise RuntimeError("Manifest root must be object")
    return payload


def _command_tokens(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item)]
    if isinstance(raw, str):
        return shlex.split(raw, posix=True)
    return []


def _extract_steps(raw_command: Any) -> int | None:
    tokens = _command_tokens(raw_command)
    for idx, token in enumerate(tokens[:-1]):
        if token == "--steps":
            try:
                return int(tokens[idx + 1])
            except ValueError:
                return None
    return None


def _family_default_steps(family: str) -> int:
    if family == "dreamerv3":
        return 110_000
    if family == "tdmpc2":
        return 7_000_000
    return 100_000


def _family_steps_per_second(family: str) -> float:
    if family == "dreamerv3":
        return 900.0
    if family == "tdmpc2":
        return 700.0
    return 500.0


def _estimate_seed_count(payload: dict[str, Any]) -> int:
    seed_policy = payload.get("seed_policy", {})
    if not isinstance(seed_policy, dict):
        return 1

    mode = str(seed_policy.get("mode", "fixed"))
    values = seed_policy.get("values", [])
    if mode == "fixed":
        if isinstance(values, list):
            numeric = [int(v) for v in values if isinstance(v, int)]
            return max(1, len(set(numeric)))
        return 1

    min_seeds = seed_policy.get("min_seeds", 20)
    if isinstance(min_seeds, int) and min_seeds > 0:
        return int(min_seeds)
    return 20


def _manifest_task_costs(path: Path) -> list[TaskCost]:
    payload = _load_manifest(path)
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise RuntimeError("manifest.tasks must be list")

    estimated_seeds = _estimate_seed_count(payload)
    train_budget = payload.get("train_budget", {})
    default_steps = (
        int(train_budget.get("steps", 0))
        if isinstance(train_budget, dict) and isinstance(train_budget.get("steps"), int)
        else 0
    )

    out: list[TaskCost] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id", "")).strip()
        if not task_id:
            continue
        family = str(task.get("family", "")).strip() or "unknown"
        fallback_steps = default_steps if default_steps > 0 else _family_default_steps(family)

        official_steps = _extract_steps((task.get("official") or {}).get("command"))
        worldflux_steps = _extract_steps((task.get("worldflux") or {}).get("command"))
        step_candidates = [
            value for value in (official_steps, worldflux_steps) if value is not None
        ]
        estimated_steps = max(step_candidates) if step_candidates else fallback_steps

        estimated_cost = float(max(1, estimated_steps) * max(1, estimated_seeds) * 2)
        estimated_duration_sec = float(estimated_cost / max(1.0, _family_steps_per_second(family)))

        out.append(
            TaskCost(
                task_id=task_id,
                family=family,
                estimated_steps=int(estimated_steps),
                estimated_seeds=int(estimated_seeds),
                estimated_cost=estimated_cost,
                estimated_duration_sec=estimated_duration_sec,
            )
        )

    if not out:
        raise RuntimeError("manifest has no tasks")
    return sorted(out, key=lambda item: item.task_id)


def _parse_instance_ids(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _parse_seed_list(raw: str) -> list[int]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        return []
    return sorted({int(v) for v in values})


def _parse_systems(raw: str) -> tuple[str, ...]:
    values = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not values:
        raise RuntimeError("--systems must include at least one value")
    allowed = {"official", "worldflux"}
    unknown = [value for value in values if value not in allowed]
    if unknown:
        raise RuntimeError(f"Unknown systems: {sorted(set(unknown))}")
    return tuple(dict.fromkeys(values))


def _parse_fleet_split(raw: str) -> tuple[int, int]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 2:
        raise RuntimeError("--fleet-split must be '<official_count>,<worldflux_count>'")
    official_count = int(parts[0])
    worldflux_count = int(parts[1])
    if official_count < 0 or worldflux_count < 0:
        raise RuntimeError("--fleet-split counts must be non-negative")
    return official_count, worldflux_count


def _parse_security_group_ids(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _manifest_relpath(manifest: Path) -> str:
    repo_root = Path.cwd().resolve()
    manifest_resolved = manifest.resolve()
    try:
        return manifest_resolved.relative_to(repo_root).as_posix()
    except Exception:
        return manifest.name


def _thread_env(profile: str) -> dict[str, str]:
    if profile == "strict1":
        return {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    return {}


def _cpu_affinity_for_slot(policy: str, slot: int | None) -> str | None:
    if policy != "p4d_8gpu_12vcpu" or slot is None:
        return None
    if slot < 0 or slot > 7:
        return None
    start = slot * 12
    end = start + 11
    return f"{start}-{end}"


def _build_task_shards(
    run_id: str,
    tasks: list[TaskCost],
    instances: list[str],
    *,
    seed_values: list[int],
    systems: tuple[str, ...],
) -> list[ShardPlan]:
    if not instances:
        raise RuntimeError("At least one instance is required for task sharding")

    buckets: list[dict[str, Any]] = [
        {"instance_id": instance, "tasks": [], "cost": 0.0, "duration": 0.0}
        for instance in instances
    ]
    for task in sorted(tasks, key=lambda row: row.estimated_cost, reverse=True):
        target = min(buckets, key=lambda item: (float(item["cost"]), len(item["tasks"])))
        target["tasks"].append(task)
        target["cost"] = float(target["cost"]) + task.estimated_cost
        target["duration"] = float(target["duration"]) + task.estimated_duration_sec

    plans: list[ShardPlan] = []
    for idx, bucket in enumerate(buckets):
        chunk: list[TaskCost] = list(bucket["tasks"])
        if not chunk:
            continue
        task_ids = tuple(sorted(task.task_id for task in chunk))
        plans.append(
            ShardPlan(
                shard_id=idx,
                instance_id=str(bucket["instance_id"]),
                task_ids=task_ids,
                shard_run_id=f"{run_id}_shard{idx:02d}",
                estimated_cost=float(bucket["cost"]),
                estimated_duration_sec=float(bucket["duration"]),
                seed_values=tuple(seed_values),
                systems=systems,
                gpu_slot=None,
            )
        )
    if not plans:
        raise RuntimeError("No shard plan generated")
    return plans


def _build_seed_system_shards(
    run_id: str,
    tasks: list[TaskCost],
    *,
    seed_values: list[int],
    systems: tuple[str, ...],
    official_instances: list[str],
    worldflux_instances: list[str],
    gpu_slots_per_instance: int,
) -> list[ShardPlan]:
    if not seed_values:
        raise RuntimeError("seed_system sharding requires non-empty seed list")
    if gpu_slots_per_instance < 1:
        raise RuntimeError("--gpu-slots-per-instance must be >= 1")
    if not official_instances:
        raise RuntimeError("seed_system sharding requires at least one official instance")
    if not worldflux_instances:
        raise RuntimeError("seed_system sharding requires at least one worldflux instance")

    task_by_id = {task.task_id: task for task in tasks}

    slot_load: dict[tuple[str, int], float] = {}
    for instance_id in official_instances + worldflux_instances:
        for slot in range(gpu_slots_per_instance):
            slot_load[(instance_id, slot)] = 0.0

    def select_slot(system: str) -> tuple[str, int]:
        candidates = official_instances if system == "official" else worldflux_instances
        return min(
            ((instance, slot) for instance in candidates for slot in range(gpu_slots_per_instance)),
            key=lambda key: (slot_load[key], key[0], key[1]),
        )

    plans: list[ShardPlan] = []
    shard_id = 0
    packed_seeds = tuple(sorted({int(seed) for seed in seed_values}))
    for task_id in sorted(task_by_id):
        task = task_by_id[task_id]
        single_system_cost = max(1.0, float(task.estimated_steps))
        single_system_duration = float(
            single_system_cost / max(1.0, _family_steps_per_second(task.family))
        )
        total_cost = single_system_cost * float(len(packed_seeds))
        total_duration = single_system_duration * float(len(packed_seeds))
        for system in systems:
            instance_id, slot = select_slot(system)
            slot_load[(instance_id, slot)] += total_cost
            plans.append(
                ShardPlan(
                    shard_id=shard_id,
                    instance_id=instance_id,
                    task_ids=(task.task_id,),
                    shard_run_id=f"{run_id}_shard{shard_id:02d}",
                    estimated_cost=total_cost,
                    estimated_duration_sec=total_duration,
                    seed_values=packed_seeds,
                    systems=(system,),
                    gpu_slot=slot,
                )
            )
            shard_id += 1

    if not plans:
        raise RuntimeError("No seed_system shards generated")
    return plans


def _sync_command(remote_run_root: str, shard_prefix: str) -> str:
    include_args = " ".join(
        [
            "--exclude '*'",
            "--include 'parity_runs.jsonl'",
            "--include 'seed_plan.json'",
            "--include 'run_context.json'",
            "--include 'run_summary.json'",
            "--include 'coverage_report.json'",
            "--include 'command_manifest.txt'",
            "--include 'runner.stdout.log'",
            "--include 'runner.stderr.log'",
        ]
    )
    return f"aws s3 sync {remote_run_root} {shard_prefix} {include_args}"


def _build_remote_commands(
    *,
    shard: ShardPlan,
    manifest_rel: str,
    run_id: str,
    s3_prefix: str,
    device: str,
    max_retries: int,
    workspace_root: str,
    worldflux_sha: str,
    dreamer_sha: str,
    tdmpc2_sha: str,
    sync_interval_sec: int,
    resume_from_s3: bool,
    thread_limit_profile: str,
    cpu_affinity_policy: str,
) -> list[str]:
    instance_root = f"{workspace_root.rstrip('/')}/{run_id}/{shard.instance_id}"
    remote_run_root = f"reports/parity/{shard.shard_run_id}"
    shard_prefix = f"{s3_prefix.rstrip('/')}/shards/{shard.shard_id:02d}"
    sync_cmd = _sync_command(remote_run_root, shard_prefix)

    task_filter = ",".join(shard.task_ids)
    systems_csv = ",".join(shard.systems)
    seeds_csv = ",".join(str(v) for v in shard.seed_values)

    wf_ref = worldflux_sha.strip() or "origin/main"
    thread_env = _thread_env(thread_limit_profile)
    cpu_affinity = _cpu_affinity_for_slot(cpu_affinity_policy, shard.gpu_slot)

    run_parts = [
        "python3 scripts/parity/run_parity_matrix.py",
        f"--manifest {shlex.quote(manifest_rel)}",
        f"--run-id {shlex.quote(shard.shard_run_id)}",
        "--output-dir reports/parity",
        f"--device {shlex.quote(device)}",
        f"--max-retries {int(max_retries)}",
        f"--task-filter {shlex.quote(task_filter)}",
        f"--systems {shlex.quote(systems_csv)}",
        "--resume",
    ]
    if seeds_csv:
        run_parts.append(f"--seed-list {shlex.quote(seeds_csv)}")

    runner_cmd = " ".join(run_parts)
    if cpu_affinity:
        runner_cmd = f"taskset -c {shlex.quote(cpu_affinity)} {runner_cmd}"

    commands = [
        "set -eu",
        f"mkdir -p {shlex.quote(instance_root)}",
        f"cd {shlex.quote(instance_root)}",
        "exec 9>.repo_setup.lock",
        "flock 9",
        "if [ ! -d worldflux/.git ]; then git clone https://github.com/worldflux/WorldFlux.git worldflux; fi",
        "if [ ! -d dreamerv3-official/.git ]; then git clone https://github.com/danijar/dreamerv3.git dreamerv3-official; fi",
        "if [ ! -d tdmpc2-official/.git ]; then git clone https://github.com/nicklashansen/tdmpc2.git tdmpc2-official; fi",
        f"cd worldflux && git fetch origin --tags --prune && git checkout {shlex.quote(wf_ref)}",
        f"cd ../dreamerv3-official && git fetch origin --tags --prune && git checkout {shlex.quote(dreamer_sha)}",
        f"cd ../tdmpc2-official && git fetch origin --tags --prune && git checkout {shlex.quote(tdmpc2_sha)}",
        "cd ../worldflux",
        "if [ ! -f ../.venv/.parity_ready ]; then",
        "  if [ ! -x ../.venv/bin/python3 ]; then python3 -m venv ../.venv; fi",
        "  . ../.venv/bin/activate",
        "  python -m pip install --upgrade pip",
        "  python -m pip install -e .",
        "  python -m pip install hydra-core omegaconf termcolor tensordict torchrl gymnasium dm-control mujoco imageio imageio-ffmpeg h5py kornia tqdm pandas wandb",
        "  python -m pip install -r ../dreamerv3-official/requirements.txt",
        "  python -m pip install pyyaml gymnasium ale-py dm-control mujoco hydra-core omegaconf",
        "  touch ../.venv/.parity_ready",
        "fi",
        ". ../.venv/bin/activate",
        "flock -u 9",
        f"mkdir -p {shlex.quote(remote_run_root)}",
    ]

    if resume_from_s3:
        commands.extend(
            [
                f"aws s3 cp {shlex.quote(shard_prefix + '/parity_runs.jsonl')} {shlex.quote(remote_run_root + '/parity_runs.jsonl')} || true",
                f"aws s3 cp {shlex.quote(shard_prefix + '/seed_plan.json')} {shlex.quote(remote_run_root + '/seed_plan.json')} || true",
                f"aws s3 cp {shlex.quote(shard_prefix + '/run_context.json')} {shlex.quote(remote_run_root + '/run_context.json')} || true",
                f"aws s3 cp {shlex.quote(shard_prefix + '/run_summary.json')} {shlex.quote(remote_run_root + '/run_summary.json')} || true",
            ]
        )

    for key, value in thread_env.items():
        commands.append(f"export {key}={shlex.quote(value)}")
    if shard.gpu_slot is not None:
        commands.append(f"export CUDA_VISIBLE_DEVICES={int(shard.gpu_slot)}")

    commands.extend(
        [
            f"{runner_cmd} > {shlex.quote(remote_run_root + '/runner.stdout.log')} 2> {shlex.quote(remote_run_root + '/runner.stderr.log')} &",
            "RUNNER_PID=$!",
            (
                f'(while kill -0 "$RUNNER_PID" 2>/dev/null; do {sync_cmd} >/dev/null 2>&1 || true; '
                f"sleep {max(10, int(sync_interval_sec))}; done) &"
            ),
            "SYNC_PID=$!",
            'if wait "$RUNNER_PID"; then RUNNER_RC=0; else RUNNER_RC=$?; fi',
            'kill "$SYNC_PID" >/dev/null 2>&1 || true',
            f"{sync_cmd} >/dev/null 2>&1 || true",
            '[ "$RUNNER_RC" -eq 0 ] || exit "$RUNNER_RC"',
            (
                "python3 scripts/parity/validate_matrix_completeness.py "
                f"--manifest {shlex.quote(manifest_rel)} "
                f"--runs {shlex.quote(remote_run_root + '/parity_runs.jsonl')} "
                f"--seed-plan {shlex.quote(remote_run_root + '/seed_plan.json')} "
                f"--run-context {shlex.quote(remote_run_root + '/run_context.json')} "
                f"--output {shlex.quote(remote_run_root + '/coverage_report.json')} "
                f"--systems {shlex.quote(systems_csv)} "
                "--max-missing-pairs 0"
            ),
            f"aws s3 cp {shlex.quote(remote_run_root + '/parity_runs.jsonl')} {shlex.quote(shard_prefix + '/parity_runs.jsonl')}",
            f"aws s3 cp {shlex.quote(remote_run_root + '/seed_plan.json')} {shlex.quote(shard_prefix + '/seed_plan.json')}",
            f"aws s3 cp {shlex.quote(remote_run_root + '/run_context.json')} {shlex.quote(shard_prefix + '/run_context.json')}",
            f"aws s3 cp {shlex.quote(remote_run_root + '/run_summary.json')} {shlex.quote(shard_prefix + '/run_summary.json')}",
            f"aws s3 cp {shlex.quote(remote_run_root + '/coverage_report.json')} {shlex.quote(shard_prefix + '/coverage_report.json')}",
            f"aws s3 cp {shlex.quote(remote_run_root + '/command_manifest.txt')} {shlex.quote(shard_prefix + '/command_manifest.txt')} || true",
            f"aws s3 cp {shlex.quote(remote_run_root + '/runner.stdout.log')} {shlex.quote(shard_prefix + '/runner.stdout.log')} || true",
            f"aws s3 cp {shlex.quote(remote_run_root + '/runner.stderr.log')} {shlex.quote(shard_prefix + '/runner.stderr.log')} || true",
        ]
    )
    return commands


def _send_command(
    *,
    region: str,
    instance_id: str,
    timeout_seconds: int,
    commands: list[str],
) -> str:
    payload = json.dumps(commands)
    cli = [
        "aws",
        "ssm",
        "send-command",
        "--region",
        region,
        "--instance-ids",
        instance_id,
        "--document-name",
        "AWS-RunShellScript",
        "--timeout-seconds",
        str(timeout_seconds),
        "--parameters",
        f"commands={payload}",
        "--query",
        "Command.CommandId",
        "--output",
        "text",
    ]
    result = _run_cli(cli)
    if result.returncode != 0:
        raise RuntimeError(f"send-command failed for {instance_id}: {result.stderr}")
    return result.stdout.strip()


def _poll_command(
    *,
    region: str,
    command_id: str,
    instance_id: str,
    poll_interval_sec: int,
) -> dict[str, Any]:
    while True:
        cli = [
            "aws",
            "ssm",
            "get-command-invocation",
            "--region",
            region,
            "--command-id",
            command_id,
            "--instance-id",
            instance_id,
            "--output",
            "json",
        ]
        result = _run_cli(cli)
        if result.returncode != 0:
            raise RuntimeError(
                f"get-command-invocation failed for {instance_id} command {command_id}: {result.stderr}"
            )
        payload = json.loads(result.stdout)
        status = str(payload.get("Status", ""))
        if status in {"Success", "Failed", "Cancelled", "TimedOut", "Undeliverable", "Terminated"}:
            return payload
        time.sleep(max(1, poll_interval_sec))


def _download_shard_artifacts(
    *,
    region: str,
    s3_prefix: str,
    shard_id: int,
    target_dir: Path,
) -> list[Path]:
    files = [
        "parity_runs.jsonl",
        "seed_plan.json",
        "run_context.json",
        "run_summary.json",
        "coverage_report.json",
        "command_manifest.txt",
        "runner.stdout.log",
        "runner.stderr.log",
    ]
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    shard_prefix = f"{s3_prefix.rstrip('/')}/shards/{shard_id:02d}"

    for name in files:
        dest = target_dir / name
        cli = ["aws", "s3", "cp", f"{shard_prefix}/{name}", str(dest), "--region", region]
        result = _run_cli(cli)
        if result.returncode != 0:
            continue
        downloaded.append(dest)
    return downloaded


def _run_local_script(
    script: Path,
    args: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(script), *args]
    result = _run_cli(command)
    if check and result.returncode != 0:
        raise RuntimeError(f"script failed: {' '.join(command)}\n{result.stderr}\n{result.stdout}")
    return result


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso8601(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _duration_seconds(started_at: Any, ended_at: Any) -> float | None:
    start = _parse_iso8601(started_at)
    end = _parse_iso8601(ended_at)
    if start is None or end is None:
        return None
    return max(0.0, float((end - start).total_seconds()))


def _timeout_risk(*, predicted_duration_sec: float, timeout_seconds: int) -> str:
    if timeout_seconds <= 0:
        return "unknown"
    ratio = predicted_duration_sec / float(timeout_seconds)
    if ratio >= 0.90:
        return "high"
    if ratio >= 0.60:
        return "medium"
    return "low"


def _upload_artifact(*, region: str, artifact: Path, final_prefix: str) -> None:
    if not artifact.exists():
        return
    cli = [
        "aws",
        "s3",
        "cp",
        str(artifact),
        f"{final_prefix}/{artifact.name}",
        "--region",
        region,
    ]
    result = _run_cli(cli)
    if result.returncode != 0:
        raise RuntimeError(f"failed uploading {artifact} to {final_prefix}: {result.stderr}")


def _wait_instances_running(region: str, instance_ids: list[str]) -> None:
    if not instance_ids:
        return
    cli = [
        "aws",
        "ec2",
        "wait",
        "instance-running",
        "--region",
        region,
        "--instance-ids",
        *instance_ids,
    ]
    result = _run_cli(cli)
    if result.returncode != 0:
        raise RuntimeError(f"instance-running wait failed: {result.stderr}")


def _wait_ssm_online(region: str, instance_ids: list[str], timeout_sec: int) -> None:
    deadline = time.time() + max(60, timeout_sec)
    pending = set(instance_ids)
    while pending and time.time() < deadline:
        done: set[str] = set()
        for instance_id in list(pending):
            cli = [
                "aws",
                "ssm",
                "describe-instance-information",
                "--region",
                region,
                "--filters",
                f"Key=InstanceIds,Values={instance_id}",
                "--query",
                "InstanceInformationList[0].PingStatus",
                "--output",
                "text",
            ]
            result = _run_cli(cli)
            if result.returncode == 0 and result.stdout.strip() == "Online":
                done.add(instance_id)
        pending -= done
        if pending:
            time.sleep(10)
    if pending:
        raise RuntimeError(f"Instances never became SSM Online: {sorted(pending)}")


def _provision_instances(args: argparse.Namespace) -> list[str]:
    if not args.image_id.strip():
        raise RuntimeError("--image-id is required when --auto-provision is enabled")
    if not args.subnet_id.strip():
        raise RuntimeError("--subnet-id is required when --auto-provision is enabled")
    if not args.security_group_ids.strip():
        raise RuntimeError("--security-group-ids is required when --auto-provision is enabled")
    if not args.iam_instance_profile.strip():
        raise RuntimeError("--iam-instance-profile is required when --auto-provision is enabled")
    if not args.key_name.strip():
        raise RuntimeError("--key-name is required when --auto-provision is enabled")

    sg_ids = _parse_security_group_ids(args.security_group_ids)
    if not sg_ids:
        raise RuntimeError("--security-group-ids must include at least one security group")

    cli = [
        "aws",
        "ec2",
        "run-instances",
        "--region",
        args.region,
        "--image-id",
        args.image_id,
        "--instance-type",
        args.instance_type,
        "--count",
        str(int(args.fleet_size)),
        "--key-name",
        args.key_name,
        "--subnet-id",
        args.subnet_id,
        "--security-group-ids",
        *sg_ids,
        "--iam-instance-profile",
        f"Name={args.iam_instance_profile}",
        "--block-device-mappings",
        (
            "DeviceName=/dev/sda1,"
            f"Ebs={{VolumeSize={int(args.volume_size_gb)},VolumeType=gp3,DeleteOnTermination=true}}"
        ),
        "--tag-specifications",
        (
            "ResourceType=instance,Tags=["
            f"{{Key=Name,Value=worldflux-parity-{args.run_id}}},"
            "{Key=ManagedBy,Value=worldflux-parity-orchestrator}]"
        ),
        "--query",
        "Instances[].InstanceId",
        "--output",
        "text",
    ]
    result = _run_cli(cli)
    if result.returncode != 0:
        raise RuntimeError(f"run-instances failed: {result.stderr}")
    instance_ids = [part.strip() for part in result.stdout.split() if part.strip()]
    if len(instance_ids) != int(args.fleet_size):
        raise RuntimeError(
            f"Expected {args.fleet_size} instances, got {len(instance_ids)}: {instance_ids}"
        )

    _wait_instances_running(args.region, instance_ids)
    _wait_ssm_online(args.region, instance_ids, timeout_sec=int(args.provision_timeout_sec))
    return instance_ids


def _terminate_instances(region: str, instance_ids: list[str]) -> None:
    if not instance_ids:
        return
    cli = ["aws", "ec2", "terminate-instances", "--region", region, "--instance-ids", *instance_ids]
    result = _run_cli(cli)
    if result.returncode != 0:
        raise RuntimeError(f"terminate-instances failed: {result.stderr}")


def _resolve_fleets(args: argparse.Namespace) -> tuple[list[str], list[str], list[str], list[str]]:
    created_ids: list[str] = []

    if args.auto_provision:
        created_ids = _provision_instances(args)
        official_count, worldflux_count = _parse_fleet_split(args.fleet_split)
        if official_count + worldflux_count != len(created_ids):
            raise RuntimeError(
                "--fleet-split does not match provisioned instance count "
                f"({official_count}+{worldflux_count}!={len(created_ids)})"
            )
        official_ids = created_ids[:official_count]
        worldflux_ids = created_ids[official_count : official_count + worldflux_count]
        return official_ids, worldflux_ids, list(created_ids), created_ids

    explicit_official = _parse_instance_ids(args.official_instance_ids)
    explicit_worldflux = _parse_instance_ids(args.worldflux_instance_ids)

    if explicit_official or explicit_worldflux:
        all_ids = sorted(set(explicit_official + explicit_worldflux))
        return explicit_official, explicit_worldflux, all_ids, created_ids

    base_ids = _parse_instance_ids(args.instance_ids)
    if not base_ids:
        raise RuntimeError(
            "Provide either --instance-ids or explicit --official-instance-ids/--worldflux-instance-ids"
        )

    if args.sharding_mode == "task":
        return [], [], base_ids, created_ids

    official_count, worldflux_count = _parse_fleet_split(args.fleet_split)
    if official_count + worldflux_count > len(base_ids):
        half = max(1, len(base_ids) // 2)
        official_ids = base_ids[:half]
        worldflux_ids = base_ids[half:]
        if not worldflux_ids:
            worldflux_ids = list(official_ids)
        return official_ids, worldflux_ids, base_ids, created_ids

    if official_count == 0 and worldflux_count == 0:
        half = len(base_ids) // 2
        official_ids = base_ids[:half]
        worldflux_ids = base_ids[half:]
    else:
        official_ids = base_ids[:official_count]
        worldflux_ids = base_ids[official_count : official_count + worldflux_count]

    return official_ids, worldflux_ids, base_ids, created_ids


def _estimate_required_seed_count_from_runs(
    *,
    runs_path: Path,
    alpha: float,
    equivalence_margin: float,
    power_target: float,
    min_seeds: int,
    max_seeds: int,
) -> int:
    if not runs_path.exists():
        return max_seeds

    by_key: dict[tuple[str, int], dict[str, float]] = {}
    with runs_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if not isinstance(entry, dict):
                continue
            if entry.get("status") != "success":
                continue
            task_id = str(entry.get("task_id", "")).strip()
            system = str(entry.get("system", "")).strip()
            seed = int(entry.get("seed", -1))
            metrics = entry.get("metrics", {})
            if not task_id or seed < 0 or system not in {"official", "worldflux"}:
                continue
            if not isinstance(metrics, dict):
                continue
            value = metrics.get("final_return_mean")
            if not isinstance(value, int | float):
                continue
            by_key.setdefault((task_id, seed), {})[system] = float(value)

    by_task: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for (task_id, _seed), pair in by_key.items():
        if "official" in pair and "worldflux" in pair:
            by_task[task_id].append((pair["official"], pair["worldflux"]))

    sigmas: list[float] = []
    for values in by_task.values():
        if len(values) < 2:
            continue
        ratios: list[float] = []
        for off, wf in values:
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


def _execute_phase(
    *,
    args: argparse.Namespace,
    manifest_path: Path,
    run_id: str,
    s3_prefix: str,
    seed_values: list[int],
    systems: tuple[str, ...],
    official_instances: list[str],
    worldflux_instances: list[str],
    all_instances: list[str],
) -> PhaseResult:
    task_costs = _manifest_task_costs(manifest_path)
    manifest_rel = _manifest_relpath(manifest_path)

    if args.sharding_mode == "seed_system":
        shards = _build_seed_system_shards(
            run_id,
            task_costs,
            seed_values=seed_values,
            systems=systems,
            official_instances=official_instances,
            worldflux_instances=worldflux_instances,
            gpu_slots_per_instance=int(args.gpu_slots_per_instance),
        )
    else:
        shards = _build_task_shards(
            run_id,
            task_costs,
            all_instances,
            seed_values=seed_values,
            systems=systems,
        )

    submissions: list[dict[str, Any]] = []
    for shard in shards:
        commands = _build_remote_commands(
            shard=shard,
            manifest_rel=manifest_rel,
            run_id=run_id,
            s3_prefix=s3_prefix,
            device=args.device,
            max_retries=args.max_retries,
            workspace_root=args.workspace_root,
            worldflux_sha=args.worldflux_sha,
            dreamer_sha=args.dreamer_sha,
            tdmpc2_sha=args.tdmpc2_sha,
            sync_interval_sec=args.sync_interval_sec,
            resume_from_s3=bool(args.resume_from_s3),
            thread_limit_profile=args.thread_limit_profile,
            cpu_affinity_policy=args.cpu_affinity_policy,
        )
        command_id = _send_command(
            region=args.region,
            instance_id=shard.instance_id,
            timeout_seconds=args.timeout_seconds,
            commands=commands,
        )
        submissions.append(
            {
                "shard_id": shard.shard_id,
                "instance_id": shard.instance_id,
                "task_count": len(shard.task_ids),
                "task_ids": list(shard.task_ids),
                "seed_values": list(shard.seed_values),
                "systems": list(shard.systems),
                "gpu_slot": shard.gpu_slot,
                "run_id": shard.shard_run_id,
                "command_id": command_id,
                "estimated_cost": shard.estimated_cost,
                "predicted_duration_sec": shard.estimated_duration_sec,
                "timeout_risk": _timeout_risk(
                    predicted_duration_sec=shard.estimated_duration_sec,
                    timeout_seconds=args.timeout_seconds,
                ),
                "submitted_at": _timestamp(),
            }
        )

    results: list[dict[str, Any]] = []
    if args.wait:
        for item in submissions:
            status_payload = _poll_command(
                region=args.region,
                command_id=item["command_id"],
                instance_id=item["instance_id"],
                poll_interval_sec=args.poll_interval_sec,
            )
            item["status"] = status_payload.get("Status")
            item["response_code"] = status_payload.get("ResponseCode")
            item["status_details"] = status_payload.get("StatusDetails")
            item["stdout"] = status_payload.get("StandardOutputContent", "")
            item["stderr"] = status_payload.get("StandardErrorContent", "")
            item["completed_at"] = _timestamp()
            item["actual_duration_sec"] = _duration_seconds(
                item["submitted_at"], item["completed_at"]
            )
            results.append(item)
    else:
        results = submissions

    failed = [row for row in results if args.wait and str(row.get("status")) != "Success"]

    local_root = (args.output_dir / run_id).resolve()
    shards_root = local_root / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)

    if not args.wait:
        summary = {
            "schema_version": "parity.v1",
            "generated_at": _timestamp(),
            "run_id": run_id,
            "region": args.region,
            "manifest": str(manifest_path.resolve()),
            "device": args.device,
            "instances": all_instances,
            "submissions": submissions,
            "results": results,
            "failed_shards": 0,
            "artifacts": {},
            "s3_final_prefix": f"{s3_prefix.rstrip('/')}/final",
            "timing": {
                "predicted_total_duration_sec": float(
                    sum(float(item.estimated_duration_sec) for item in shards)
                ),
                "actual_wait_duration_sec": None,
                "timeout_seconds": int(args.timeout_seconds),
            },
        }
        summary_path = local_root / "orchestrator_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return PhaseResult(
            run_id=run_id,
            return_code=0,
            summary_path=summary_path,
            summary=summary,
        )

    shard_jsonl_paths: list[Path] = []
    for item in results:
        shard_dir = shards_root / f"shard_{int(item['shard_id']):02d}"
        downloaded = _download_shard_artifacts(
            region=args.region,
            s3_prefix=s3_prefix,
            shard_id=int(item["shard_id"]),
            target_dir=shard_dir,
        )
        for path in downloaded:
            if path.name == "parity_runs.jsonl":
                shard_jsonl_paths.append(path)

    merged_runs = local_root / "parity_runs.jsonl"
    merge_summary = local_root / "merge_summary.json"
    merge_script = Path("scripts/parity/merge_parity_runs.py").resolve()

    merge_args: list[str] = ["--output", str(merged_runs), "--summary-output", str(merge_summary)]
    for shard_path in sorted(set(shard_jsonl_paths)):
        merge_args.extend(["--input", str(shard_path)])
    if "--input" not in merge_args:
        raise RuntimeError("No shard parity_runs.jsonl artifacts were downloaded from S3.")
    _run_local_script(merge_script, merge_args)

    seed_plan_path = None
    run_context_path = None
    for shard_dir in sorted(shards_root.glob("shard_*")):
        if seed_plan_path is None and (shard_dir / "seed_plan.json").exists():
            seed_plan_path = shard_dir / "seed_plan.json"
        if run_context_path is None and (shard_dir / "run_context.json").exists():
            run_context_path = shard_dir / "run_context.json"

    coverage_report = local_root / "coverage_report.json"
    rerun_manifest = local_root / "rerun_missing_manifest.json"
    coverage_script = Path("scripts/parity/validate_matrix_completeness.py").resolve()
    coverage_args = [
        "--manifest",
        str(manifest_path.resolve()),
        "--runs",
        str(merged_runs),
        "--output",
        str(coverage_report),
        "--systems",
        ",".join(systems),
        "--max-missing-pairs",
        "0",
        "--rerun-manifest-output",
        str(rerun_manifest),
    ]
    if seed_plan_path is not None:
        coverage_args.extend(["--seed-plan", str(seed_plan_path)])
    if run_context_path is not None:
        coverage_args.extend(["--run-context", str(run_context_path)])
    coverage_result = _run_local_script(coverage_script, coverage_args, check=False)

    coverage_payload = (
        json.loads(coverage_report.read_text(encoding="utf-8")) if coverage_report.exists() else {}
    )
    missing_pairs = int(coverage_payload.get("missing_pairs", 0) or 0)

    equivalence_report = local_root / "equivalence_report.json"
    equivalence_md = local_root / "equivalence_report.md"
    validity_report = local_root / "validity_report.json"
    stats_failed = False
    stats_error = ""

    if coverage_result.returncode == 0:
        try:
            stats_script = Path("scripts/parity/stats_equivalence.py").resolve()
            _run_local_script(
                stats_script,
                [
                    "--input",
                    str(merged_runs),
                    "--output",
                    str(equivalence_report),
                    "--manifest",
                    str(manifest_path.resolve()),
                    "--systems",
                    ",".join(systems),
                    "--strict-completeness",
                    "--strict-validity",
                    "--proof-mode",
                    "--validity-report",
                    str(validity_report),
                ],
            )

            report_script = Path("scripts/parity/report_markdown.py").resolve()
            _run_local_script(
                report_script,
                [
                    "--input",
                    str(equivalence_report),
                    "--output",
                    str(equivalence_md),
                ],
            )
        except RuntimeError as exc:
            stats_failed = True
            stats_error = str(exc)
    else:
        stats_failed = True
        stats_error = (
            "coverage validation failed before stats/report generation. "
            "Use rerun_missing_manifest.json with --resume to rerun missing pairs only."
        )

    final_prefix = f"{s3_prefix.rstrip('/')}/final"
    for artifact in (
        merged_runs,
        coverage_report,
        rerun_manifest,
        validity_report,
        equivalence_report,
        equivalence_md,
        merge_summary,
    ):
        _upload_artifact(region=args.region, artifact=artifact, final_prefix=final_prefix)

    predicted_total_duration_sec = float(sum(float(item.estimated_duration_sec) for item in shards))
    actual_wait_duration_sec: float | None = None
    if submissions and results:
        starts = [_parse_iso8601(item.get("submitted_at")) for item in submissions]
        ends = [_parse_iso8601(item.get("completed_at")) for item in results]
        starts = [stamp for stamp in starts if stamp is not None]
        ends = [stamp for stamp in ends if stamp is not None]
        if starts and ends:
            actual_wait_duration_sec = max(0.0, float((max(ends) - min(starts)).total_seconds()))

    summary = {
        "schema_version": "parity.v1",
        "generated_at": _timestamp(),
        "run_id": run_id,
        "region": args.region,
        "manifest": str(manifest_path.resolve()),
        "device": args.device,
        "systems": list(systems),
        "instances": all_instances,
        "submissions": submissions,
        "results": results,
        "failed_shards": len(failed),
        "coverage": {
            "return_code": int(coverage_result.returncode),
            "missing_pairs": missing_pairs,
            "pass": bool(coverage_result.returncode == 0),
            "rerun_manifest": str(rerun_manifest) if rerun_manifest.exists() else "",
        },
        "timing": {
            "predicted_total_duration_sec": predicted_total_duration_sec,
            "actual_wait_duration_sec": actual_wait_duration_sec,
            "timeout_seconds": int(args.timeout_seconds),
            "timeout_risk": _timeout_risk(
                predicted_duration_sec=predicted_total_duration_sec,
                timeout_seconds=args.timeout_seconds,
            ),
        },
        "artifacts": {
            "merged_runs": str(merged_runs),
            "coverage_report": str(coverage_report),
            "rerun_manifest": str(rerun_manifest) if rerun_manifest.exists() else "",
            "validity_report": str(validity_report) if validity_report.exists() else "",
            "equivalence_report": str(equivalence_report) if equivalence_report.exists() else "",
            "equivalence_markdown": str(equivalence_md) if equivalence_md.exists() else "",
            "merge_summary": str(merge_summary),
        },
        "s3_final_prefix": final_prefix,
        "errors": {
            "stats_or_report": stats_error,
        },
    }

    summary_path = local_root / "orchestrator_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    rc = 0
    if failed:
        rc = 1
    if coverage_result.returncode != 0:
        rc = 1
    if stats_failed:
        rc = 1

    return PhaseResult(
        run_id=run_id,
        return_code=rc,
        summary_path=summary_path,
        summary=summary,
    )


def _phase_gate_passed(report_path: Path) -> bool:
    if not report_path.exists():
        return False
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    global_payload = payload.get("global", {}) if isinstance(payload, dict) else {}
    if not isinstance(global_payload, dict):
        return False
    return bool(global_payload.get("parity_pass_final", False))


def _run_single_phase(args: argparse.Namespace) -> int:
    systems = _parse_systems(args.systems)
    seed_values = _parse_seed_list(args.seed_list)
    if not seed_values:
        manifest_payload = _load_manifest(args.manifest)
        seed_policy = manifest_payload.get("seed_policy", {})
        if isinstance(seed_policy, dict) and str(seed_policy.get("mode", "fixed")) == "fixed":
            values = seed_policy.get("values", [])
            if isinstance(values, list) and all(isinstance(v, int) for v in values):
                seed_values = sorted(set(values))
        if not seed_values:
            seed_values = list(range(max(1, int(args.seed_min))))

    official_instances, worldflux_instances, all_instances, created_ids = _resolve_fleets(args)

    try:
        phase = _execute_phase(
            args=args,
            manifest_path=args.manifest,
            run_id=args.run_id,
            s3_prefix=args.s3_prefix,
            seed_values=seed_values,
            systems=systems,
            official_instances=official_instances,
            worldflux_instances=worldflux_instances,
            all_instances=all_instances,
        )
        return int(phase.return_code)
    finally:
        if created_ids and args.auto_terminate:
            _terminate_instances(args.region, created_ids)


def _run_two_stage_proof(args: argparse.Namespace) -> int:
    systems = _parse_systems(args.systems)
    if systems != ("official", "worldflux"):
        raise RuntimeError("two_stage_proof requires --systems official,worldflux")

    pilot_seeds = _parse_seed_list(args.pilot_seed_list)
    if not pilot_seeds:
        raise RuntimeError("--pilot-seed-list must not be empty")

    official_instances, worldflux_instances, all_instances, created_ids = _resolve_fleets(args)
    root_dir = (args.output_dir / args.run_id).resolve()
    root_dir.mkdir(parents=True, exist_ok=True)

    two_stage_summary: dict[str, Any] = {
        "schema_version": "parity.v1",
        "generated_at": _timestamp(),
        "run_id": args.run_id,
        "phase_plan": "two_stage_proof",
        "region": args.region,
        "manifest": str(args.manifest.resolve()),
        "full_manifest": str(args.full_manifest.resolve()),
        "systems": list(systems),
        "pilot_seeds": pilot_seeds,
        "phases": {},
        "gate": {},
        "instances": {
            "official": official_instances,
            "worldflux": worldflux_instances,
            "all": all_instances,
            "created": created_ids,
        },
    }

    try:
        pilot_run_id = f"{args.run_id}_pilot"
        pilot_prefix = f"{args.s3_prefix.rstrip('/')}/pilot"
        pilot_phase = _execute_phase(
            args=args,
            manifest_path=args.manifest,
            run_id=pilot_run_id,
            s3_prefix=pilot_prefix,
            seed_values=pilot_seeds,
            systems=systems,
            official_instances=official_instances,
            worldflux_instances=worldflux_instances,
            all_instances=all_instances,
        )
        two_stage_summary["phases"]["pilot"] = {
            "run_id": pilot_phase.run_id,
            "return_code": pilot_phase.return_code,
            "summary": str(pilot_phase.summary_path),
            "artifacts": pilot_phase.summary.get("artifacts", {}),
        }
        if pilot_phase.return_code != 0:
            summary_path = root_dir / "two_stage_summary.json"
            summary_path.write_text(
                json.dumps(two_stage_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return 1

        merged_runs_path = Path(pilot_phase.summary.get("artifacts", {}).get("merged_runs", ""))
        if not merged_runs_path.exists():
            raise RuntimeError("Pilot merged runs artifact missing")

        n_required = _estimate_required_seed_count_from_runs(
            runs_path=merged_runs_path,
            alpha=float(args.alpha),
            equivalence_margin=float(args.equivalence_margin),
            power_target=float(args.power_target),
            min_seeds=int(args.seed_min),
            max_seeds=int(args.seed_max),
        )
        full_seeds = list(range(int(n_required)))
        two_stage_summary["gate"]["n_required"] = int(n_required)
        two_stage_summary["gate"]["full_seeds"] = full_seeds

        full_run_id = f"{args.run_id}_full"
        full_prefix = f"{args.s3_prefix.rstrip('/')}/full"
        full_phase = _execute_phase(
            args=args,
            manifest_path=args.manifest,
            run_id=full_run_id,
            s3_prefix=full_prefix,
            seed_values=full_seeds,
            systems=systems,
            official_instances=official_instances,
            worldflux_instances=worldflux_instances,
            all_instances=all_instances,
        )
        two_stage_summary["phases"]["full"] = {
            "run_id": full_phase.run_id,
            "return_code": full_phase.return_code,
            "summary": str(full_phase.summary_path),
            "artifacts": full_phase.summary.get("artifacts", {}),
        }
        if full_phase.return_code != 0:
            summary_path = root_dir / "two_stage_summary.json"
            summary_path.write_text(
                json.dumps(two_stage_summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return 1

        full_eq_path = Path(full_phase.summary.get("artifacts", {}).get("equivalence_report", ""))
        gate_pass = _phase_gate_passed(full_eq_path)
        two_stage_summary["gate"]["strict_pass"] = gate_pass

        run_suite = gate_pass or args.phase_gate == "always_continue"
        two_stage_summary["gate"]["run_full_suite"] = bool(run_suite)

        suite_rc = 0
        if run_suite:
            suite_run_id = f"{args.run_id}_suite65"
            suite_prefix = f"{args.s3_prefix.rstrip('/')}/suite65"
            suite_phase = _execute_phase(
                args=args,
                manifest_path=args.full_manifest,
                run_id=suite_run_id,
                s3_prefix=suite_prefix,
                seed_values=full_seeds,
                systems=systems,
                official_instances=official_instances,
                worldflux_instances=worldflux_instances,
                all_instances=all_instances,
            )
            two_stage_summary["phases"]["suite65"] = {
                "run_id": suite_phase.run_id,
                "return_code": suite_phase.return_code,
                "summary": str(suite_phase.summary_path),
                "artifacts": suite_phase.summary.get("artifacts", {}),
            }
            suite_rc = int(suite_phase.return_code)

        summary_path = root_dir / "two_stage_summary.json"
        summary_path.write_text(
            json.dumps(two_stage_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if args.phase_gate == "strict_pass" and not gate_pass:
            return 1
        return int(suite_rc)
    finally:
        if created_ids and args.auto_terminate:
            _terminate_instances(args.region, created_ids)


def main() -> int:
    args = _parse_args()
    if args.phase_plan == "two_stage_proof":
        return _run_two_stage_proof(args)
    return _run_single_phase(args)


if __name__ == "__main__":
    raise SystemExit(main())
