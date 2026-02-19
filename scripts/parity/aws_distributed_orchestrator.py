#!/usr/bin/env python3
"""Run official-vs-WorldFlux parity matrix across AWS SSM shard workers."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--instance-ids", type=str, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--s3-prefix", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
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


def _manifest_tasks(path: Path) -> list[str]:
    return [item.task_id for item in _manifest_task_costs(path)]


def _parse_instance_ids(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise RuntimeError("--instance-ids must include at least one ID")
    return values


def _build_shards(run_id: str, tasks: list[TaskCost], instances: list[str]) -> list[ShardPlan]:
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
            )
        )
    if not plans:
        raise RuntimeError("No shard plan generated")
    return plans


def _manifest_relpath(manifest: Path) -> str:
    repo_root = Path.cwd().resolve()
    manifest_resolved = manifest.resolve()
    try:
        return manifest_resolved.relative_to(repo_root).as_posix()
    except Exception:
        return manifest.name


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
) -> list[str]:
    shard_dir = (
        f"{workspace_root.rstrip('/')}/{run_id}/{shard.instance_id}/shard_{shard.shard_id:02d}"
    )
    wf_sha_expr = worldflux_sha.strip() or "$(git rev-parse HEAD)"
    task_filter = ",".join(shard.task_ids)
    remote_run_root = f"reports/parity/{shard.shard_run_id}"
    shard_prefix = f"{s3_prefix.rstrip('/')}/shards/{shard.shard_id:02d}"
    sync_cmd = _sync_command(remote_run_root, shard_prefix)

    commands = [
        "set -euo pipefail",
        f"mkdir -p {shard_dir} && cd {shard_dir}",
        "git clone https://github.com/worldflux/WorldFlux.git worldflux",
        "git clone https://github.com/danijar/dreamerv3.git dreamerv3-official",
        "git clone https://github.com/nicklashansen/tdmpc2.git tdmpc2-official",
        f"cd worldflux && git checkout {wf_sha_expr}",
        f"cd ../dreamerv3-official && git checkout {dreamer_sha}",
        f"cd ../tdmpc2-official && git checkout {tdmpc2_sha}",
        "cd ../worldflux",
        f"mkdir -p {remote_run_root}",
    ]

    if resume_from_s3:
        commands.extend(
            [
                f"aws s3 cp {shard_prefix}/parity_runs.jsonl {remote_run_root}/parity_runs.jsonl || true",
                f"aws s3 cp {shard_prefix}/seed_plan.json {remote_run_root}/seed_plan.json || true",
                f"aws s3 cp {shard_prefix}/run_context.json {remote_run_root}/run_context.json || true",
                f"aws s3 cp {shard_prefix}/run_summary.json {remote_run_root}/run_summary.json || true",
            ]
        )

    commands.extend(
        [
            (
                "python3 scripts/parity/run_parity_matrix.py "
                f"--manifest {manifest_rel} "
                f"--run-id {shard.shard_run_id} "
                "--output-dir reports/parity "
                f"--device {device} "
                f"--max-retries {max_retries} "
                f"--task-filter '{task_filter}' "
                "--resume "
                f"> {remote_run_root}/runner.stdout.log 2> {remote_run_root}/runner.stderr.log &"
            ),
            "RUNNER_PID=$!",
            (
                f'(while kill -0 "$RUNNER_PID" 2>/dev/null; do {sync_cmd} >/dev/null 2>&1 || true; '
                f"sleep {max(10, int(sync_interval_sec))}; done) &"
            ),
            "SYNC_PID=$!",
            'wait "$RUNNER_PID"',
            "RUNNER_RC=$?",
            'kill "$SYNC_PID" >/dev/null 2>&1 || true',
            f"{sync_cmd} >/dev/null 2>&1 || true",
            '[ "$RUNNER_RC" -eq 0 ] || exit "$RUNNER_RC"',
            (
                "python3 scripts/parity/validate_matrix_completeness.py "
                f"--manifest {manifest_rel} "
                f"--runs {remote_run_root}/parity_runs.jsonl "
                f"--seed-plan {remote_run_root}/seed_plan.json "
                f"--run-context {remote_run_root}/run_context.json "
                f"--output {remote_run_root}/coverage_report.json "
                "--max-missing-pairs 0"
            ),
            f"aws s3 cp {remote_run_root}/parity_runs.jsonl {shard_prefix}/parity_runs.jsonl",
            f"aws s3 cp {remote_run_root}/seed_plan.json {shard_prefix}/seed_plan.json",
            f"aws s3 cp {remote_run_root}/run_context.json {shard_prefix}/run_context.json",
            f"aws s3 cp {remote_run_root}/run_summary.json {shard_prefix}/run_summary.json",
            f"aws s3 cp {remote_run_root}/coverage_report.json {shard_prefix}/coverage_report.json",
            f"aws s3 cp {remote_run_root}/command_manifest.txt {shard_prefix}/command_manifest.txt || true",
            f"aws s3 cp {remote_run_root}/runner.stdout.log {shard_prefix}/runner.stdout.log || true",
            f"aws s3 cp {remote_run_root}/runner.stderr.log {shard_prefix}/runner.stderr.log || true",
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


def main() -> int:
    args = _parse_args()

    task_costs = _manifest_task_costs(args.manifest)
    instances = _parse_instance_ids(args.instance_ids)
    shards = _build_shards(args.run_id, task_costs, instances)
    manifest_rel = _manifest_relpath(args.manifest)

    submissions: list[dict[str, Any]] = []
    for shard in shards:
        commands = _build_remote_commands(
            shard=shard,
            manifest_rel=manifest_rel,
            run_id=args.run_id,
            s3_prefix=args.s3_prefix,
            device=args.device,
            max_retries=args.max_retries,
            workspace_root=args.workspace_root,
            worldflux_sha=args.worldflux_sha,
            dreamer_sha=args.dreamer_sha,
            tdmpc2_sha=args.tdmpc2_sha,
            sync_interval_sec=args.sync_interval_sec,
            resume_from_s3=bool(args.resume_from_s3),
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

    local_root = (args.output_dir / args.run_id).resolve()
    shards_root = local_root / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)

    if not args.wait:
        summary = {
            "schema_version": "parity.v1",
            "generated_at": _timestamp(),
            "run_id": args.run_id,
            "region": args.region,
            "manifest": str(args.manifest.resolve()),
            "device": args.device,
            "instances": instances,
            "submissions": submissions,
            "results": results,
            "failed_shards": 0,
            "artifacts": {},
            "s3_final_prefix": f"{args.s3_prefix.rstrip('/')}/final",
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
        return 0

    shard_jsonl_paths: list[Path] = []
    for item in results:
        shard_dir = shards_root / f"shard_{int(item['shard_id']):02d}"
        downloaded = _download_shard_artifacts(
            region=args.region,
            s3_prefix=args.s3_prefix,
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
        str(args.manifest.resolve()),
        "--runs",
        str(merged_runs),
        "--output",
        str(coverage_report),
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
                    str(args.manifest.resolve()),
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

    final_prefix = f"{args.s3_prefix.rstrip('/')}/final"
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
        "run_id": args.run_id,
        "region": args.region,
        "manifest": str(args.manifest.resolve()),
        "device": args.device,
        "instances": instances,
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

    if failed:
        return 1
    if coverage_result.returncode != 0:
        return 1
    if stats_failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
