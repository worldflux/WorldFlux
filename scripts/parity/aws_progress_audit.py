#!/usr/bin/env python3
"""Audit AWS parity pilot progress for official and worldflux systems."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

TERMINAL_SSM_STATUSES = {
    "Success",
    "Cancelled",
    "Failed",
    "TimedOut",
    "Undeliverable",
    "Terminated",
}


@dataclass(frozen=True)
class InProgressRecord:
    command_id: str
    instance_id: str
    requested_at: datetime | None
    run_id: str
    shard_id: str
    system: str
    seed_list: str
    status: str


@dataclass
class ShardState:
    shard_id: str
    system: str = "unknown"
    expected: int = 0
    started: int = 0
    success: int = 0
    failed: int = 0
    running: int = 0
    has_parity_runs: bool = False
    phase_progress_last_modified: datetime | None = None
    run_context_last_modified: datetime | None = None
    parity_runs_last_modified: datetime | None = None


@dataclass(frozen=True)
class SystemProgress:
    total: int
    completed: int
    inprogress: int
    unstarted: int
    completed_shards: tuple[str, ...]
    inprogress_shards: tuple[str, ...]
    unstarted_shards: tuple[str, ...]


@dataclass(frozen=True)
class LatestTimestamps:
    latest_any: datetime | None
    latest_phase_progress: datetime | None
    latest_parity_runs: datetime | None


@dataclass(frozen=True)
class CorruptArtifact:
    shard_id: str
    key: str
    error: str


@dataclass(frozen=True)
class AuditReport:
    run_id: str
    region: str
    bucket: str
    phase: str
    s3_prefix: str
    control_plane_instances: tuple[str, ...]
    worker_instances: tuple[str, ...]
    inprogress_records: tuple[InProgressRecord, ...]
    progress_official: SystemProgress
    progress_worldflux: SystemProgress
    latest: LatestTimestamps
    corrupt_artifacts: tuple[CorruptArtifact, ...]
    warnings: tuple[str, ...]
    stall_log_output: str
    generated_at: datetime


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp_to_utc(value: str) -> datetime | None:
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


def _run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, capture_output=True)


def _run_cli_json(command: list[str]) -> dict[str, Any]:
    result = _run_cli(command)
    if result.returncode != 0:
        raise RuntimeError(f"AWS CLI failed: {' '.join(command)}\n{result.stderr.strip()}")
    payload = result.stdout.strip()
    if not payload:
        return {}
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON output: {' '.join(command)}\n{payload}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object from command: {' '.join(command)}\n{payload}")
    return parsed


def _paginate_aws_list(command_base: list[str], list_key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    token: str | None = None
    seen_tokens: set[str] = set()
    page_count = 0
    while True:
        page_count += 1
        if page_count > 1000:
            raise RuntimeError("pagination exceeded safety page limit (1000)")
        command = list(command_base)
        if token:
            if token in seen_tokens:
                break
            seen_tokens.add(token)
            command.extend(["--next-token", token])
        payload = _run_cli_json(command)
        entries = payload.get(list_key, [])
        if isinstance(entries, list):
            for item in entries:
                if isinstance(item, dict):
                    out.append(item)
        token_raw = payload.get("NextToken")
        token = str(token_raw).strip() if isinstance(token_raw, str) else None
        if not token:
            break
    return out


def _option_value(tokens: list[str], option: str) -> str:
    for idx, token in enumerate(tokens[:-1]):
        if token == option:
            return tokens[idx + 1]
    return ""


def _extract_run_parity_command(
    commands: list[str], target_run_id: str, phase: str
) -> tuple[str, str, str, str] | None:
    run_prefix = f"{target_run_id}_{phase}_shard"
    for line in commands:
        if "run_parity_matrix.py" not in line:
            continue
        try:
            tokens = shlex.split(line, posix=True)
        except ValueError:
            continue
        run_id = _option_value(tokens, "--run-id")
        if not run_id.startswith(run_prefix):
            continue
        shard_id = run_id.replace(run_prefix, "", 1)
        if not shard_id.isdigit():
            continue
        system = _option_value(tokens, "--systems")
        seed_list = _option_value(tokens, "--seed-list")
        return run_id, shard_id, system, seed_list
    return None


def _command_invocation_for_id(region: str, command_id: str) -> list[dict[str, Any]]:
    payload = _run_cli_json(
        [
            "aws",
            "ssm",
            "list-command-invocations",
            "--region",
            region,
            "--command-id",
            command_id,
            "--details",
            "--output",
            "json",
        ]
    )
    rows = payload.get("CommandInvocations", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _discover_inprogress_records(
    *, region: str, target_run_id: str, phase: str
) -> list[InProgressRecord]:
    commands = _paginate_aws_list(
        [
            "aws",
            "ssm",
            "list-commands",
            "--region",
            region,
            "--filters",
            "key=Status,value=InProgress",
            "--max-results",
            "50",
            "--output",
            "json",
        ],
        list_key="Commands",
    )

    records: list[InProgressRecord] = []
    for row in commands:
        command_id = str(row.get("CommandId", "")).strip()
        if not command_id:
            continue
        params = row.get("Parameters", {})
        if not isinstance(params, dict):
            continue
        command_lines = params.get("commands", [])
        if not isinstance(command_lines, list):
            continue
        command_lines_text = [str(item) for item in command_lines]
        extracted = _extract_run_parity_command(
            command_lines_text, target_run_id=target_run_id, phase=phase
        )
        if extracted is None:
            continue
        run_id, shard_id, system, seed_list = extracted
        requested_at = _timestamp_to_utc(str(row.get("RequestedDateTime", "")))
        invocations = _command_invocation_for_id(region, command_id)
        for inv in invocations:
            instance_id = str(inv.get("InstanceId", "")).strip()
            status = str(inv.get("Status", "InProgress")).strip() or "InProgress"
            if not instance_id:
                continue
            records.append(
                InProgressRecord(
                    command_id=command_id,
                    instance_id=instance_id,
                    requested_at=requested_at,
                    run_id=run_id,
                    shard_id=shard_id,
                    system=system,
                    seed_list=seed_list,
                    status=status,
                )
            )
    return sorted(records, key=lambda item: (item.shard_id, item.command_id, item.instance_id))


def _list_s3_objects(region: str, bucket: str, prefix: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    token: str | None = None
    seen_tokens: set[str] = set()
    page_count = 0
    while True:
        page_count += 1
        if page_count > 1000:
            raise RuntimeError("s3 pagination exceeded safety page limit (1000)")
        command = [
            "aws",
            "s3api",
            "list-objects-v2",
            "--region",
            region,
            "--bucket",
            bucket,
            "--prefix",
            prefix,
            "--max-keys",
            "1000",
            "--output",
            "json",
        ]
        if token:
            if token in seen_tokens:
                break
            seen_tokens.add(token)
            command.extend(["--continuation-token", token])
        payload = _run_cli_json(command)
        rows = payload.get("Contents", [])
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    out.append(row)
        token_raw = payload.get("NextContinuationToken")
        token = str(token_raw).strip() if isinstance(token_raw, str) else None
        if not token:
            break
    return out


def _s3_read_json(region: str, bucket: str, key: str) -> dict[str, Any]:
    result = _run_cli(
        [
            "aws",
            "s3",
            "cp",
            f"s3://{bucket}/{key}",
            "-",
            "--region",
            region,
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to read s3://{bucket}/{key}: {result.stderr.strip()}")
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in s3://{bucket}/{key}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object in s3://{bucket}/{key}")
    return payload


def _collect_shard_states(
    *,
    region: str,
    bucket: str,
    run_id: str,
    phase: str,
) -> tuple[dict[str, ShardState], LatestTimestamps, list[CorruptArtifact]]:
    prefix = f"{run_id}/{phase}/shards/"
    rows = _list_s3_objects(region=region, bucket=bucket, prefix=prefix)
    shard_states: dict[str, ShardState] = {}
    latest_any: datetime | None = None
    latest_phase: datetime | None = None
    latest_runs: datetime | None = None
    corrupt_artifacts: list[CorruptArtifact] = []

    key_re = re.compile(rf"^{re.escape(prefix)}(?P<shard>[^/]+)/(?P<name>[^/]+)$")

    for row in rows:
        key = str(row.get("Key", "")).strip()
        match = key_re.match(key)
        if match is None:
            continue
        shard_id = match.group("shard")
        name = match.group("name")
        modified = _timestamp_to_utc(str(row.get("LastModified", "")))
        state = shard_states.setdefault(shard_id, ShardState(shard_id=shard_id))

        if modified is not None and (latest_any is None or modified > latest_any):
            latest_any = modified
        if name == "phase_progress.json":
            state.phase_progress_last_modified = modified
            if modified is not None and (latest_phase is None or modified > latest_phase):
                latest_phase = modified
        if name == "run_context.json":
            state.run_context_last_modified = modified
        if name == "parity_runs.jsonl":
            state.has_parity_runs = True
            state.parity_runs_last_modified = modified
            if modified is not None and (latest_runs is None or modified > latest_runs):
                latest_runs = modified

    for shard_id, state in list(shard_states.items()):
        run_context_key = f"{prefix}{shard_id}/run_context.json"
        if state.run_context_last_modified is not None:
            try:
                payload = _s3_read_json(region=region, bucket=bucket, key=run_context_key)
                systems = payload.get("systems", [])
                if isinstance(systems, list) and systems:
                    state.system = str(systems[0]).strip() or state.system
            except RuntimeError as exc:
                corrupt_artifacts.append(
                    CorruptArtifact(
                        shard_id=shard_id,
                        key=run_context_key,
                        error=str(exc),
                    )
                )

        phase_key = f"{prefix}{shard_id}/phase_progress.json"
        if state.phase_progress_last_modified is not None:
            try:
                payload = _s3_read_json(region=region, bucket=bucket, key=phase_key)
                state.expected = int(payload.get("expected", 0) or 0)
                state.started = int(payload.get("started", 0) or 0)
                state.success = int(payload.get("success", 0) or 0)
                state.failed = int(payload.get("failed", 0) or 0)
                state.running = int(payload.get("running", 0) or 0)
            except RuntimeError as exc:
                corrupt_artifacts.append(
                    CorruptArtifact(
                        shard_id=shard_id,
                        key=phase_key,
                        error=str(exc),
                    )
                )

    return (
        shard_states,
        LatestTimestamps(
            latest_any=latest_any,
            latest_phase_progress=latest_phase,
            latest_parity_runs=latest_runs,
        ),
        corrupt_artifacts,
    )


def _progress_for_system(
    shard_states: dict[str, ShardState],
    inprogress_records: list[InProgressRecord],
    *,
    system: str,
) -> SystemProgress:
    by_system = {
        shard_id: state
        for shard_id, state in shard_states.items()
        if state.system.lower() == system
    }
    inprogress_shards = {row.shard_id for row in inprogress_records if row.system.lower() == system}

    completed_ids: list[str] = []
    inprogress_ids: list[str] = []
    unstarted_ids: list[str] = []

    for shard_id, state in sorted(by_system.items(), key=lambda item: int(item[0])):
        if state.success >= 1 and state.has_parity_runs:
            completed_ids.append(shard_id)
            continue
        if shard_id in inprogress_shards:
            inprogress_ids.append(shard_id)
            continue
        unstarted_ids.append(shard_id)

    return SystemProgress(
        total=len(by_system),
        completed=len(completed_ids),
        inprogress=len(inprogress_ids),
        unstarted=len(unstarted_ids),
        completed_shards=tuple(completed_ids),
        inprogress_shards=tuple(inprogress_ids),
        unstarted_shards=tuple(unstarted_ids),
    )


def _discover_control_plane_instances(
    *, region: str, run_id: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    payload = _run_cli_json(
        [
            "aws",
            "ec2",
            "describe-instances",
            "--region",
            region,
            "--filters",
            f"Name=tag:ParityRunTag,Values={run_id}",
            "Name=instance-state-name,Values=pending,running,stopping,stopped",
            "--output",
            "json",
        ]
    )
    reservations = payload.get("Reservations", [])
    control_plane_ids: list[str] = []
    worker_ids_from_tags: set[str] = set()
    if isinstance(reservations, list):
        for reservation in reservations:
            if not isinstance(reservation, dict):
                continue
            instances = reservation.get("Instances", [])
            if not isinstance(instances, list):
                continue
            for instance in instances:
                if not isinstance(instance, dict):
                    continue
                instance_id = str(instance.get("InstanceId", "")).strip()
                if instance_id:
                    control_plane_ids.append(instance_id)
                tags = instance.get("Tags", [])
                if not isinstance(tags, list):
                    continue
                for tag in tags:
                    if not isinstance(tag, dict):
                        continue
                    if str(tag.get("Key", "")).strip() != "ParityGpuInstanceId":
                        continue
                    value = str(tag.get("Value", "")).strip()
                    if value:
                        worker_ids_from_tags.add(value)
    return tuple(sorted(set(control_plane_ids))), tuple(sorted(worker_ids_from_tags))


def _warnings_for_stall(
    *,
    now: datetime,
    inprogress_records: list[InProgressRecord],
    shard_states: dict[str, ShardState],
    progress_official: SystemProgress,
    progress_worldflux: SystemProgress,
    stale_inprogress_hours: float,
    stale_progress_minutes: float,
) -> list[str]:
    warnings: list[str] = []
    threshold_seconds_ssm = max(0.0, stale_inprogress_hours * 3600.0)
    threshold_seconds_progress = max(0.0, stale_progress_minutes * 60.0)

    for record in sorted(
        inprogress_records, key=lambda row: (row.system, int(row.shard_id), row.command_id)
    ):
        if record.requested_at is None:
            continue
        age_sec = max(0.0, (now - record.requested_at).total_seconds())
        if age_sec >= threshold_seconds_ssm:
            warnings.append(
                "SSM InProgress stale: "
                f"system={record.system} shard={record.shard_id} seed={record.seed_list} "
                f"age_hours={age_sec / 3600.0:.2f} command_id={record.command_id}"
            )

    active_shards = set(progress_official.inprogress_shards) | set(
        progress_worldflux.inprogress_shards
    )
    for shard_id in sorted(active_shards, key=int):
        state = shard_states.get(shard_id)
        if state is None:
            continue
        if state.phase_progress_last_modified is None:
            warnings.append(
                "phase_progress missing: "
                f"system={state.system} shard={state.shard_id} while SSM reports InProgress"
            )
            continue
        age_sec = max(0.0, (now - state.phase_progress_last_modified).total_seconds())
        if age_sec >= threshold_seconds_progress:
            warnings.append(
                "phase_progress stale: "
                f"system={state.system} shard={state.shard_id} age_minutes={age_sec / 60.0:.1f}"
            )
    return warnings


def _tail_stall_logs(
    *,
    region: str,
    control_plane_instance_id: str,
    run_id: str,
    tail_lines: int,
) -> str:
    parameters = json.dumps(
        {
            "commands": [
                f"tail -n {tail_lines} /var/log/orchestrator-{run_id}.log || true",
                "echo '---'",
                f"tail -n {tail_lines} /var/log/orchestrator-seed_system-v2.log || true",
            ]
        }
    )
    send = _run_cli(
        [
            "aws",
            "ssm",
            "send-command",
            "--region",
            region,
            "--instance-ids",
            control_plane_instance_id,
            "--document-name",
            "AWS-RunShellScript",
            "--parameters",
            parameters,
            "--query",
            "Command.CommandId",
            "--output",
            "text",
        ]
    )
    if send.returncode != 0:
        return f"Failed to request stall logs: {send.stderr.strip()}"
    command_id = send.stdout.strip()
    if not command_id:
        return "Failed to request stall logs: missing command id"

    for _ in range(15):
        inv = _run_cli_json(
            [
                "aws",
                "ssm",
                "get-command-invocation",
                "--region",
                region,
                "--command-id",
                command_id,
                "--instance-id",
                control_plane_instance_id,
                "--output",
                "json",
            ]
        )
        status = str(inv.get("Status", "")).strip()
        if status in TERMINAL_SSM_STATUSES:
            stdout = str(inv.get("StandardOutputContent", ""))
            stderr = str(inv.get("StandardErrorContent", ""))
            if stderr.strip():
                return stdout + ("\n" if stdout and not stdout.endswith("\n") else "") + stderr
            return stdout
        time.sleep(2)
    return f"Failed to fetch stall logs in time for command_id={command_id}"


def _fmt_dt(value: datetime | None) -> str:
    if value is None:
        return "-"
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _fmt_shards(values: tuple[str, ...]) -> str:
    if not values:
        return "-"
    return ",".join(values)


def _render_report(report: AuditReport) -> str:
    lines: list[str] = []
    lines.append("[1] 対象Run情報")
    lines.append(f"- run_id: {report.run_id}")
    lines.append(f"- region: {report.region}")
    lines.append(f"- bucket: {report.bucket}")
    lines.append(f"- phase: {report.phase}")
    lines.append(f"- s3_prefix: {report.s3_prefix}")
    lines.append(
        "- control_plane_instances: "
        + (",".join(report.control_plane_instances) if report.control_plane_instances else "-")
    )
    lines.append(
        "- worker_instances: "
        + (",".join(report.worker_instances) if report.worker_instances else "-")
    )
    lines.append(f"- observed_inprogress_commands: {len(report.inprogress_records)}")

    lines.append("")
    lines.append("[2] official 進捗")
    lines.append(f"- total: {report.progress_official.total}")
    lines.append(
        f"- completed: {report.progress_official.completed} "
        f"(shards={_fmt_shards(report.progress_official.completed_shards)})"
    )
    lines.append(
        f"- inprogress: {report.progress_official.inprogress} "
        f"(shards={_fmt_shards(report.progress_official.inprogress_shards)})"
    )
    lines.append(
        f"- unstarted: {report.progress_official.unstarted} "
        f"(shards={_fmt_shards(report.progress_official.unstarted_shards)})"
    )

    lines.append("")
    lines.append("[3] worldflux 進捗")
    lines.append(f"- total: {report.progress_worldflux.total}")
    lines.append(
        f"- completed: {report.progress_worldflux.completed} "
        f"(shards={_fmt_shards(report.progress_worldflux.completed_shards)})"
    )
    lines.append(
        f"- inprogress: {report.progress_worldflux.inprogress} "
        f"(shards={_fmt_shards(report.progress_worldflux.inprogress_shards)})"
    )
    lines.append(
        f"- unstarted: {report.progress_worldflux.unstarted} "
        f"(shards={_fmt_shards(report.progress_worldflux.unstarted_shards)})"
    )

    lines.append("")
    lines.append("[4] 直近更新時刻（S3 LastModified, UTC）")
    lines.append(f"- latest_any: {_fmt_dt(report.latest.latest_any)}")
    lines.append(f"- latest_phase_progress: {_fmt_dt(report.latest.latest_phase_progress)}")
    lines.append(f"- latest_parity_runs: {_fmt_dt(report.latest.latest_parity_runs)}")
    lines.append(f"- generated_at: {_fmt_dt(report.generated_at)}")

    lines.append("")
    lines.append("[5] 停滞警告")
    if not report.warnings:
        lines.append("- none")
    else:
        for warning in report.warnings:
            lines.append(f"- {warning}")

    if report.stall_log_output.strip():
        lines.append("")
        lines.append("[stall-log-tail]")
        lines.append(report.stall_log_output.rstrip())
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=str, default="cloud_proof_20260221T220330Z")
    parser.add_argument("--region", type=str, default="us-west-2")
    parser.add_argument("--bucket", type=str, default="worldflux-parity")
    parser.add_argument("--phase", type=str, default="pilot")
    parser.add_argument("--stale-inprogress-hours", type=float, default=4.0)
    parser.add_argument("--stale-progress-minutes", type=float, default=30.0)
    parser.add_argument(
        "--fetch-stall-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fetch control-plane log tails via read-only SSM command when warnings are detected.",
    )
    parser.add_argument("--stall-log-lines", type=int, default=80)
    parser.add_argument(
        "--output-json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print report payload as JSON instead of the 5-block text report.",
    )
    return parser.parse_args()


def run_audit(args: argparse.Namespace) -> AuditReport:
    now = _utc_now()
    control_plane_ids, worker_ids_from_tags = _discover_control_plane_instances(
        region=args.region,
        run_id=args.run_id,
    )
    inprogress_records = _discover_inprogress_records(
        region=args.region, target_run_id=args.run_id, phase=args.phase
    )
    shard_states, latest, corrupt_artifacts = _collect_shard_states(
        region=args.region,
        bucket=args.bucket,
        run_id=args.run_id,
        phase=args.phase,
    )

    for row in inprogress_records:
        state = shard_states.get(row.shard_id)
        if state is None:
            state = ShardState(shard_id=row.shard_id, system=row.system)
            shard_states[row.shard_id] = state
        if state.system == "unknown" and row.system:
            state.system = row.system

    progress_official = _progress_for_system(
        shard_states=shard_states,
        inprogress_records=inprogress_records,
        system="official",
    )
    progress_worldflux = _progress_for_system(
        shard_states=shard_states,
        inprogress_records=inprogress_records,
        system="worldflux",
    )

    warnings = _warnings_for_stall(
        now=now,
        inprogress_records=inprogress_records,
        shard_states=shard_states,
        progress_official=progress_official,
        progress_worldflux=progress_worldflux,
        stale_inprogress_hours=float(args.stale_inprogress_hours),
        stale_progress_minutes=float(args.stale_progress_minutes),
    )
    for row in corrupt_artifacts:
        warnings.append(
            "corrupt_artifact: " f"shard={row.shard_id} key={row.key} error={row.error}"
        )

    worker_instances = set(worker_ids_from_tags)
    worker_instances.update(row.instance_id for row in inprogress_records if row.instance_id)

    stall_log_output = ""
    if warnings and args.fetch_stall_logs and control_plane_ids:
        stall_log_output = _tail_stall_logs(
            region=args.region,
            control_plane_instance_id=control_plane_ids[0],
            run_id=args.run_id,
            tail_lines=max(1, int(args.stall_log_lines)),
        )

    return AuditReport(
        run_id=args.run_id,
        region=args.region,
        bucket=args.bucket,
        phase=args.phase,
        s3_prefix=f"s3://{args.bucket}/{args.run_id}/{args.phase}/",
        control_plane_instances=tuple(sorted(control_plane_ids)),
        worker_instances=tuple(sorted(worker_instances)),
        inprogress_records=tuple(inprogress_records),
        progress_official=progress_official,
        progress_worldflux=progress_worldflux,
        latest=latest,
        corrupt_artifacts=tuple(corrupt_artifacts),
        warnings=tuple(warnings),
        stall_log_output=stall_log_output,
        generated_at=now,
    )


def _json_report(report: AuditReport) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "region": report.region,
        "bucket": report.bucket,
        "phase": report.phase,
        "s3_prefix": report.s3_prefix,
        "control_plane_instances": list(report.control_plane_instances),
        "worker_instances": list(report.worker_instances),
        "inprogress_records": [
            {
                "command_id": row.command_id,
                "instance_id": row.instance_id,
                "requested_at": _fmt_dt(row.requested_at),
                "run_id": row.run_id,
                "shard_id": row.shard_id,
                "system": row.system,
                "seed_list": row.seed_list,
                "status": row.status,
            }
            for row in report.inprogress_records
        ],
        "progress": {
            "official": {
                "total": report.progress_official.total,
                "completed": report.progress_official.completed,
                "inprogress": report.progress_official.inprogress,
                "unstarted": report.progress_official.unstarted,
                "completed_shards": list(report.progress_official.completed_shards),
                "inprogress_shards": list(report.progress_official.inprogress_shards),
                "unstarted_shards": list(report.progress_official.unstarted_shards),
            },
            "worldflux": {
                "total": report.progress_worldflux.total,
                "completed": report.progress_worldflux.completed,
                "inprogress": report.progress_worldflux.inprogress,
                "unstarted": report.progress_worldflux.unstarted,
                "completed_shards": list(report.progress_worldflux.completed_shards),
                "inprogress_shards": list(report.progress_worldflux.inprogress_shards),
                "unstarted_shards": list(report.progress_worldflux.unstarted_shards),
            },
        },
        "latest": {
            "latest_any": _fmt_dt(report.latest.latest_any),
            "latest_phase_progress": _fmt_dt(report.latest.latest_phase_progress),
            "latest_parity_runs": _fmt_dt(report.latest.latest_parity_runs),
            "generated_at": _fmt_dt(report.generated_at),
        },
        "warnings": list(report.warnings),
        "corrupt_artifacts": [
            {
                "shard_id": row.shard_id,
                "key": row.key,
                "error": row.error,
            }
            for row in report.corrupt_artifacts
        ],
        "stall_log_output": report.stall_log_output,
    }


def main() -> int:
    args = _parse_args()
    try:
        report = run_audit(args)
    except RuntimeError as exc:
        print(f"[aws-progress-audit] ERROR: {exc}", file=sys.stderr)
        return 1

    if args.output_json:
        print(json.dumps(_json_report(report), indent=2, sort_keys=True))
    else:
        print(_render_report(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
