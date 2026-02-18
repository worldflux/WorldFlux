#!/usr/bin/env python3
"""Run official-vs-WorldFlux parity matrix across AWS SSM shard workers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ShardPlan:
    shard_id: int
    instance_id: str
    task_ids: tuple[str, ...]
    shard_run_id: str


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


def _manifest_tasks(path: Path) -> list[str]:
    payload = _load_manifest(path)
    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        raise RuntimeError("manifest.tasks must be list")
    out: list[str] = []
    for item in tasks:
        if not isinstance(item, dict):
            continue
        task_id = item.get("task_id")
        if isinstance(task_id, str) and task_id.strip():
            out.append(task_id.strip())
    if not out:
        raise RuntimeError("manifest has no tasks")
    return sorted(set(out))


def _parse_instance_ids(raw: str) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if not values:
        raise RuntimeError("--instance-ids must include at least one ID")
    return values


def _chunk_round_robin(values: list[str], n: int) -> list[list[str]]:
    chunks: list[list[str]] = [[] for _ in range(max(1, n))]
    for index, value in enumerate(values):
        chunks[index % len(chunks)].append(value)
    return chunks


def _build_shards(run_id: str, tasks: list[str], instances: list[str]) -> list[ShardPlan]:
    chunks = _chunk_round_robin(tasks, len(instances))
    plans: list[ShardPlan] = []
    for idx, instance in enumerate(instances):
        chunk = tuple(chunks[idx])
        if not chunk:
            continue
        plans.append(
            ShardPlan(
                shard_id=idx,
                instance_id=instance,
                task_ids=chunk,
                shard_run_id=f"{run_id}_shard{idx:02d}",
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
) -> list[str]:
    shard_dir = (
        f"{workspace_root.rstrip('/')}/{run_id}/{shard.instance_id}/shard_{shard.shard_id:02d}"
    )
    wf_sha_expr = worldflux_sha.strip() or "$(git rev-parse HEAD)"
    task_filter = ",".join(shard.task_ids)
    remote_run_root = f"reports/parity/{shard.shard_run_id}"
    shard_prefix = f"{s3_prefix.rstrip('/')}/shards/{shard.shard_id:02d}"

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
        (
            "python3 scripts/parity/run_parity_matrix.py "
            f"--manifest {manifest_rel} "
            f"--run-id {shard.shard_run_id} "
            "--output-dir reports/parity "
            f"--device {device} "
            f"--max-retries {max_retries} "
            f"--task-filter '{task_filter}'"
        ),
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
    ]
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


def _run_local_script(script: Path, args: list[str]) -> None:
    command = [sys.executable, str(script), *args]
    result = _run_cli(command)
    if result.returncode != 0:
        raise RuntimeError(f"script failed: {' '.join(command)}\n{result.stderr}\n{result.stdout}")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    args = _parse_args()

    tasks = _manifest_tasks(args.manifest)
    instances = _parse_instance_ids(args.instance_ids)
    shards = _build_shards(args.run_id, tasks, instances)
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
    ]
    if seed_plan_path is not None:
        coverage_args.extend(["--seed-plan", str(seed_plan_path)])
    if run_context_path is not None:
        coverage_args.extend(["--run-context", str(run_context_path)])
    _run_local_script(coverage_script, coverage_args)

    equivalence_report = local_root / "equivalence_report.json"
    equivalence_md = local_root / "equivalence_report.md"
    validity_report = local_root / "validity_report.json"

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

    final_prefix = f"{args.s3_prefix.rstrip('/')}/final"
    for artifact in (
        merged_runs,
        coverage_report,
        validity_report,
        equivalence_report,
        equivalence_md,
        merge_summary,
    ):
        cli = [
            "aws",
            "s3",
            "cp",
            str(artifact),
            f"{final_prefix}/{artifact.name}",
            "--region",
            args.region,
        ]
        result = _run_cli(cli)
        if result.returncode != 0:
            raise RuntimeError(f"failed uploading {artifact} to {final_prefix}: {result.stderr}")

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
        "artifacts": {
            "merged_runs": str(merged_runs),
            "coverage_report": str(coverage_report),
            "validity_report": str(validity_report),
            "equivalence_report": str(equivalence_report),
            "equivalence_markdown": str(equivalence_md),
            "merge_summary": str(merge_summary),
        },
        "s3_final_prefix": final_prefix,
    }

    summary_path = local_root / "orchestrator_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
