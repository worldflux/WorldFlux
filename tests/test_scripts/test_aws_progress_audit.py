"""Tests for AWS parity progress audit script."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "parity" / "aws_progress_audit.py"
    )
    spec = importlib.util.spec_from_file_location("aws_progress_audit", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load aws_progress_audit")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_extract_run_parity_command() -> None:
    mod = _load_module()
    lines = [
        "echo pre",
        (
            "python3 scripts/parity/run_parity_matrix.py --manifest x --run-id "
            "cloud_proof_20260221T220330Z_pilot_shard15 --systems worldflux --seed-list 5"
        ),
    ]
    out = mod._extract_run_parity_command(
        lines, target_run_id="cloud_proof_20260221T220330Z", phase="pilot"
    )
    assert out is not None
    run_id, shard_id, system, seed_list = out
    assert run_id == "cloud_proof_20260221T220330Z_pilot_shard15"
    assert shard_id == "15"
    assert system == "worldflux"
    assert seed_list == "5"


def test_progress_for_system_counts_completed_inprogress_unstarted() -> None:
    mod = _load_module()
    states = {
        "00": mod.ShardState(
            shard_id="00",
            system="official",
            success=1,
            has_parity_runs=True,
        ),
        "01": mod.ShardState(
            shard_id="01",
            system="official",
            success=0,
            has_parity_runs=False,
        ),
        "02": mod.ShardState(
            shard_id="02",
            system="official",
            success=0,
            has_parity_runs=False,
        ),
    }
    inprogress = [
        mod.InProgressRecord(
            command_id="c-1",
            instance_id="i-1",
            requested_at=datetime(2026, 2, 22, 1, 0, tzinfo=timezone.utc),
            run_id="cloud_proof_20260221T220330Z_pilot_shard01",
            shard_id="01",
            system="official",
            seed_list="1",
            status="InProgress",
        )
    ]
    progress = mod._progress_for_system(states, inprogress, system="official")
    assert progress.total == 3
    assert progress.completed == 1
    assert progress.inprogress == 1
    assert progress.unstarted == 1
    assert progress.completed_shards == ("00",)
    assert progress.inprogress_shards == ("01",)
    assert progress.unstarted_shards == ("02",)


def test_warnings_for_stall_detects_old_ssm_and_phase_progress() -> None:
    mod = _load_module()
    now = datetime(2026, 2, 22, 2, 0, tzinfo=timezone.utc)
    inprogress = [
        mod.InProgressRecord(
            command_id="cmd-1",
            instance_id="i-1",
            requested_at=datetime(2026, 2, 21, 20, 0, tzinfo=timezone.utc),
            run_id="cloud_proof_20260221T220330Z_pilot_shard10",
            shard_id="10",
            system="worldflux",
            seed_list="0",
            status="InProgress",
        )
    ]
    states = {
        "10": mod.ShardState(
            shard_id="10",
            system="worldflux",
            phase_progress_last_modified=datetime(2026, 2, 22, 1, 0, tzinfo=timezone.utc),
        )
    }
    progress_official = mod.SystemProgress(
        total=0,
        completed=0,
        inprogress=0,
        unstarted=0,
        completed_shards=(),
        inprogress_shards=(),
        unstarted_shards=(),
    )
    progress_worldflux = mod.SystemProgress(
        total=1,
        completed=0,
        inprogress=1,
        unstarted=0,
        completed_shards=(),
        inprogress_shards=("10",),
        unstarted_shards=(),
    )
    warnings = mod._warnings_for_stall(
        now=now,
        inprogress_records=inprogress,
        shard_states=states,
        progress_official=progress_official,
        progress_worldflux=progress_worldflux,
        stale_inprogress_hours=4.0,
        stale_progress_minutes=30.0,
    )
    assert any("SSM InProgress stale" in item for item in warnings)
    assert any("phase_progress stale" in item for item in warnings)


def test_run_audit_happy_path_with_mocked_aws(monkeypatch) -> None:
    mod = _load_module()

    now = datetime(2026, 2, 22, 1, 38, 6, tzinfo=timezone.utc)
    monkeypatch.setattr(mod, "_utc_now", lambda: now)

    s3_bucket = "worldflux-parity"
    run_id = "cloud_proof_20260221T220330Z"
    phase = "pilot"
    prefix = f"{run_id}/{phase}/shards/"

    s3_objects = [
        {
            "Key": f"{prefix}00/run_context.json",
            "LastModified": "2026-02-22T00:56:39+00:00",
        },
        {
            "Key": f"{prefix}00/phase_progress.json",
            "LastModified": "2026-02-22T00:56:41+00:00",
        },
        {
            "Key": f"{prefix}00/parity_runs.jsonl",
            "LastModified": "2026-02-22T00:56:38+00:00",
        },
        {
            "Key": f"{prefix}10/run_context.json",
            "LastModified": "2026-02-22T01:35:31+00:00",
        },
        {
            "Key": f"{prefix}10/phase_progress.json",
            "LastModified": "2026-02-22T01:35:35+00:00",
        },
    ]

    command_payload = {
        "Commands": [
            {
                "CommandId": "cmd-inprogress",
                "RequestedDateTime": "2026-02-21T17:15:26-08:00",
                "Parameters": {
                    "commands": [
                        (
                            "python3 scripts/parity/run_parity_matrix.py --manifest x "
                            "--run-id cloud_proof_20260221T220330Z_pilot_shard10 "
                            "--systems worldflux --seed-list 0"
                        )
                    ]
                },
            }
        ]
    }

    command_invocations = {
        "CommandInvocations": [
            {
                "CommandId": "cmd-inprogress",
                "InstanceId": "i-worker",
                "Status": "InProgress",
            }
        ]
    }

    run_context_00 = {"systems": ["official"], "seeds": [0]}
    run_context_10 = {"systems": ["worldflux"], "seeds": [0]}
    phase_progress_00 = {
        "expected": 1,
        "started": 1,
        "success": 1,
        "failed": 0,
        "running": 0,
    }
    phase_progress_10 = {
        "expected": 1,
        "started": 1,
        "success": 0,
        "failed": 0,
        "running": 1,
    }

    control_plane_instances = {
        "Reservations": [
            {
                "Instances": [
                    {
                        "InstanceId": "i-control",
                        "Tags": [
                            {"Key": "ParityRunTag", "Value": run_id},
                            {"Key": "ParityGpuInstanceId", "Value": "i-worker"},
                        ],
                    }
                ]
            }
        ]
    }

    def fake_run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
        cmd = " ".join(command)
        if "ec2 describe-instances" in cmd and "ParityRunTag" in cmd:
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps(control_plane_instances), stderr=""
            )
        if "ssm list-commands" in cmd and "InProgress" in cmd:
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps(command_payload), stderr=""
            )
        if "ssm list-command-invocations" in cmd and "--command-id cmd-inprogress" in cmd:
            return subprocess.CompletedProcess(
                command, 0, stdout=json.dumps(command_invocations), stderr=""
            )
        if "s3api list-objects-v2" in cmd:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"Contents": s3_objects}),
                stderr="",
            )
        if command[:3] == ["aws", "s3", "cp"] and command[-1] == "us-west-2":
            key = command[3].replace(f"s3://{s3_bucket}/", "")
            if key.endswith("00/run_context.json"):
                payload = run_context_00
            elif key.endswith("10/run_context.json"):
                payload = run_context_10
            elif key.endswith("00/phase_progress.json"):
                payload = phase_progress_00
            elif key.endswith("10/phase_progress.json"):
                payload = phase_progress_10
            else:
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="missing")
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(mod, "_run_cli", fake_run_cli)

    args = Namespace(
        run_id=run_id,
        region="us-west-2",
        bucket=s3_bucket,
        phase=phase,
        stale_inprogress_hours=4.0,
        stale_progress_minutes=30.0,
        fetch_stall_logs=False,
        stall_log_lines=80,
        output_json=False,
    )
    report = mod.run_audit(args)
    assert report.progress_official.total == 1
    assert report.progress_official.completed == 1
    assert report.progress_worldflux.total == 1
    assert report.progress_worldflux.inprogress == 1
    assert "i-control" in report.control_plane_instances
    assert "i-worker" in report.worker_instances
    assert report.latest.latest_phase_progress is not None

    rendered = mod._render_report(report)
    assert "[1] 対象Run情報" in rendered
    assert "[2] official 進捗" in rendered
    assert "[3] worldflux 進捗" in rendered
    assert "[4] 直近更新時刻（S3 LastModified, UTC）" in rendered
    assert "[5] 停滞警告" in rendered
