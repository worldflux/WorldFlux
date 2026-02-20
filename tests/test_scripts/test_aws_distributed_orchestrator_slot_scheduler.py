"""Tests for slot-level scheduler behavior in AWS parity orchestrator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "parity"
        / "aws_distributed_orchestrator.py"
    )
    spec = importlib.util.spec_from_file_location("aws_distributed_orchestrator", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load aws_distributed_orchestrator")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_slot_scheduler_never_runs_two_jobs_on_same_slot(monkeypatch) -> None:
    mod = _load_module()

    args = SimpleNamespace(region="us-west-2", poll_interval_sec=0)
    shards = [
        mod.ShardPlan(
            shard_id=0,
            instance_id="i-a",
            task_ids=("atari100k_pong",),
            shard_run_id="run_shard00",
            estimated_cost=1.0,
            estimated_duration_sec=1.0,
            seed_values=(0,),
            systems=("official",),
            gpu_slot=0,
        ),
        mod.ShardPlan(
            shard_id=1,
            instance_id="i-a",
            task_ids=("atari100k_pong",),
            shard_run_id="run_shard01",
            estimated_cost=1.0,
            estimated_duration_sec=1.0,
            seed_values=(1,),
            systems=("official",),
            gpu_slot=0,
        ),
        mod.ShardPlan(
            shard_id=2,
            instance_id="i-a",
            task_ids=("dog-run",),
            shard_run_id="run_shard02",
            estimated_cost=1.0,
            estimated_duration_sec=1.0,
            seed_values=(0,),
            systems=("worldflux",),
            gpu_slot=1,
        ),
    ]

    submitted: list[int] = []

    def fake_submit_shard(**kwargs):
        shard = kwargs["shard"]
        submitted.append(int(shard.shard_id))
        return {
            "shard_id": shard.shard_id,
            "instance_id": shard.instance_id,
            "task_count": len(shard.task_ids),
            "task_ids": list(shard.task_ids),
            "seed_values": list(shard.seed_values),
            "systems": list(shard.systems),
            "gpu_slot": shard.gpu_slot,
            "run_id": shard.shard_run_id,
            "command_id": f"cmd-{shard.shard_id}",
            "estimated_cost": shard.estimated_cost,
            "predicted_duration_sec": shard.estimated_duration_sec,
            "timeout_risk": "low",
            "submitted_at": "2026-01-01T00:00:00+00:00",
        }

    status_plan = {
        "cmd-0": ["Success"],
        "cmd-1": ["Success"],
        "cmd-2": ["InProgress", "Success"],
    }
    status_index = {key: 0 for key in status_plan}

    def fake_get_command_invocation(*, command_id: str, **_kwargs):
        idx = status_index[command_id]
        values = status_plan[command_id]
        status = values[min(idx, len(values) - 1)]
        status_index[command_id] = idx + 1
        return {
            "Status": status,
            "ResponseCode": 0 if status == "Success" else -1,
            "StatusDetails": status,
            "StandardOutputContent": "",
            "StandardErrorContent": "",
        }

    monkeypatch.setattr(mod, "_submit_shard", fake_submit_shard)
    monkeypatch.setattr(mod, "_get_command_invocation", fake_get_command_invocation)
    monkeypatch.setattr(mod.time, "sleep", lambda _sec: None)
    monkeypatch.setattr(mod, "_timestamp", lambda: "2026-01-01T00:00:00+00:00")

    submissions, results = mod._dispatch_with_slot_scheduler(
        args=args,
        shards=shards,
        manifest_rel="scripts/parity/manifests/official_vs_worldflux_v1.yaml",
        run_id="run_demo",
        s3_prefix="s3://bucket/run_demo",
    )

    # First wave should submit one shard per slot: shard0(slot0) and shard2(slot1).
    assert submitted[:2] == [0, 2]
    # shard1 (same slot as shard0) is submitted only after shard0 completes.
    assert submitted[2] == 1
    assert len(submissions) == 3
    assert len(results) == 3
    assert all(str(row.get("status")) == "Success" for row in results)
