"""Tests for GPU slot resolution behavior in AWS parity orchestrator."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


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


def test_resolve_instance_gpu_counts_and_auto_slot_caps(monkeypatch) -> None:
    mod = _load_module()

    def fake_run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["aws", "ec2", "describe-instances"]:
            payload = {
                "Reservations": [
                    {
                        "Instances": [
                            {"InstanceId": "i-a", "InstanceType": "g6.2xlarge"},
                            {"InstanceId": "i-b", "InstanceType": "p4d.24xlarge"},
                        ]
                    }
                ]
            }
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

        if command[:3] == ["aws", "ec2", "describe-instance-types"]:
            payload = {
                "InstanceTypes": [
                    {"InstanceType": "g6.2xlarge", "GpuInfo": {"Gpus": [{"Count": 1}]}},
                    {"InstanceType": "p4d.24xlarge", "GpuInfo": {"Gpus": [{"Count": 8}]}},
                ]
            }
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(mod, "_run_cli", fake_run_cli)

    gpu_counts, instance_types = mod._resolve_instance_gpu_counts(
        region="us-west-2",
        instance_ids=["i-a", "i-b"],
    )
    assert gpu_counts == {"i-a": 1, "i-b": 8}
    assert instance_types == {"i-a": "g6.2xlarge", "i-b": "p4d.24xlarge"}

    slot_caps, mismatch_events = mod._resolve_instance_slot_caps(
        instance_ids=["i-a", "i-b"],
        instance_gpu_counts=gpu_counts,
        requested_slots_raw="auto",
        mismatch_policy="fail_fast",
    )
    assert slot_caps == {"i-a": 1, "i-b": 8}
    assert mismatch_events == []


def test_resolve_instance_slot_caps_fail_fast_on_mismatch() -> None:
    mod = _load_module()

    with pytest.raises(RuntimeError, match="Requested GPU slots exceed detected GPU devices"):
        mod._resolve_instance_slot_caps(
            instance_ids=["i-a"],
            instance_gpu_counts={"i-a": 1},
            requested_slots_raw="8",
            mismatch_policy="fail_fast",
        )


def test_resolve_instance_slot_caps_clamps_on_mismatch() -> None:
    mod = _load_module()

    slot_caps, mismatch_events = mod._resolve_instance_slot_caps(
        instance_ids=["i-a"],
        instance_gpu_counts={"i-a": 1},
        requested_slots_raw="8",
        mismatch_policy="clamp",
    )
    assert slot_caps == {"i-a": 1}
    assert len(mismatch_events) == 1
    assert mismatch_events[0]["instance_id"] == "i-a"
    assert mismatch_events[0]["requested_slots"] == 8
    assert mismatch_events[0]["detected_slots"] == 1
