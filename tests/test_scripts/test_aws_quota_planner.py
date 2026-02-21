"""Tests for AWS quota planner."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

_SCRIPT = (
    Path(__file__).resolve().parent.parent.parent / "scripts" / "parity" / "aws_quota_planner.py"
)
_spec = importlib.util.spec_from_file_location("aws_quota_planner", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["aws_quota_planner"] = _mod
_spec.loader.exec_module(_mod)

FleetPlan = _mod.FleetPlan
detect_gpu_quota = _mod.detect_gpu_quota
compute_optimal_fleet = _mod.compute_optimal_fleet
_count_manifest_shards = _mod._count_manifest_shards


def _mock_run(responses: dict[str, tuple[int, str]]):
    def side_effect(cmd, **kwargs):
        cmd_str = " ".join(str(c) for c in cmd)
        for pattern, (rc, stdout) in responses.items():
            if pattern in cmd_str:
                return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="not found")

    return side_effect


class TestDetectGpuQuota:
    def test_returns_quota_dict(self) -> None:
        responses = {
            "describe-instance-type-offerings": (
                0,
                json.dumps(["p4d.24xlarge", "g5.xlarge"]),
            ),
            "get-service-quota": (0, "96\n"),
            "describe-instance-types": (
                0,
                json.dumps(
                    [
                        {"Type": "p4d.24xlarge", "VCpuCount": 96},
                        {"Type": "g5.xlarge", "VCpuCount": 4},
                    ]
                ),
            ),
        }
        with mock.patch.object(_mod, "_run_cli", side_effect=_mock_run(responses)):
            quota = detect_gpu_quota("us-west-2")
        assert isinstance(quota, dict)
        assert quota.get("p4d.24xlarge") == 1
        assert quota.get("g5.xlarge") == 24

    def test_empty_on_no_offerings(self) -> None:
        responses = {
            "describe-instance-type-offerings": (0, json.dumps([])),
        }
        with mock.patch.object(_mod, "_run_cli", side_effect=_mock_run(responses)):
            quota = detect_gpu_quota("us-east-1")
        assert quota == {}

    def test_handles_cli_failure(self) -> None:
        responses = {
            "describe-instance-type-offerings": (1, ""),
        }
        with mock.patch.object(_mod, "_run_cli", side_effect=_mock_run(responses)):
            quota = detect_gpu_quota("eu-west-1")
        assert quota == {}


class TestCountManifestShards:
    def test_counts_tasks(self, tmp_path: Path) -> None:
        manifest = {
            "tasks": [
                {"task_id": "t1", "family": "dreamerv3"},
                {"task_id": "t2", "family": "tdmpc2"},
            ],
            "seed_policy": {"mode": "fixed", "values": [0, 1, 2]},
        }
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(manifest), encoding="utf-8")
        count = _count_manifest_shards(p)
        assert count == 12  # 2 tasks * 2 systems * 3 seeds

    def test_handles_empty_manifest(self, tmp_path: Path) -> None:
        manifest = {"tasks": []}
        p = tmp_path / "empty.json"
        p.write_text(json.dumps(manifest), encoding="utf-8")
        count = _count_manifest_shards(p)
        assert count == 0


class TestComputeOptimalFleet:
    def test_selects_p4d_when_available(self, tmp_path: Path) -> None:
        manifest = {
            "tasks": [{"task_id": f"t{i}", "family": "dreamerv3"} for i in range(10)],
            "seed_policy": {"mode": "fixed", "values": [0, 1]},
        }
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(manifest), encoding="utf-8")

        quota = {"p4d.24xlarge": 4, "g5.xlarge": 20}
        plan = compute_optimal_fleet(p, quota)
        assert plan is not None
        assert plan.instance_type == "p4d.24xlarge"
        assert plan.fleet_size >= 1
        assert plan.total_gpu_slots == plan.fleet_size * 8

    def test_falls_back_to_g5(self, tmp_path: Path) -> None:
        manifest = {
            "tasks": [{"task_id": "t1", "family": "dreamerv3"}],
            "seed_policy": {"mode": "fixed", "values": [0]},
        }
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(manifest), encoding="utf-8")

        quota = {"g5.xlarge": 10}
        plan = compute_optimal_fleet(p, quota)
        assert plan is not None
        assert plan.instance_type == "g5.xlarge"

    def test_returns_none_with_no_quota(self, tmp_path: Path) -> None:
        manifest = {
            "tasks": [{"task_id": "t1", "family": "dreamerv3"}],
            "seed_policy": {"mode": "fixed", "values": [0]},
        }
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(manifest), encoding="utf-8")
        plan = compute_optimal_fleet(p, {})
        assert plan is None

    def test_fleet_plan_fields(self, tmp_path: Path) -> None:
        manifest = {
            "tasks": [{"task_id": "t1", "family": "dreamerv3"}],
            "seed_policy": {"mode": "fixed", "values": [0, 1]},
        }
        p = tmp_path / "manifest.json"
        p.write_text(json.dumps(manifest), encoding="utf-8")

        quota = {"p4d.24xlarge": 2}
        plan = compute_optimal_fleet(p, quota)
        assert plan is not None
        assert isinstance(plan.estimated_wall_clock_hours, float)
        assert isinstance(plan.estimated_cost_usd, float)
        assert plan.estimated_cost_usd >= 0
        assert plan.fleet_split
