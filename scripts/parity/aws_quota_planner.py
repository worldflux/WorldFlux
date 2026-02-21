#!/usr/bin/env python3
"""Detect AWS GPU quota and compute optimal fleet plan for parity proof runs."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# On-demand pricing per hour (USD) for common GPU instance types.
_ON_DEMAND_HOURLY_USD: dict[str, float] = {
    "p4d.24xlarge": 32.77,
    "p4de.24xlarge": 40.97,
    "p5.48xlarge": 98.32,
    "g5.xlarge": 1.006,
    "g5.2xlarge": 1.212,
    "g5.4xlarge": 1.624,
    "g5.12xlarge": 5.672,
    "g5.48xlarge": 16.288,
    "g6.xlarge": 0.8049,
    "g6.2xlarge": 0.978,
}

# Known GPU counts per instance type.
_INSTANCE_GPU_COUNTS: dict[str, int] = {
    "p4d.24xlarge": 8,
    "p4de.24xlarge": 8,
    "p5.48xlarge": 8,
    "g5.xlarge": 1,
    "g5.2xlarge": 1,
    "g5.4xlarge": 1,
    "g5.12xlarge": 4,
    "g5.48xlarge": 8,
    "g6.xlarge": 1,
    "g6.2xlarge": 1,
}


@dataclass(frozen=True)
class FleetPlan:
    instance_type: str
    fleet_size: int
    fleet_split: str
    estimated_wall_clock_hours: float
    estimated_cost_usd: float
    total_gpu_slots: int
    shards_count: int


def _run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, text=True, capture_output=True)


def detect_gpu_quota(region: str) -> dict[str, int]:
    """Query AWS service-quotas and ec2 offerings to find available GPU instances.

    Returns a mapping of instance_type -> max allowed instances.
    """
    # Step 1: discover which GPU instance types are offered in the region.
    offered: set[str] = set()
    cli = [
        "aws",
        "ec2",
        "describe-instance-type-offerings",
        "--region",
        region,
        "--location-type",
        "region",
        "--filters",
        "Name=instance-type,Values=p4d.*,p4de.*,p5.*,g5.*,g6.*",
        "--query",
        "InstanceTypeOfferings[].InstanceType",
        "--output",
        "json",
    ]
    result = _run_cli(cli)
    if result.returncode == 0:
        try:
            for entry in json.loads(result.stdout):
                offered.add(str(entry))
        except (json.JSONDecodeError, TypeError):
            pass

    if not offered:
        return {}

    # Step 2: check service-quotas for vCPU limits.
    # The P-family on-demand quota code is L-417A185B,
    # G-family on-demand quota code is L-DB2E81BA.
    quota_codes = {
        "p": "L-417A185B",
        "g": "L-DB2E81BA",
    }
    family_vcpu_limits: dict[str, int] = {}
    for prefix, code in quota_codes.items():
        cli = [
            "aws",
            "service-quotas",
            "get-service-quota",
            "--region",
            region,
            "--service-code",
            "ec2",
            "--quota-code",
            code,
            "--query",
            "Quota.Value",
            "--output",
            "text",
        ]
        result = _run_cli(cli)
        if result.returncode == 0:
            try:
                family_vcpu_limits[prefix] = int(float(result.stdout.strip()))
            except (ValueError, TypeError):
                pass

    # Step 3: resolve vCPU counts per instance type.
    vcpu_counts: dict[str, int] = {}
    offered_list = sorted(offered)
    cli = [
        "aws",
        "ec2",
        "describe-instance-types",
        "--region",
        region,
        "--instance-types",
        *offered_list,
        "--query",
        "InstanceTypes[].{Type:InstanceType,VCpuCount:VCpuInfo.DefaultVCpus}",
        "--output",
        "json",
    ]
    result = _run_cli(cli)
    if result.returncode == 0:
        try:
            for row in json.loads(result.stdout):
                itype = str(row.get("Type", ""))
                vcpus = row.get("VCpuCount", 0)
                if itype and isinstance(vcpus, int) and vcpus > 0:
                    vcpu_counts[itype] = vcpus
        except (json.JSONDecodeError, TypeError):
            pass

    # Step 4: compute max instance count = floor(vcpu_limit / vcpu_per_instance).
    quota: dict[str, int] = {}
    for itype in sorted(offered):
        prefix = itype[0].lower()
        vcpu_limit = family_vcpu_limits.get(prefix)
        vcpu_per = vcpu_counts.get(itype)
        if vcpu_limit is not None and vcpu_per is not None and vcpu_per > 0:
            quota[itype] = vcpu_limit // vcpu_per

    return quota


def _count_manifest_shards(
    manifest_path: Path, systems: tuple[str, ...] = ("official", "worldflux")
) -> int:
    """Count the number of task * system pairs in a manifest (minimum shard count)."""
    text = manifest_path.read_text(encoding="utf-8")
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

    tasks = payload.get("tasks", [])
    if not isinstance(tasks, list):
        return 0
    task_count = sum(1 for t in tasks if isinstance(t, dict) and t.get("task_id"))

    seed_policy = payload.get("seed_policy", {})
    seed_count = 1
    if isinstance(seed_policy, dict):
        mode = str(seed_policy.get("mode", "fixed"))
        if mode == "fixed":
            values = seed_policy.get("values", [])
            if isinstance(values, list):
                seed_count = max(1, len(values))
        else:
            seed_count = max(1, int(seed_policy.get("min_seeds", 20)))

    return task_count * len(systems) * seed_count


def compute_optimal_fleet(
    manifest_path: Path,
    quota: dict[str, int],
    max_wall_clock_hours: float = 48.0,
) -> FleetPlan | None:
    """Select the best instance type and fleet size from available quota.

    Strategy: prefer p4d.24xlarge (8 GPUs), then fall back to other GPU types.
    Fleet size is the minimum of quota and shards_count, bounded by wall-clock target.
    """
    preferred_order = [
        "p4d.24xlarge",
        "p4de.24xlarge",
        "p5.48xlarge",
        "g5.48xlarge",
        "g5.12xlarge",
        "g5.4xlarge",
        "g5.2xlarge",
        "g5.xlarge",
        "g6.2xlarge",
        "g6.xlarge",
    ]

    shards_count = _count_manifest_shards(manifest_path)
    if shards_count <= 0:
        return None

    for itype in preferred_order:
        max_instances = quota.get(itype, 0)
        if max_instances <= 0:
            continue

        gpus_per_instance = _INSTANCE_GPU_COUNTS.get(itype, 1)
        hourly_cost = _ON_DEMAND_HOURLY_USD.get(itype, 0.0)

        # Fleet size: enough instances so that total GPU slots >= shards.
        # But capped by quota and a practical maximum.
        needed = max(1, math.ceil(shards_count / gpus_per_instance))
        fleet_size = min(needed, max_instances)
        total_gpu_slots = fleet_size * gpus_per_instance

        # Estimate wall clock: each GPU slot runs ~1 shard sequentially.
        shards_per_slot = (
            math.ceil(shards_count / total_gpu_slots) if total_gpu_slots > 0 else shards_count
        )
        # Rough estimate: 2 hours per shard (dreamerv3 ~2h, tdmpc2 ~3h).
        est_hours = shards_per_slot * 2.5

        if est_hours > max_wall_clock_hours and fleet_size < max_instances:
            # Scale up to meet wall-clock target.
            needed_slots = math.ceil(shards_count * 2.5 / max_wall_clock_hours)
            fleet_size = min(max_instances, max(1, math.ceil(needed_slots / gpus_per_instance)))
            total_gpu_slots = fleet_size * gpus_per_instance
            shards_per_slot = (
                math.ceil(shards_count / total_gpu_slots) if total_gpu_slots > 0 else shards_count
            )
            est_hours = shards_per_slot * 2.5

        est_cost = fleet_size * hourly_cost * est_hours

        # Split fleet evenly: half official, half worldflux.
        half = fleet_size // 2
        remainder = fleet_size - half * 2
        fleet_split = f"{half + remainder},{half}" if fleet_size > 1 else "1"

        return FleetPlan(
            instance_type=itype,
            fleet_size=fleet_size,
            fleet_split=fleet_split,
            estimated_wall_clock_hours=round(est_hours, 2),
            estimated_cost_usd=round(est_cost, 2),
            total_gpu_slots=total_gpu_slots,
            shards_count=shards_count,
        )

    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-wall-clock-hours", type=float, default=48.0)
    args = parser.parse_args()

    print(f"[quota-planner] detecting GPU quota in {args.region} ...")
    quota = detect_gpu_quota(args.region)
    if not quota:
        print("[quota-planner] ERROR: no GPU instance types available in region", file=sys.stderr)
        return 1

    print(f"[quota-planner] available GPU quota: {json.dumps(quota, indent=2)}")

    plan = compute_optimal_fleet(args.manifest, quota, args.max_wall_clock_hours)
    if plan is None:
        print("[quota-planner] ERROR: could not compute a viable fleet plan", file=sys.stderr)
        return 1

    print("[quota-planner] fleet plan:")
    print(f"  instance_type:            {plan.instance_type}")
    print(f"  fleet_size:               {plan.fleet_size}")
    print(f"  fleet_split:              {plan.fleet_split}")
    print(f"  total_gpu_slots:          {plan.total_gpu_slots}")
    print(f"  shards_count:             {plan.shards_count}")
    print(f"  estimated_wall_clock_hours: {plan.estimated_wall_clock_hours}")
    print(f"  estimated_cost_usd:       ${plan.estimated_cost_usd:.2f}")

    plan_dict = {
        "instance_type": plan.instance_type,
        "fleet_size": plan.fleet_size,
        "fleet_split": plan.fleet_split,
        "estimated_wall_clock_hours": plan.estimated_wall_clock_hours,
        "estimated_cost_usd": plan.estimated_cost_usd,
        "total_gpu_slots": plan.total_gpu_slots,
        "shards_count": plan.shards_count,
    }
    print(json.dumps(plan_dict, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
