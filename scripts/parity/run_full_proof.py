#!/usr/bin/env python3
"""One-command full parity proof: quota detection, fleet planning, orchestration, reporting."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _run_subprocess(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def _build_orchestrator_command(
    args: argparse.Namespace, *, fleet_plan: dict[str, Any] | None = None
) -> list[str]:
    """Build the aws_distributed_orchestrator.py invocation from CLI args."""
    script = str(Path(__file__).resolve().parent / "aws_distributed_orchestrator.py")
    run_id = args.run_id or f"proof_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    cmd = [
        sys.executable,
        script,
        "--region",
        args.region,
        "--manifest",
        str(args.manifest),
        "--run-id",
        run_id,
        "--s3-prefix",
        f"s3://worldflux-parity/{run_id}",
        "--device",
        "cuda",
        "--auto-provision",
    ]

    # Fleet configuration: use fleet_plan if auto-quota was used, otherwise CLI args.
    if fleet_plan is not None:
        cmd += ["--instance-type", fleet_plan["instance_type"]]
        cmd += ["--fleet-size", str(fleet_plan["fleet_size"])]
        cmd += ["--fleet-split", fleet_plan["fleet_split"]]
    else:
        cmd += ["--instance-type", args.instance_type]
        cmd += ["--fleet-size", str(args.fleet_size)]
        cmd += ["--fleet-split", args.fleet_split]

    # Provisioning parameters.
    if args.image_id:
        cmd += ["--image-id", args.image_id]
    if args.subnet_id:
        cmd += ["--subnet-id", args.subnet_id]
    if args.security_group_ids:
        cmd += ["--security-group-ids", args.security_group_ids]
    if args.iam_instance_profile:
        cmd += ["--iam-instance-profile", args.iam_instance_profile]
    if args.key_name:
        cmd += ["--key-name", args.key_name]

    # Sharding and phase options.
    cmd += ["--phase-plan", args.phase_plan]
    cmd += ["--sharding-mode", args.sharding_mode]
    cmd += ["--seed-shard-unit", args.seed_shard_unit]
    cmd += ["--thread-limit-profile", args.thread_limit_profile]
    cmd += ["--cpu-affinity-policy", args.cpu_affinity_policy]

    if args.auto_terminate:
        cmd += ["--auto-terminate"]

    # Post-run hooks.
    hooks: list[str] = []
    if args.paper_comparison:
        hooks.append("paper_comparison")
    if args.plot_curves:
        hooks.append("plot_curves")
    if hooks:
        cmd += ["--post-run-hooks", ",".join(hooks)]

    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", type=str, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-id", type=str, default="")

    # Fleet configuration.
    parser.add_argument("--instance-type", type=str, default="p4d.24xlarge")
    parser.add_argument("--fleet-size", type=int, default=4)
    parser.add_argument("--fleet-split", type=str, default="2,2")
    parser.add_argument(
        "--auto-quota",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Auto-detect GPU quota and compute fleet plan.",
    )
    parser.add_argument("--max-wall-clock-hours", type=float, default=48.0)

    # Provisioning parameters.
    parser.add_argument("--image-id", type=str, default="")
    parser.add_argument("--subnet-id", type=str, default="")
    parser.add_argument("--security-group-ids", type=str, default="")
    parser.add_argument("--iam-instance-profile", type=str, default="")
    parser.add_argument("--key-name", type=str, default="")

    # Sharding and phase.
    parser.add_argument(
        "--phase-plan", type=str, choices=["single", "two_stage_proof"], default="single"
    )
    parser.add_argument(
        "--sharding-mode", type=str, choices=["task", "seed_system"], default="task"
    )
    parser.add_argument("--seed-shard-unit", type=str, choices=["packed", "pair"], default="packed")
    parser.add_argument("--auto-terminate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--thread-limit-profile", type=str, choices=["off", "strict1"], default="off"
    )
    parser.add_argument(
        "--cpu-affinity-policy", type=str, choices=["none", "p4d_8gpu_12vcpu"], default="none"
    )

    # Post-run hooks.
    parser.add_argument(
        "--paper-comparison",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run paper baseline comparison after proof completes.",
    )
    parser.add_argument(
        "--plot-curves",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Generate learning curve plots after proof completes.",
    )

    args = parser.parse_args()

    fleet_plan: dict[str, Any] | None = None

    # Step 1: quota detection + fleet planning (if --auto-quota).
    if args.auto_quota:
        print("[full-proof] step 1/3: detecting GPU quota and computing fleet plan ...")
        planner_script = str(Path(__file__).resolve().parent / "aws_quota_planner.py")
        result = _run_subprocess(
            [
                sys.executable,
                planner_script,
                "--region",
                args.region,
                "--manifest",
                str(args.manifest),
                "--max-wall-clock-hours",
                str(args.max_wall_clock_hours),
            ],
            check=False,
        )
        if result.returncode != 0:
            print(f"[full-proof] ERROR: quota planner failed:\n{result.stderr}", file=sys.stderr)
            return 1

        # Parse the JSON fleet plan from the last line of stdout.
        lines = result.stdout.strip().splitlines()
        json_lines: list[str] = []
        brace_depth = 0
        for line in reversed(lines):
            stripped = line.strip()
            if stripped.endswith("}"):
                brace_depth += stripped.count("}") - stripped.count("{")
                json_lines.insert(0, line)
            elif brace_depth > 0:
                brace_depth += stripped.count("{") - stripped.count("}")
                json_lines.insert(0, line)
                if brace_depth <= 0:
                    break

        try:
            fleet_plan = json.loads("\n".join(json_lines))
        except (json.JSONDecodeError, IndexError):
            print(
                "[full-proof] ERROR: failed to parse fleet plan from quota planner output",
                file=sys.stderr,
            )
            return 1

        print(f"[full-proof] fleet plan: {json.dumps(fleet_plan, indent=2)}")
    else:
        print("[full-proof] step 1/3: using explicit fleet configuration")

    # Step 2: run orchestrator.
    print("[full-proof] step 2/3: launching orchestrator ...")
    orchestrator_cmd = _build_orchestrator_command(args, fleet_plan=fleet_plan)
    print(f"[full-proof] command: {' '.join(orchestrator_cmd)}")

    result = _run_subprocess(orchestrator_cmd, check=False)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        print(
            f"[full-proof] ERROR: orchestrator exited with code {result.returncode}",
            file=sys.stderr,
        )
        return result.returncode

    # Step 3: post-run hooks.
    print("[full-proof] step 3/3: post-run hooks")
    if args.paper_comparison:
        print("[full-proof] paper_comparison hook: not yet implemented (Phase 4)")
    if args.plot_curves:
        print("[full-proof] plot_curves hook: not yet implemented (Phase 5)")

    print("[full-proof] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
