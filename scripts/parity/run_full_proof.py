#!/usr/bin/env python3
"""One-command full parity proof: quota detection, fleet planning, orchestration, reporting."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
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
    run_id = str(args.run_id)

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


def _assemble_evidence_bundle(*, summary_path: Path) -> Path:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    artifacts = summary.get("artifacts", {}) if isinstance(summary, dict) else {}
    bundle_dir = summary_path.parent / "evidence_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = artifacts.get("manifest")
    merged_runs = artifacts.get("merged_runs")
    coverage_report = artifacts.get("coverage_report")
    validity_report = artifacts.get("validity_report")
    equivalence_report = artifacts.get("equivalence_report")
    equivalence_md = artifacts.get("equivalence_markdown")
    component_report = str(summary_path.parent / "component_match_report.json")
    merged_runs_path = Path(merged_runs) if merged_runs else None
    artifact_manifests_summary: dict[str, Any] = {}
    if merged_runs_path is not None and merged_runs_path.exists():
        lines = [
            json.loads(line)
            for line in merged_runs_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for system in ("official", "worldflux"):
            rows = [row for row in lines if str(row.get("system", "")) == system]
            artifact_manifests_summary[system] = {
                "count": len(rows),
                "backend_kinds": sorted(
                    {
                        str(row.get("backend_kind", "")).strip()
                        for row in rows
                        if str(row.get("backend_kind", "")).strip()
                    }
                ),
                "adapter_ids": sorted(
                    {
                        str(row.get("adapter_id", "")).strip()
                        for row in rows
                        if str(row.get("adapter_id", "")).strip()
                    }
                ),
                "recipe_hashes": sorted(
                    {
                        str(row.get("recipe_hash", "")).strip()
                        for row in rows
                        if str(row.get("recipe_hash", "")).strip()
                    }
                ),
                "artifact_manifest_count": sum(
                    1 for row in rows if isinstance(row.get("artifact_manifest"), dict)
                ),
            }

    artifact_summary_path = bundle_dir / "artifact_manifest_summary.json"
    artifact_summary_path.write_text(
        json.dumps(artifact_manifests_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    provenance_summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": str(summary_path),
        "execution_result": summary.get("execution_result") if isinstance(summary, dict) else None,
        "manifest": manifest_path,
        "merged_runs": merged_runs,
        "coverage_report": coverage_report,
        "validity_report": validity_report,
        "equivalence_report": equivalence_report,
        "equivalence_markdown": equivalence_md,
        "component_report": component_report if Path(component_report).exists() else None,
        "artifact_manifests": artifact_manifests_summary,
        "artifact_manifest_summary": str(artifact_summary_path),
    }
    (bundle_dir / "provenance_summary.json").write_text(
        json.dumps(provenance_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    included_paths = [
        Path(path)
        for path in (
            summary_path,
            Path(manifest_path) if manifest_path else None,
            Path(merged_runs) if merged_runs else None,
            Path(coverage_report) if coverage_report else None,
            Path(validity_report) if validity_report else None,
            Path(equivalence_report) if equivalence_report else None,
            Path(equivalence_md) if equivalence_md else None,
            Path(component_report) if Path(component_report).exists() else None,
            artifact_summary_path,
            bundle_dir / "provenance_summary.json",
        )
        if path is not None and Path(path).exists()
    ]

    index_payload = {
        "schema_version": "parity.evidence_bundle.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": [str(path.resolve()) for path in included_paths],
    }
    index_path = bundle_dir / "bundle_index.json"
    index_path.write_text(
        json.dumps(index_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    included_paths.append(index_path)

    zip_path = summary_path.parent / "evidence_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in included_paths:
            handle.write(path, arcname=path.relative_to(summary_path.parent))
    return zip_path


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
    if not args.run_id:
        args.run_id = f"proof_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

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

    summary_path = Path("reports/parity") / str(args.run_id) / "orchestrator_summary.json"
    if summary_path.exists():
        bundle_path = _assemble_evidence_bundle(summary_path=summary_path)
        print(f"[full-proof] evidence bundle: {bundle_path}")

    print("[full-proof] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
