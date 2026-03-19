#!/usr/bin/env python3
"""Launch a full parity campaign from a manifest YAML.

This script orchestrates multi-seed, multi-task parity evaluation campaigns
for DreamerV3 (Atari 100K) and TD-MPC2 (DMControl) benchmarks.  It wraps
the MultiSeedCampaign class and supports dry-run mode for infrastructure
validation without GPU resources.

Usage:
    # Dry-run to validate manifest and commands
    python scripts/launch_parity_campaign.py \
        --manifest scripts/parity/manifests/dreamerv3_atari100k_full.yaml \
        --dry-run

    # Full execution with custom seeds
    python scripts/launch_parity_campaign.py \
        --manifest scripts/parity/manifests/tdmpc2_dmcontrol_full.yaml \
        --seeds 0,1,2 \
        --device cuda

    # Generate badge after campaign completes
    python scripts/launch_parity_campaign.py \
        --manifest scripts/parity/manifests/dreamerv3_atari100k_full.yaml \
        --badge-only \
        --badge-output reports/parity/badges/dreamerv3_atari100k.svg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))


def _parse_seeds(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    seeds = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    return sorted(set(seeds)) if seeds else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch a parity campaign from a manifest YAML/JSON.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the campaign manifest file.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds (e.g. 0,1,2,3,4). Overrides manifest defaults.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device string (e.g. cpu, cuda, cuda:0). Default: cpu.",
    )
    parser.add_argument(
        "--mode",
        choices=("worldflux", "oracle", "both"),
        default="both",
        help="Which sources to run. Default: both.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Path to write the campaign summary JSON.",
    )
    parser.add_argument(
        "--badge-only",
        action="store_true",
        help="Skip campaign execution and only generate a badge from existing results.",
    )
    parser.add_argument(
        "--badge-output",
        type=Path,
        default=None,
        help="Path for the generated SVG badge.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd(),
        help="Working directory for subprocess commands.",
    )

    args = parser.parse_args()

    from worldflux.parity.badge import save_badge
    from worldflux.parity.campaign import MultiSeedCampaign, load_campaign_spec
    from worldflux.parity.errors import ParityError

    try:
        spec = load_campaign_spec(args.manifest)
    except ParityError as exc:
        print(f"[error] Failed to load manifest: {exc}", file=sys.stderr)
        return 1

    seeds = _parse_seeds(args.seeds)
    if seeds is None:
        seeds = list(spec.default_seeds) if spec.default_seeds else [0, 1, 2, 3, 4]

    print(f"[campaign] Suite: {spec.suite_id} ({spec.family})")
    print(f"[campaign] Tasks: {len(spec.tasks)}")
    print(f"[campaign] Seeds: {seeds}")
    print(f"[campaign] Mode: {args.mode}")
    print(f"[campaign] Device: {args.device}")

    if args.badge_only:
        badge_path = args.badge_output or Path(f"reports/parity/badges/{spec.suite_id}.svg")
        # Generate a placeholder badge (real data would come from aggregated results)
        save_badge(
            badge_path,
            family=spec.family,
            passed=False,
            confidence=0.95,
            margin=0.05,
        )
        print(f"[campaign] Badge written to {badge_path}")
        return 0

    campaign = MultiSeedCampaign(
        spec=spec,
        seeds=seeds,
        device=args.device,
        mode=args.mode,
        workdir=args.workdir,
    )

    try:
        jobs = campaign.launch_all(dry_run=args.dry_run)
    except ParityError as exc:
        print(f"[error] Campaign failed: {exc}", file=sys.stderr)
        return 1

    print(f"[campaign] Jobs: {len(jobs)}")
    for job in jobs[:5]:
        print(f"  - {job.source}/{job.task}/seed_{job.seed}: {job.status}")
    if len(jobs) > 5:
        print(f"  ... and {len(jobs) - 5} more")

    if not args.dry_run:
        try:
            report = campaign.generate_report()
            summary = {
                "suite_id": report.suite_id,
                "family": report.family,
                "seeds": report.seeds,
                "generated_at_utc": report.generated_at_utc,
                "worldflux_tasks": len(report.worldflux_results),
                "oracle_tasks": len(report.oracle_results),
            }

            if args.output_json:
                args.output_json.parent.mkdir(parents=True, exist_ok=True)
                args.output_json.write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(f"[campaign] Summary written to {args.output_json}")
            else:
                print(json.dumps(summary, indent=2, sort_keys=True))

            # Generate badge if requested
            if args.badge_output:
                tasks_pass = sum(
                    1
                    for r in report.worldflux_results
                    if any(
                        o.task == r.task and abs(r.mean - o.mean) / max(abs(o.mean), 1.0) < 0.05
                        for o in report.oracle_results
                    )
                )
                total = max(len(report.worldflux_results), 1)
                save_badge(
                    args.badge_output,
                    family=spec.family,
                    passed=tasks_pass >= total * 0.8,
                    confidence=0.95,
                    margin=0.05,
                )
                print(f"[campaign] Badge written to {args.badge_output}")

        except ParityError as exc:
            print(f"[warning] Report generation failed: {exc}", file=sys.stderr)

    print("[campaign] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
