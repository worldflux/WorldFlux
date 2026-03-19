#!/usr/bin/env python3
"""Launch hyperparameter sensitivity analysis for DreamerV3.

Generates one-at-a-time sweep configurations and (optionally) executes them
on a fast environment. Produces a JSON report and Markdown summary.

Usage:
    # List all configurations without running
    python scripts/run_sensitivity.py --dry-run

    # Full execution (requires environment + GPU)
    python scripts/run_sensitivity.py \
        --environment CartPole-v1 \
        --seeds 0,1,2 \
        --steps 100000 \
        --output reports/parity/sensitivity/dreamerv3_sensitivity.json

    # Generate Markdown report from existing JSON
    python scripts/run_sensitivity.py \
        --report-from reports/parity/sensitivity/dreamerv3_sensitivity.json \
        --output-md docs/reference/hyperparameter-sensitivity.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))


def _parse_seeds(raw: str | None) -> list[int]:
    if raw is None:
        return [0, 1, 2]
    return sorted({int(s.strip()) for s in raw.split(",") if s.strip()})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        default="CartPole-v1",
        help="Environment to evaluate (default: CartPole-v1).",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds (default: 0,1,2).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
        help="Environment steps per run (default: 100000).",
    )
    parser.add_argument(
        "--family",
        default="dreamerv3",
        help="Model family (default: dreamerv3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for JSON output.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Path for Markdown output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configurations without executing.",
    )
    parser.add_argument(
        "--report-from",
        type=Path,
        default=None,
        help="Generate report from existing JSON results instead of running.",
    )

    args = parser.parse_args()

    from worldflux.parity.sensitivity import (
        DREAMERV3_SWEEPS,
        SensitivityAnalysis,
        SensitivityReport,
        render_sensitivity_markdown,
    )

    seeds = _parse_seeds(args.seeds)

    # Report-from mode: just render existing results
    if args.report_from is not None:
        raw = json.loads(args.report_from.read_text(encoding="utf-8"))
        from worldflux.parity.sensitivity import ParameterSensitivity

        params = []
        for p in raw.get("parameters", []):
            params.append(
                ParameterSensitivity(
                    name=p["name"],
                    default_value=p["default_value"],
                    values=p["values"],
                    mean_rewards=p["mean_rewards"],
                    std_rewards=p["std_rewards"],
                    sensitivity_score=p["sensitivity_score"],
                    default_rank_percentile=p["default_rank_percentile"],
                )
            )
        report = SensitivityReport(
            family=raw.get("family", "dreamerv3"),
            environment=raw.get("environment", "unknown"),
            seeds=raw.get("seeds", []),
            total_steps=raw.get("total_steps", 0),
            parameters=params,
            generated_at_utc=raw.get("generated_at_utc", ""),
        )
        md = render_sensitivity_markdown(report)
        if args.output_md:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(md, encoding="utf-8")
            print(f"[sensitivity] Markdown written to {args.output_md}")
        else:
            print(md)
        return 0

    analysis = SensitivityAnalysis(
        sweeps=DREAMERV3_SWEEPS,
        family=args.family,
        environment=args.environment,
        seeds=seeds,
        total_steps=args.steps,
    )

    configs = analysis.generate_run_configs()
    print(f"[sensitivity] Total configurations: {len(configs)}")
    print(f"[sensitivity] Parameters: {[s.name for s in analysis.sweeps]}")
    print(f"[sensitivity] Seeds: {seeds}")
    print(f"[sensitivity] Steps: {args.steps}")

    if args.dry_run:
        print("\n[dry-run] Configurations:")
        for i, cfg in enumerate(configs):
            print(f"  [{i+1:3d}] {cfg['param_name']}={cfg['param_value']} " f"seed={cfg['seed']}")
        print(f"\n[dry-run] Would run {len(configs)} experiments. Exiting.")
        return 0

    # Actual execution would happen here. For now, we just generate the
    # infrastructure. Real runs require environment + model setup.
    print(
        "[sensitivity] Actual execution not available in this script. "
        "Use the generated configs with your training pipeline."
    )
    print("[sensitivity] Exporting run configs for external execution...")

    if args.output:
        # Export the configs as a launch manifest
        manifest = {
            "schema_version": "worldflux.sensitivity.manifest.v1",
            "family": args.family,
            "environment": args.environment,
            "seeds": seeds,
            "total_steps": args.steps,
            "total_runs": len(configs),
            "configs": configs,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[sensitivity] Manifest written to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
