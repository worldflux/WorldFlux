#!/usr/bin/env python3
"""Run parity campaign generation from command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from worldflux.parity import CampaignRunOptions, load_campaign_spec, parse_seed_csv, run_campaign
from worldflux.parity.errors import ParityError


def _resolve_seeds(cli_seeds: tuple[int, ...], default_seeds: tuple[int, ...]) -> tuple[int, ...]:
    if cli_seeds:
        return cli_seeds
    if default_seeds:
        return default_seeds
    raise ParityError("No seeds provided. Pass --seeds or define campaign.default_seeds.")


def _serialize_summary(summary: dict[str, Any], output_json: Path | None) -> None:
    if output_json is None:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[campaign] wrote summary: {output_json}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True, help="Campaign spec path.")
    parser.add_argument(
        "--mode",
        choices=("worldflux", "oracle", "both"),
        default="worldflux",
        help="Execution mode.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds (e.g. 0,1,2). Uses campaign.default_seeds when omitted.",
    )
    parser.add_argument(
        "--device", default="cpu", help="Device value passed into command template."
    )
    parser.add_argument("--output", type=Path, default=None, help="Override worldflux output path.")
    parser.add_argument(
        "--oracle-output",
        type=Path,
        default=None,
        help="Override oracle output path.",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd(),
        help="Working directory for command execution.",
    )
    parser.add_argument(
        "--pair-output-root",
        type=Path,
        default=None,
        help="Root directory for per-task/seed command outputs.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing canonical outputs and skip finished task/seed pairs.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write campaign summary JSON.",
    )
    args = parser.parse_args()

    try:
        spec = load_campaign_spec(args.campaign.resolve())
        seeds = _resolve_seeds(parse_seed_csv(args.seeds), spec.default_seeds)
        options = CampaignRunOptions(
            mode=args.mode,
            seeds=seeds,
            device=str(args.device),
            output=args.output.resolve() if args.output is not None else None,
            oracle_output=args.oracle_output.resolve() if args.oracle_output is not None else None,
            resume=bool(args.resume),
            dry_run=bool(args.dry_run),
            workdir=args.workdir.resolve(),
            pair_output_root=(
                args.pair_output_root.resolve() if args.pair_output_root is not None else None
            ),
        )
        summary = run_campaign(spec, options)
        _serialize_summary(summary, args.summary_json)
    except (ParityError, ValueError, OSError) as exc:
        print(f"[campaign] failed: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
