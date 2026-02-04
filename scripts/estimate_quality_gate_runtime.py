#!/usr/bin/env python3
"""
Estimate runtime for quality-gate measurements using existing seed run results.

Reads reports/quality-gates/seed_runs.json and estimates total runtime for
given (seeds x steps) scenarios using average seconds-per-step.
"""

from __future__ import annotations

import argparse
import json
from statistics import mean, pstdev
from typing import Any


def _parse_scenarios(raw: str) -> list[tuple[int, int]]:
    scenarios: list[tuple[int, int]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "x" not in part:
            raise ValueError(f"Invalid scenario '{part}'. Use seedsxsteps (e.g. 5x5000).")
        seeds_str, steps_str = part.split("x", 1)
        scenarios.append((int(seeds_str), int(steps_str)))
    return scenarios


def _seconds_per_step(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_model: dict[str, list[float]] = {}
    for run in runs:
        steps = float(run.get("steps", 0))
        duration = float(run.get("duration_sec", 0))
        if steps <= 0 or duration <= 0:
            continue
        by_model.setdefault(run["model"], []).append(duration / steps)

    stats: dict[str, dict[str, float]] = {}
    for model, values in by_model.items():
        stats[model] = {
            "mean_s_per_step": mean(values),
            "std_s_per_step": pstdev(values) if len(values) > 1 else 0.0,
            "samples": float(len(values)),
        }
    return stats


def _format_seconds(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}m"
    return f"{sec / 3600:.2f}h"


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate runtime for quality-gate scenarios")
    parser.add_argument(
        "--input",
        type=str,
        default="reports/quality-gates/seed_runs.json",
        help="Path to seed_runs.json",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model IDs to include or 'all'",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default="5x5000,7x10000,10x10000,7x20000,10x20000",
        help="Comma-separated scenarios as seedsxsteps",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path for estimates",
    )

    args = parser.parse_args()

    data = json.loads(open(args.input).read())
    runs = data.get("runs", [])
    stats = _seconds_per_step(runs)

    selected_models = list(stats.keys())
    if args.models != "all":
        requested = [m.strip() for m in args.models.split(",") if m.strip()]
        selected_models = [m for m in requested if m in stats]

    scenarios = _parse_scenarios(args.scenarios)

    estimates: dict[str, Any] = {"models": {}, "scenarios": []}

    for model in selected_models:
        model_stats = stats[model]
        estimates["models"][model] = model_stats

    for seeds, steps in scenarios:
        entry: dict[str, Any] = {"seeds": seeds, "steps": steps, "models": {}}
        total_sec = 0.0
        for model in selected_models:
            mean_s = stats[model]["mean_s_per_step"]
            est = mean_s * seeds * steps
            entry["models"][model] = {
                "seconds": est,
                "human": _format_seconds(est),
            }
            total_sec += est
        entry["total"] = {"seconds": total_sec, "human": _format_seconds(total_sec)}
        estimates["scenarios"].append(entry)

    print("Runtime estimates (based on measured seconds/step):")
    for entry in estimates["scenarios"]:
        print(f"- {entry['seeds']} seeds × {entry['steps']} steps")
        for model, info in entry["models"].items():
            print(f"  {model}: {info['human']}")
        print(f"  total: {entry['total']['human']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(estimates, f, indent=2)
        print(f"Wrote estimates to {args.output}")


if __name__ == "__main__":
    main()
