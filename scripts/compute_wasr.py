#!/usr/bin/env python3
"""Compute simple WASR metrics from local JSONL telemetry."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_events(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []

    events: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                events.append(value)
    return events


def _week_bucket(epoch_seconds: float) -> str:
    # UTC week bucket label like 2026-W06
    ts = int(epoch_seconds)
    tm = time.gmtime(ts)
    year, week, _ = time.strftime("%G %V %u", tm).split(" ")
    return f"{year}-W{week}"


def compute_wasr(
    events: list[dict[str, Any]],
    *,
    now: float,
    lookback_days: int = 7,
) -> dict[str, Any]:
    window_start = now - lookback_days * 24 * 60 * 60

    recent = [e for e in events if float(e.get("timestamp", 0.0)) >= window_start]
    success_recent = [e for e in recent if bool(e.get("success", False))]

    unique_success_run_ids = {
        str(e.get("run_id", "")) for e in success_recent if str(e.get("run_id", ""))
    }

    quickstart_events = [
        e
        for e in recent
        if str(e.get("scenario", "")).strip().startswith("quickstart")
        and str(e.get("event", "")).strip() == "run_complete"
    ]
    quickstart_successes = [e for e in quickstart_events if bool(e.get("success", False))]

    scenario_counts: dict[str, int] = defaultdict(int)
    for e in recent:
        scenario_counts[str(e.get("scenario", "unknown"))] += 1

    # Minimal week-over-week retention proxy: users active in consecutive weeks.
    runs_by_week: dict[str, set[str]] = defaultdict(set)
    for e in events:
        run_id = str(e.get("run_id", ""))
        ts = float(e.get("timestamp", 0.0))
        if run_id:
            runs_by_week[_week_bucket(ts)].add(run_id)

    weeks_sorted = sorted(runs_by_week.keys())
    retention = []
    for idx in range(1, len(weeks_sorted)):
        prev_ids = runs_by_week[weeks_sorted[idx - 1]]
        cur_ids = runs_by_week[weeks_sorted[idx]]
        base = len(prev_ids)
        rate = (len(prev_ids & cur_ids) / base) if base > 0 else math.nan
        retention.append(
            {
                "from_week": weeks_sorted[idx - 1],
                "to_week": weeks_sorted[idx],
                "retained": len(prev_ids & cur_ids),
                "base": base,
                "rate": rate,
            }
        )

    return {
        "window": {
            "lookback_days": int(lookback_days),
            "start": float(window_start),
            "end": float(now),
        },
        "wasr": int(len(unique_success_run_ids)),
        "quickstart": {
            "attempts": int(len(quickstart_events)),
            "successes": int(len(quickstart_successes)),
            "success_rate": (
                float(len(quickstart_successes) / len(quickstart_events))
                if quickstart_events
                else 0.0
            ),
        },
        "scenarios": dict(sorted(scenario_counts.items())),
        "retention": retention,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute WASR metrics from local JSONL logs")
    parser.add_argument(
        "--input",
        type=str,
        default=".worldflux/metrics.jsonl",
        help="Path to telemetry JSONL",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="-",
        help="Output path for JSON report (default: stdout)",
    )
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument(
        "--now",
        type=float,
        default=None,
        help="Optional unix timestamp for deterministic computation",
    )
    args = parser.parse_args()

    events = _load_events(args.input)
    result = compute_wasr(
        events,
        now=float(args.now if args.now is not None else time.time()),
        lookback_days=int(args.lookback_days),
    )

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output == "-":
        print(text)
    else:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
