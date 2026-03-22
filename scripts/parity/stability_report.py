#!/usr/bin/env python3
"""Generate a deterministic stability report for proof-grade parity runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to parity_runs.jsonl.")
    parser.add_argument(
        "--equivalence-report",
        type=Path,
        required=True,
        help="Path to equivalence_report.json.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path to stability_report.json.")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object at {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_stability_report(*, runs_path: Path, equivalence_report_path: Path) -> dict[str, Any]:
    rows = _load_jsonl(runs_path)
    equivalence_report = _load_json(equivalence_report_path)
    global_block = equivalence_report.get("global", {})
    if not isinstance(global_block, dict):
        global_block = {}

    tasks: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "systems": set(),
            "seeds": set(),
            "success_rows": 0,
            "failed_rows": 0,
        }
    )
    system_success_counts: dict[str, int] = defaultdict(int)
    system_failure_counts: dict[str, int] = defaultdict(int)

    for row in rows:
        task_id = str(row.get("task_id", "")).strip() or "<unknown>"
        system = str(row.get("system", "")).strip() or "<unknown>"
        seed = int(row.get("seed", -1))
        status = str(row.get("status", "")).strip().lower()

        task_entry = tasks[task_id]
        task_entry["systems"].add(system)
        if seed >= 0:
            task_entry["seeds"].add(seed)

        if status == "success":
            task_entry["success_rows"] += 1
            system_success_counts[system] += 1
        else:
            task_entry["failed_rows"] += 1
            system_failure_counts[system] += 1

    rendered_tasks = []
    for task_id in sorted(tasks):
        payload = tasks[task_id]
        rendered_tasks.append(
            {
                "task_id": task_id,
                "system_count": len(payload["systems"]),
                "systems": sorted(payload["systems"]),
                "seed_count": len(payload["seeds"]),
                "seed_values": sorted(payload["seeds"]),
                "success_rows": int(payload["success_rows"]),
                "failed_rows": int(payload["failed_rows"]),
                "artifact_complete": bool(
                    payload["failed_rows"] == 0 and payload["success_rows"] > 0
                ),
            }
        )

    stability_status = "stable"
    if any(task["failed_rows"] > 0 for task in rendered_tasks):
        stability_status = "unstable"
    elif not bool(global_block.get("parity_pass_final", False)):
        stability_status = "verdict_failed"

    return {
        "schema_version": "parity.stability.v1",
        "generated_at": _timestamp(),
        "input": str(runs_path.resolve()),
        "equivalence_report": str(equivalence_report_path.resolve()),
        "status": stability_status,
        "global": {
            "parity_pass_final": bool(global_block.get("parity_pass_final", False)),
            "validity_pass": bool(global_block.get("validity_pass", False)),
            "missing_pairs": int(global_block.get("missing_pairs", 0) or 0),
            "task_count": len(rendered_tasks),
            "row_count": len(rows),
            "successful_system_rows": dict(sorted(system_success_counts.items())),
            "failed_system_rows": dict(sorted(system_failure_counts.items())),
        },
        "tasks": rendered_tasks,
        "rerun_consistency": {
            "mode": "single_run",
            "verdict_flip_detected": False,
            "pairwise_metric_sign_flip_detected": False,
            "bayesian_frequentist_mismatch_detected": False,
        },
    }


def main() -> int:
    args = _parse_args()
    report = build_stability_report(
        runs_path=args.input,
        equivalence_report_path=args.equivalence_report,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
