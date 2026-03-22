#!/usr/bin/env python3
"""Generate a proof-grade stability report from current and prior proof reports."""

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
        help="Path to the current equivalence_report.json.",
    )
    parser.add_argument(
        "--history-equivalence-report",
        action="append",
        default=[],
        help="Optional prior equivalence_report.json path. Repeatable.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path to stability_report.json.")
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _report_verdict(report: dict[str, Any]) -> bool:
    global_block = report.get("global", {})
    if not isinstance(global_block, dict):
        return False
    return bool(global_block.get("parity_pass_final", False))


def _report_validity(report: dict[str, Any]) -> bool:
    global_block = report.get("global", {})
    if not isinstance(global_block, dict):
        return False
    return bool(global_block.get("validity_pass", False))


def _report_missing_pairs(report: dict[str, Any]) -> int:
    global_block = report.get("global", {})
    if not isinstance(global_block, dict):
        return 0
    return int(global_block.get("missing_pairs", 0) or 0)


def _report_bayes_freq_mismatch(report: dict[str, Any]) -> bool:
    global_block = report.get("global", {})
    if not isinstance(global_block, dict):
        return False
    if "parity_pass_bayesian" not in global_block or "parity_pass_frequentist" not in global_block:
        return False
    return bool(global_block.get("parity_pass_bayesian")) != bool(
        global_block.get("parity_pass_frequentist")
    )


def _sign(value: Any) -> int | None:
    if not isinstance(value, int | float):
        return None
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _metric_signatures(report: dict[str, Any]) -> dict[tuple[str, str], int]:
    signatures: dict[tuple[str, str], int] = {}
    tasks = report.get("tasks", [])
    if not isinstance(tasks, list):
        return signatures
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id", "")).strip() or "<unknown>"
        metrics = task.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for metric_name, metric_payload in metrics.items():
            if not isinstance(metric_payload, dict):
                continue
            metric_sign = _sign(metric_payload.get("ratio_mean"))
            if metric_sign is None:
                continue
            signatures[(task_id, str(metric_name))] = metric_sign
    return signatures


def _build_task_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "systems": set(),
            "seeds": set(),
            "success_rows": 0,
            "failed_rows": 0,
        }
    )

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
        else:
            task_entry["failed_rows"] += 1

    rendered: list[dict[str, Any]] = []
    for task_id in sorted(tasks):
        payload = tasks[task_id]
        rendered.append(
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
    return rendered


def build_stability_report(
    *,
    runs_path: Path,
    equivalence_report_path: Path,
    history_equivalence_report_paths: list[Path] | None = None,
) -> dict[str, Any]:
    rows = _load_jsonl(runs_path)
    current_report = _load_json(equivalence_report_path)
    history_paths = [path.resolve() for path in (history_equivalence_report_paths or [])]
    history_reports = [_load_json(path) for path in history_paths]

    system_success_counts: dict[str, int] = defaultdict(int)
    system_failure_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        system = str(row.get("system", "")).strip() or "<unknown>"
        status = str(row.get("status", "")).strip().lower()
        if status == "success":
            system_success_counts[system] += 1
        else:
            system_failure_counts[system] += 1

    tasks = _build_task_summary(rows)
    current_signatures = _metric_signatures(current_report)
    current_verdict = _report_verdict(current_report)

    verdict_flip_detected = any(
        _report_verdict(report) != current_verdict for report in history_reports
    )
    pairwise_metric_sign_flip_detected = False
    bayesian_frequentist_mismatch_detected = _report_bayes_freq_mismatch(current_report)
    if not bayesian_frequentist_mismatch_detected:
        bayesian_frequentist_mismatch_detected = any(
            _report_bayes_freq_mismatch(report) for report in history_reports
        )

    for report in history_reports:
        history_signatures = _metric_signatures(report)
        shared_keys = set(current_signatures).intersection(history_signatures)
        for key in shared_keys:
            if current_signatures[key] != history_signatures[key]:
                pairwise_metric_sign_flip_detected = True
                break
        if pairwise_metric_sign_flip_detected:
            break

    status = "stable"
    if any(task["failed_rows"] > 0 for task in tasks):
        status = "unstable"
    elif not current_verdict:
        status = "verdict_failed"
    elif (
        verdict_flip_detected
        or pairwise_metric_sign_flip_detected
        or bayesian_frequentist_mismatch_detected
    ):
        status = "unstable"

    return {
        "schema_version": "parity.stability.v1",
        "generated_at": _timestamp(),
        "input": str(runs_path.resolve()),
        "equivalence_report": str(equivalence_report_path.resolve()),
        "history_equivalence_reports": [str(path) for path in history_paths],
        "status": status,
        "global": {
            "parity_pass_final": current_verdict,
            "validity_pass": _report_validity(current_report),
            "missing_pairs": _report_missing_pairs(current_report),
            "task_count": len(tasks),
            "row_count": len(rows),
            "history_run_count": len(history_reports),
            "successful_system_rows": dict(sorted(system_success_counts.items())),
            "failed_system_rows": dict(sorted(system_failure_counts.items())),
        },
        "tasks": tasks,
        "rerun_consistency": {
            "mode": "multi_run" if history_reports else "single_run",
            "verdict_flip_detected": verdict_flip_detected,
            "pairwise_metric_sign_flip_detected": pairwise_metric_sign_flip_detected,
            "bayesian_frequentist_mismatch_detected": bayesian_frequentist_mismatch_detected,
        },
    }


def main() -> int:
    args = _parse_args()
    report = build_stability_report(
        runs_path=args.input,
        equivalence_report_path=args.equivalence_report,
        history_equivalence_report_paths=[Path(path) for path in args.history_equivalence_report],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
