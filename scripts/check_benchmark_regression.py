#!/usr/bin/env python3
"""Benchmark regression detection for CI.

Compares current benchmark results against baseline values.
Exit codes: 0 = pass (or warnings only), 1 = regression detected.

Baseline update policy: additive-only. Degradations require explicit approval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> dict[str, Any]:
    """Load a benchmark summary JSON file."""
    return json.loads(path.read_text())


def compute_regression(
    current: dict[str, Any],
    baseline: dict[str, Any],
    warn_threshold: float = 0.05,
    fail_threshold: float = 0.15,
) -> dict[str, Any]:
    """Compare metrics and detect regressions.

    Returns structured report with per-metric status.
    """
    results = {}
    current_metrics = current.get("metrics", {})
    baseline_metrics = baseline.get("metrics", {})

    for key in sorted(set(current_metrics) | set(baseline_metrics)):
        cur = current_metrics.get(key)
        base = baseline_metrics.get(key)

        if cur is None or base is None:
            results[key] = {"status": "skipped", "reason": "missing in current or baseline"}
            continue

        if base == 0:
            results[key] = {"status": "skipped", "reason": "baseline is zero"}
            continue

        relative_change = (cur - base) / abs(base)

        if relative_change >= 0:
            status = "improved"
        elif relative_change >= -warn_threshold:
            status = "stable"
        elif relative_change >= -fail_threshold:
            status = "warning"
        else:
            status = "regression"

        results[key] = {
            "current": cur,
            "baseline": base,
            "relative_change": round(relative_change, 6),
            "status": status,
        }

    has_regression = any(r.get("status") == "regression" for r in results.values())
    has_warning = any(r.get("status") == "warning" for r in results.values())

    return {
        "model_id": current.get("model_id", "unknown"),
        "verdict": "fail" if has_regression else ("warn" if has_warning else "pass"),
        "metrics": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark regression detection")
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Path to current summary JSON or directory of JSONs",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline summary JSON or directory of JSONs",
    )
    parser.add_argument(
        "--warn-threshold",
        type=float,
        default=0.05,
        help="Relative degradation for warning (default: 0.05)",
    )
    parser.add_argument(
        "--fail-threshold",
        type=float,
        default=0.15,
        help="Relative degradation for failure (default: 0.15)",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output JSON report path")
    args = parser.parse_args()

    # Handle directory or single file
    if args.current.is_dir():
        current_files = sorted(args.current.glob("*.json"))
    else:
        current_files = [args.current]

    if args.baseline.is_dir():
        baseline_files = {f.name: f for f in args.baseline.glob("*.json")}
    else:
        baseline_files = {args.baseline.name: args.baseline}

    reports = []
    overall_verdict = "pass"

    for cf in current_files:
        bf = baseline_files.get(cf.name)
        if bf is None:
            continue
        current_data = load_summary(cf)
        baseline_data = load_summary(bf)
        report = compute_regression(
            current_data, baseline_data, args.warn_threshold, args.fail_threshold
        )
        reports.append(report)
        if report["verdict"] == "fail":
            overall_verdict = "fail"
        elif report["verdict"] == "warn" and overall_verdict == "pass":
            overall_verdict = "warn"

    output = {"verdict": overall_verdict, "reports": reports}
    output_json = json.dumps(output, indent=2)
    print(output_json)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json)

    return 1 if overall_verdict == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
