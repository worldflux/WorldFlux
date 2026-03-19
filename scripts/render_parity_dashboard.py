#!/usr/bin/env python3
"""Render a lightweight HTML dashboard from a parity aggregate report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _load_aggregate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Aggregate payload must be an object: {path}")
    return payload


def _suite_rows(suites: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for suite in suites:
        suite_id = html.escape(str(suite.get("suite_id", "unknown")))
        family = html.escape(str(suite.get("family", "unknown")))
        passed = bool(suite.get("pass_non_inferiority", False))
        status = "PASS" if passed else "FAIL"
        ci_upper = html.escape(str(suite.get("ci_upper_ratio", "")))
        margin = html.escape(str(suite.get("margin_ratio", "")))
        verdict = html.escape(str(suite.get("verdict_reason", "")))
        rows.append(
            "<tr>"
            f"<td>{suite_id}</td>"
            f"<td>{family}</td>"
            f"<td>{status}</td>"
            f"<td>{ci_upper}</td>"
            f"<td>{margin}</td>"
            f"<td>{verdict}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def render_dashboard(aggregate_path: Path, *, output_path: Path) -> Path:
    aggregate = _load_aggregate(aggregate_path)
    suites = aggregate.get("suites", [])
    if not isinstance(suites, list):
        raise ValueError("aggregate.suites must be a list")

    generated_at = html.escape(str(aggregate.get("generated_at_utc", "unknown")))
    overall = "PASS" if bool(aggregate.get("all_suites_pass", False)) else "FAIL"
    suite_pass_count = html.escape(str(aggregate.get("suite_pass_count", 0)))
    suite_fail_count = html.escape(str(aggregate.get("suite_fail_count", 0)))

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>WorldFlux Parity Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d0d7de; padding: 0.5rem; text-align: left; }}
    th {{ background: #f6f8fa; }}
  </style>
</head>
<body>
  <h1>WorldFlux Parity Dashboard</h1>
  <p>Generated at: {generated_at}</p>
  <p>Overall status: <strong>{overall}</strong></p>
  <p>Suites: pass={suite_pass_count}, fail={suite_fail_count}</p>
  <table>
    <thead>
      <tr>
        <th>Suite</th>
        <th>Family</th>
        <th>Status</th>
        <th>CI Upper</th>
        <th>Margin</th>
        <th>Verdict</th>
      </tr>
    </thead>
    <tbody>
      {_suite_rows(suites)}
    </tbody>
  </table>
</body>
</html>
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(document, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("aggregate", type=Path, help="Path to aggregate parity JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/parity/dashboard.html"),
        help="Output HTML path",
    )
    args = parser.parse_args()
    render_dashboard(args.aggregate, output_path=args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
