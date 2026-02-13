#!/usr/bin/env python3
"""Generate machine + markdown release verification report artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATUS_VALUES = {"pass", "fail", "skipped"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_check(raw: str) -> dict[str, str]:
    if "=" not in raw:
        raise ValueError(f"Invalid --check entry {raw!r}; expected name=status")
    name, status = raw.split("=", 1)
    normalized_name = name.strip()
    normalized_status = status.strip().lower()
    if not normalized_name:
        raise ValueError("Check name must not be empty")
    if normalized_status not in STATUS_VALUES:
        raise ValueError(f"Check status must be one of {sorted(STATUS_VALUES)}, got {status!r}")
    return {"name": normalized_name, "status": normalized_status}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def _render_markdown(payload: dict[str, Any]) -> str:
    checks = payload.get("checks", [])
    summary = payload.get("summary", {})
    parity = payload.get("parity", {})

    lines = [
        "# Release Verification Report",
        "",
        f"Generated at: {payload.get('generated_at_utc', 'unknown')}",
        "",
        "## Check Summary",
        "",
        f"- pass: {summary.get('pass', 0)}",
        f"- fail: {summary.get('fail', 0)}",
        f"- skipped: {summary.get('skipped', 0)}",
        "",
        "## Checks",
        "",
        "| Check | Status |",
        "| --- | --- |",
    ]

    for check in checks:
        if not isinstance(check, dict):
            continue
        lines.append(f"| {check.get('name', 'unknown')} | {check.get('status', 'unknown')} |")

    lines.extend(
        [
            "",
            "## Parity",
            "",
            f"- aggregate_present: {parity.get('aggregate_present', False)}",
            f"- all_suites_pass: {parity.get('all_suites_pass', 'unknown')}",
            f"- suite_fail_count: {parity.get('suite_fail_count', 'unknown')}",
        ]
    )
    suites = parity.get("suites", [])
    if isinstance(suites, list) and suites:
        lines.extend(
            [
                "",
                "| Suite | Samples | Upper CI | Margin | Pass | Reason |",
                "| --- | ---: | ---: | ---: | --- | --- |",
            ]
        )
        for suite in suites:
            if not isinstance(suite, dict):
                continue
            lines.append(
                "| {suite_id} | {sample_size} | {ci_upper_ratio:.6f} | {margin_ratio:.6f} | {passed} | {reason} |".format(
                    suite_id=suite.get("suite_id", "unknown"),
                    sample_size=int(suite.get("sample_size", 0)),
                    ci_upper_ratio=float(suite.get("ci_upper_ratio", 0.0)),
                    margin_ratio=float(suite.get("margin_ratio", 0.0)),
                    passed=bool(suite.get("pass_non_inferiority", False)),
                    reason=str(suite.get("verdict_reason", "")).replace("|", "\\|"),
                )
            )

    lock = payload.get("upstream_lock")
    if isinstance(lock, dict):
        suite_count = len(lock.get("suites", {})) if isinstance(lock.get("suites"), dict) else 0
        lines.extend(
            [
                "",
                "## Upstream Lock",
                "",
                f"- schema_version: {lock.get('schema_version', 'unknown')}",
                f"- suite_count: {suite_count}",
            ]
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="append",
        default=[],
        help="Verification check entry in name=status format (status: pass|fail|skipped)",
    )
    parser.add_argument(
        "--parity-aggregate",
        type=Path,
        default=Path("reports/parity/aggregate.json"),
        help="Path to aggregate parity artifact (optional).",
    )
    parser.add_argument(
        "--upstream-lock",
        type=Path,
        default=Path("reports/parity/upstream_lock.json"),
        help="Path to upstream lock file (optional).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/release/verification-report.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/release/verification-report.md"),
    )
    args = parser.parse_args()

    checks: list[dict[str, str]] = []
    for raw in args.check:
        try:
            checks.append(_parse_check(raw))
        except ValueError as exc:
            print(f"[verification-report] {exc}")
            return 1

    counter = Counter(check["status"] for check in checks)

    parity_payload = _load_json(args.parity_aggregate)
    parity_suites: list[dict[str, Any]] = []
    if parity_payload is not None:
        suites_payload = parity_payload.get("suites", [])
        if isinstance(suites_payload, list):
            for suite in suites_payload:
                if not isinstance(suite, dict):
                    continue
                parity_suites.append(
                    {
                        "suite_id": str(suite.get("suite_id", "unknown")),
                        "sample_size": int(suite.get("sample_size", 0)),
                        "ci_upper_ratio": float(suite.get("ci_upper_ratio", 0.0)),
                        "margin_ratio": float(suite.get("margin_ratio", 0.0)),
                        "pass_non_inferiority": bool(suite.get("pass_non_inferiority", False)),
                        "verdict_reason": str(suite.get("verdict_reason", "")),
                    }
                )
            parity_suites.sort(key=lambda row: row["suite_id"])
    parity = {
        "aggregate_present": parity_payload is not None,
        "all_suites_pass": (
            bool(parity_payload.get("all_suites_pass", False))
            if parity_payload is not None
            else None
        ),
        "suite_fail_count": (
            int(parity_payload.get("suite_fail_count", 0)) if parity_payload is not None else None
        ),
        "path": str(args.parity_aggregate),
        "suites": parity_suites,
    }

    lock_payload = _load_json(args.upstream_lock)

    report = {
        "schema_version": "worldflux.verification.v1",
        "generated_at_utc": _utc_now_iso(),
        "checks": checks,
        "summary": {
            "pass": int(counter.get("pass", 0)),
            "fail": int(counter.get("fail", 0)),
            "skipped": int(counter.get("skipped", 0)),
        },
        "parity": parity,
        "upstream_lock": lock_payload,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_md.write_text(_render_markdown(report), encoding="utf-8")

    print(f"[verification-report] wrote {args.output_json}")
    print(f"[verification-report] wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
