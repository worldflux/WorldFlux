#!/usr/bin/env python3
"""Validate fixed parity artifacts for release gating."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_REQUIRED_SUITES = ("dreamer_atari100k", "tdmpc2_dmcontrol39")
DEFAULT_RUN_PATHS = {
    "dreamer_atari100k": Path("reports/parity/runs/dreamer_atari100k.json"),
    "tdmpc2_dmcontrol39": Path("reports/parity/runs/tdmpc2_dmcontrol39.json"),
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be object: {path}")
    return payload


def _expected_run_paths(required_suites: list[str], provided_run_paths: list[Path]) -> list[Path]:
    if provided_run_paths:
        return provided_run_paths
    paths: list[Path] = []
    for suite_id in required_suites:
        path = DEFAULT_RUN_PATHS.get(suite_id)
        if path is not None:
            paths.append(path)
    return paths


def _suite_missing_count(run_payload: dict[str, Any]) -> int:
    counts = run_payload.get("counts", {})
    if not isinstance(counts, dict):
        return 0
    upstream_only = int(counts.get("upstream_only", 0))
    worldflux_only = int(counts.get("worldflux_only", 0))
    return upstream_only + worldflux_only


def validate_parity_artifacts(
    *,
    run_paths: list[Path],
    aggregate_path: Path,
    lock_path: Path,
    required_suites: list[str],
    max_missing_pairs: int,
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []

    if max_missing_pairs < 0:
        failures.append("max_missing_pairs must be >= 0")

    lock_payload: dict[str, Any] | None = None
    if not lock_path.exists():
        failures.append(f"upstream lock file is missing: {lock_path}")
    else:
        lock_payload = _load_json(lock_path)

    run_by_suite: dict[str, dict[str, Any]] = {}
    for run_path in run_paths:
        if not run_path.exists():
            failures.append(f"run artifact is missing: {run_path}")
            continue

        run_payload = _load_json(run_path)
        suite = run_payload.get("suite")
        if not isinstance(suite, dict):
            failures.append(f"run artifact missing suite metadata: {run_path}")
            continue

        suite_id = str(suite.get("suite_id", "")).strip()
        if not suite_id:
            failures.append(f"run artifact suite_id is empty: {run_path}")
            continue
        if suite_id in run_by_suite:
            failures.append(f"duplicate suite artifact detected for {suite_id}: {run_path}")
            continue

        run_by_suite[suite_id] = run_payload

        stats = run_payload.get("stats", {})
        if not isinstance(stats, dict):
            failures.append(f"run artifact stats missing or invalid: {run_path}")
        elif not bool(stats.get("pass_non_inferiority", False)):
            failures.append(f"suite {suite_id} failed non-inferiority in {run_path}")

        if _suite_missing_count(run_payload) > max_missing_pairs:
            failures.append(
                f"suite {suite_id} exceeds missing-pair threshold: "
                f"{_suite_missing_count(run_payload)} > {max_missing_pairs}"
            )

        evaluation_manifest = run_payload.get("evaluation_manifest")
        if not isinstance(evaluation_manifest, dict):
            failures.append(f"suite {suite_id} missing evaluation_manifest")

        artifact_integrity = run_payload.get("artifact_integrity")
        if not isinstance(artifact_integrity, dict):
            failures.append(f"suite {suite_id} missing artifact_integrity")

        if lock_payload is not None:
            suites = lock_payload.get("suites", {})
            suite_lock = suites.get(suite_id) if isinstance(suites, dict) else None
            if not isinstance(suite_lock, dict):
                failures.append(f"suite {suite_id} not found in upstream lock")
            else:
                expected_commit = suite_lock.get("commit")
                source_commit = run_payload.get("sources", {}).get("upstream", {}).get("commit")
                if expected_commit is None or source_commit is None:
                    failures.append(f"suite {suite_id} has missing upstream commit metadata")
                elif str(expected_commit) != str(source_commit):
                    failures.append(
                        f"suite {suite_id} upstream commit mismatch: "
                        f"lock={expected_commit} run={source_commit}"
                    )

                suite_lock_ref = run_payload.get("suite_lock_ref")
                if not isinstance(suite_lock_ref, dict):
                    failures.append(f"suite {suite_id} missing suite_lock_ref")
                else:
                    matches_lock = suite_lock_ref.get("matches_lock")
                    if matches_lock is not True:
                        failures.append(f"suite {suite_id} suite_lock_ref.matches_lock is not true")

    missing_required = sorted(set(required_suites) - set(run_by_suite))
    for suite_id in missing_required:
        failures.append(f"required suite artifact missing: {suite_id}")

    if not aggregate_path.exists():
        failures.append(f"aggregate artifact is missing: {aggregate_path}")
        aggregate_payload = {}
    else:
        aggregate_payload = _load_json(aggregate_path)

    suites_payload = (
        aggregate_payload.get("suites", []) if isinstance(aggregate_payload, dict) else []
    )
    suite_rows = {
        str(row.get("suite_id", "")): row
        for row in suites_payload
        if isinstance(row, dict) and str(row.get("suite_id", "")).strip()
    }

    if not bool(aggregate_payload.get("all_suites_pass", False)):
        failures.append("aggregate artifact reports all_suites_pass=false")

    for suite_id in required_suites:
        row = suite_rows.get(suite_id)
        if not isinstance(row, dict):
            failures.append(f"aggregate missing suite row: {suite_id}")
            continue
        if not bool(row.get("pass_non_inferiority", False)):
            failures.append(f"aggregate suite failed non-inferiority: {suite_id}")

    report = {
        "required_suites": required_suites,
        "run_paths": [str(path) for path in run_paths],
        "aggregate_path": str(aggregate_path),
        "lock_path": str(lock_path),
        "failures": failures,
        "success": len(failures) == 0,
    }
    return failures, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        type=Path,
        default=[],
        help="Run artifact path (repeatable). If omitted, fixed required paths are used.",
    )
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=Path("reports/parity/aggregate.json"),
        help="Aggregate parity artifact path.",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("reports/parity/upstream_lock.json"),
        help="Upstream lock file path.",
    )
    parser.add_argument(
        "--required-suite",
        action="append",
        default=[],
        help="Required suite id (repeatable).",
    )
    parser.add_argument(
        "--max-missing-pairs",
        type=int,
        default=0,
        help="Allowed missing pairs per suite (upstream_only + worldflux_only).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/parity/validation-summary.json"),
        help="Validation summary output path.",
    )
    args = parser.parse_args()

    required_suites = args.required_suite or list(DEFAULT_REQUIRED_SUITES)
    run_paths = _expected_run_paths(required_suites, args.run)

    failures, report = validate_parity_artifacts(
        run_paths=run_paths,
        aggregate_path=args.aggregate,
        lock_path=args.lock,
        required_suites=required_suites,
        max_missing_pairs=args.max_missing_pairs,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if failures:
        print("[parity-validation] failed")
        for failure in failures:
            print(f"  - {failure}")
        print(f"[parity-validation] summary: {args.output_json}")
        return 1

    print("[parity-validation] passed")
    print(f"[parity-validation] summary: {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
