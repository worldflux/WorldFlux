#!/usr/bin/env python3
"""Validate parity suite coverage policy and required-suite lock alignment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ALLOWED_STAGES = {"experimental", "required"}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be object: {path}")
    return payload


def _required_suites_from_policy(policy: dict[str, Any]) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    families = policy.get("families")
    if not isinstance(families, dict):
        return [], ["suite policy must define an object at `families`"]

    required: set[str] = set()
    for family, payload in families.items():
        if not isinstance(payload, dict):
            failures.append(f"family policy must be object: {family}")
            continue
        stage = str(payload.get("stage", "")).strip().lower()
        if stage not in ALLOWED_STAGES:
            failures.append(
                f"family {family} has invalid stage {stage!r}; "
                f"expected one of {sorted(ALLOWED_STAGES)}"
            )
            continue
        suites_raw = payload.get("suites", [])
        if not isinstance(suites_raw, list) or not suites_raw:
            failures.append(f"family {family} must define non-empty suites list")
            continue
        suites = [str(suite).strip() for suite in suites_raw if str(suite).strip()]
        if not suites:
            failures.append(f"family {family} suites list contains no valid suite IDs")
            continue
        if stage == "required":
            required.update(suites)

    return sorted(required), failures


def _validate_aggregate_for_required(
    aggregate: dict[str, Any],
    required_suites: list[str],
    *,
    enforce_pass: bool,
) -> list[str]:
    failures: list[str] = []
    suites_payload = aggregate.get("suites", [])
    if not isinstance(suites_payload, list):
        return ["aggregate.suites must be a list"]
    suite_rows = {
        str(row.get("suite_id", "")).strip(): row
        for row in suites_payload
        if isinstance(row, dict) and str(row.get("suite_id", "")).strip()
    }
    for suite_id in required_suites:
        row = suite_rows.get(suite_id)
        if row is None:
            failures.append(f"aggregate missing required suite row: {suite_id}")
            continue
        if enforce_pass and not bool(row.get("pass_non_inferiority", False)):
            failures.append(f"aggregate required suite failed non-inferiority: {suite_id}")
    if enforce_pass and not bool(aggregate.get("all_suites_pass", False)):
        failures.append("aggregate reports all_suites_pass=false")
    return failures


def validate_parity_suite_coverage(
    *,
    policy_path: Path,
    lock_path: Path,
    aggregate_path: Path | None,
    enforce_pass: bool,
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []

    if not policy_path.exists():
        failures.append(f"suite policy file is missing: {policy_path}")
        return failures, {"success": False, "failures": failures}
    if not lock_path.exists():
        failures.append(f"upstream lock file is missing: {lock_path}")
        return failures, {"success": False, "failures": failures}

    policy = _load_json(policy_path)
    lock = _load_json(lock_path)

    schema_version = str(policy.get("schema_version", "")).strip()
    if schema_version and schema_version != "worldflux.parity.policy.v1":
        failures.append(
            f"unsupported suite policy schema_version {schema_version!r}; "
            "expected 'worldflux.parity.policy.v1'"
        )

    required_suites, required_failures = _required_suites_from_policy(policy)
    failures.extend(required_failures)
    if not required_suites:
        failures.append("suite policy defines no required suites")

    lock_suites = lock.get("suites", {})
    if not isinstance(lock_suites, dict):
        failures.append("upstream lock must define object at suites")
        lock_suites = {}

    for suite_id in required_suites:
        if suite_id not in lock_suites:
            failures.append(f"required suite missing in upstream lock: {suite_id}")

    aggregate_checked = False
    if aggregate_path is not None:
        if not aggregate_path.exists():
            failures.append(f"aggregate artifact is missing: {aggregate_path}")
        else:
            aggregate_payload = _load_json(aggregate_path)
            aggregate_checked = True
            failures.extend(
                _validate_aggregate_for_required(
                    aggregate_payload,
                    required_suites,
                    enforce_pass=enforce_pass,
                )
            )

    report = {
        "schema_version": "worldflux.parity.policy.validation.v1",
        "policy_path": str(policy_path),
        "lock_path": str(lock_path),
        "aggregate_path": str(aggregate_path) if aggregate_path is not None else None,
        "required_suites": required_suites,
        "enforce_pass": enforce_pass,
        "aggregate_checked": aggregate_checked,
        "failures": failures,
        "success": len(failures) == 0,
    }
    return failures, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("reports/parity/suite_policy.json"),
        help="Parity suite policy path.",
    )
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("reports/parity/upstream_lock.json"),
        help="Parity upstream lock path.",
    )
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=None,
        help="Optional aggregate parity artifact path.",
    )
    parser.add_argument(
        "--enforce-pass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require required suites to pass in aggregate artifact when provided.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/parity/suite-policy-validation.json"),
        help="Validation summary output path.",
    )
    args = parser.parse_args()

    failures, report = validate_parity_suite_coverage(
        policy_path=args.policy,
        lock_path=args.lock,
        aggregate_path=args.aggregate,
        enforce_pass=bool(args.enforce_pass),
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if failures:
        print("[parity-suite-coverage] failed")
        for failure in failures:
            print(f"  - {failure}")
        print(f"[parity-suite-coverage] summary: {args.output_json}")
        return 1

    print("[parity-suite-coverage] passed")
    print(f"[parity-suite-coverage] summary: {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
