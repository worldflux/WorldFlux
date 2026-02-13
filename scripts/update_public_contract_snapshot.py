#!/usr/bin/env python3
"""Update public contract snapshot in additive mode and block breaking changes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from worldflux._internal.public_contract import (
    capture_public_contract_snapshot,
    classify_public_contract_diff,
)


def _load_snapshot(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Snapshot payload must be object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("tests/fixtures/public_contract_snapshot.json"),
        help="Snapshot path to compare and update.",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=Path("reports/public-contract/update-result.json"),
        help="Result payload path for CI consumption.",
    )
    args = parser.parse_args()

    previous = _load_snapshot(args.snapshot)
    current = capture_public_contract_snapshot()
    diff = classify_public_contract_diff(previous, current)

    updated = False
    if diff.classification == "additive":
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        updated = True

    result = {
        "classification": diff.classification,
        "updated": updated,
        "snapshot": str(args.snapshot),
        "additive_changes": list(diff.additive_changes),
        "breaking_reasons": list(diff.breaking_reasons),
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"[public-contract] classification: {diff.classification}")
    if updated:
        print(f"[public-contract] snapshot updated: {args.snapshot}")
    for change in diff.additive_changes:
        print(f"  + {change}")
    for reason in diff.breaking_reasons:
        print(f"  ! {reason}")

    return 1 if diff.classification == "breaking" else 0


if __name__ == "__main__":
    sys.exit(main())
