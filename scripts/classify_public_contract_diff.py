#!/usr/bin/env python3
"""Classify public contract snapshot diff as none/additive/breaking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from worldflux._internal.public_contract import classify_public_contract_diff


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Snapshot payload must be object: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True, help="Previous snapshot path.")
    parser.add_argument("--current", type=Path, required=True, help="Current snapshot path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for classification result.",
    )
    args = parser.parse_args()

    previous = _load_json(args.previous)
    current = _load_json(args.current)
    diff = classify_public_contract_diff(previous, current)

    result = {
        "classification": diff.classification,
        "additive_changes": list(diff.additive_changes),
        "breaking_reasons": list(diff.breaking_reasons),
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    print(f"[public-contract] classification: {diff.classification}")
    for change in diff.additive_changes:
        print(f"  + {change}")
    for reason in diff.breaking_reasons:
        print(f"  ! {reason}")

    return 1 if diff.classification == "breaking" else 0


if __name__ == "__main__":
    sys.exit(main())
