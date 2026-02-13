#!/usr/bin/env python3
"""Export current public contract snapshot to JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from worldflux._internal.public_contract import capture_public_contract_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/public_contract_snapshot.json"),
        help="Output JSON path.",
    )
    args = parser.parse_args()

    payload = capture_public_contract_snapshot()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[public-contract] wrote snapshot: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
