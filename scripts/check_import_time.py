"""Measure import latency for a Python statement and enforce a maximum budget."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from typing import Any


def _measure_statement(statement: str) -> float:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, time; "
                "start = time.perf_counter(); "
                f"{statement}; "
                "print(json.dumps({'seconds': time.perf_counter() - start}))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(
            f"Import statement failed (exit={completed.returncode}): {statement}\n{stderr}"
        )

    payload = json.loads(completed.stdout)
    return float(payload["seconds"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statement", required=True, help="Python statement to measure.")
    parser.add_argument("--label", default="", help="Display label for logs.")
    parser.add_argument(
        "--max-seconds",
        type=float,
        required=True,
        help="Maximum allowed median import time in seconds.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=3,
        help="Number of isolated measurements to run (default: 3).",
    )
    args = parser.parse_args()

    label = args.label or args.statement
    repeats = max(1, int(args.repeat))
    samples = [_measure_statement(args.statement) for _ in range(repeats)]
    median = statistics.median(samples)

    payload: dict[str, Any] = {
        "label": label,
        "statement": args.statement,
        "samples_sec": samples,
        "median_sec": median,
        "max_sec": float(args.max_seconds),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if median > args.max_seconds:
        print(
            f"[import-time] FAIL {label}: median {median:.4f}s exceeds budget {args.max_seconds:.4f}s",
            file=sys.stderr,
        )
        return 1

    print(f"[import-time] PASS {label}: median {median:.4f}s within budget {args.max_seconds:.4f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
