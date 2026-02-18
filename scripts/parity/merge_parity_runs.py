#!/usr/bin/env python3
"""Merge shard-level parity run JSONL files into one canonical run log."""

from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help="Input JSONL path (repeatable).",
    )
    parser.add_argument(
        "--inputs-glob",
        type=str,
        default="",
        help="Optional glob pattern for input JSONL files.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    return parser.parse_args()


def _status_rank(status: str) -> int:
    if status == "success":
        return 3
    if status == "failed":
        return 2
    if status == "planned":
        return 1
    return 0


def _record_key(record: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(record.get("task_id", "")),
        int(record.get("seed", -1)),
        str(record.get("system", "")),
    )


def _record_priority(record: dict[str, Any]) -> tuple[int, int, str]:
    status = str(record.get("status", ""))
    attempt = int(record.get("attempt", 0) or 0)
    timestamp = str(record.get("timestamp", ""))
    return (_status_rank(status), attempt, timestamp)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _resolve_inputs(raw_inputs: list[Path], inputs_glob: str) -> list[Path]:
    inputs = [path.resolve() for path in raw_inputs]
    if inputs_glob.strip():
        inputs.extend(Path(path).resolve() for path in glob.glob(inputs_glob.strip()))
    unique = sorted({path.resolve() for path in inputs if path.exists()})
    return unique


def main() -> int:
    args = _parse_args()
    inputs = _resolve_inputs(args.input, args.inputs_glob)
    if not inputs:
        raise SystemExit("No input files provided. Use --input or --inputs-glob.")

    merged: dict[tuple[str, int, str], dict[str, Any]] = {}
    source_counts: dict[str, int] = {}

    for path in inputs:
        rows = _load_jsonl(path)
        source_counts[str(path)] = len(rows)
        for row in rows:
            key = _record_key(row)
            current = merged.get(key)
            if current is None or _record_priority(row) > _record_priority(current):
                merged[key] = row

    ordered = [merged[key] for key in sorted(merged)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    status_counts: dict[str, int] = {}
    for row in ordered:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1

    summary = {
        "schema_version": "parity.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(path) for path in inputs],
        "input_record_counts": source_counts,
        "output": str(args.output),
        "merged_records": len(ordered),
        "status_counts": status_counts,
    }

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
