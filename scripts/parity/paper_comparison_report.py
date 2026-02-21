#!/usr/bin/env python3
"""Generate a paper-baseline comparison markdown report from parity run JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from worldflux.parity.paper_comparison import compare_against_paper


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to parity run results JSON")
    parser.add_argument("--suite-id", required=True, help="Suite identifier to match baselines")
    parser.add_argument("--output", type=Path, required=True, help="Output markdown path")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        print(f"Error: input JSON must be an object: {args.input}", file=sys.stderr)
        return 1

    # Extract per-task mean scores from the pairs list
    pairs = payload.get("pairs", [])
    task_scores: dict[str, list[float]] = {}
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        task = pair.get("task")
        score = pair.get("worldflux_score")
        if task is not None and score is not None:
            task_scores.setdefault(str(task), []).append(float(score))

    run_scores: dict[str, float] = {
        task: sum(scores) / len(scores) for task, scores in task_scores.items()
    }

    report = compare_against_paper(args.suite_id, run_scores)
    if report is None:
        print(f"No baselines found for suite {args.suite_id!r}", file=sys.stderr)
        return 1

    markdown = report.render_markdown()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
