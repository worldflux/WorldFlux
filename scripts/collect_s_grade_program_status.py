#!/usr/bin/env python3
"""Collect the current status of Phase 0 S-grade program artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_JSON = Path("reports/s-grade/program-status.json")
REQUIRED_ARTIFACTS: tuple[str, ...] = (
    "AGENTS.md",
    "docs/roadmap/2026-q2-worldflux-quality-program.md",
    "docs/tasks/README.md",
    "docs/prd/architecture-s-grade.md",
    "docs/prd/api-s-grade.md",
    "docs/prd/ml-correctness-s-grade.md",
    "docs/prd/code-quality-s-grade.md",
    "docs/prd/differentiation-s-grade.md",
    "docs/prd/production-maturity-s-grade.md",
    "docs/prd/oss-readiness-s-grade.md",
    "docs/prd/scalability-s-grade.md",
    "docs/superpowers/plans/2026-03-19-worldflux-s-grade-program.md",
)


def collect_status(
    repo_root: Path, required_paths: tuple[str, ...] | None = None
) -> dict[str, object]:
    """Return a JSON-serializable report for required program artifacts."""
    required_paths = REQUIRED_ARTIFACTS if required_paths is None else required_paths
    artifacts: list[dict[str, object]] = []
    present = 0

    for relative_path in required_paths:
        exists = (repo_root / relative_path).exists()
        if exists:
            present += 1
        artifacts.append({"path": relative_path, "present": exists})

    return {
        "repo_root": str(repo_root),
        "summary": {
            "required": len(required_paths),
            "present": present,
            "missing": len(required_paths) - present,
        },
        "artifacts": artifacts,
    }


def _write_report(output_json: Path, payload: dict[str, object]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    payload = collect_status(args.repo_root.resolve())
    _write_report(args.output_json, payload)

    summary = payload["summary"]
    print("[s-grade-collect] report written")
    print(f"  - repo_root: {payload['repo_root']}")
    print(f"  - required: {summary['required']}")
    print(f"  - present: {summary['present']}")
    print(f"  - missing: {summary['missing']}")
    print("  - mode: collect-only (never fails on missing artifacts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
