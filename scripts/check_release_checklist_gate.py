#!/usr/bin/env python3
"""Ensure CI captures the documented release checklist gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REQUIRED_CI_SNIPPETS: tuple[str, ...] = (
    "uvx ruff check src/ tests/ examples/ benchmarks/ scripts/",
    "uvx ruff format --check src/ tests/ examples/ benchmarks/ scripts/",
    "uv run mypy src/worldflux/",
    "uv run pytest tests/ -v --tb=short",
    "scripts/update_public_contract_snapshot.py",
    "examples/quickstart_cpu_success.py --quick",
    "examples/compare_unified_training.py --quick",
    "examples/train_dreamer.py --test",
    "examples/train_tdmpc2.py --test",
    "benchmarks/benchmark_dreamerv3_atari.py --quick --seed 42",
    "benchmarks/benchmark_tdmpc2_mujoco.py --quick --seed 42",
    "benchmarks/benchmark_diffusion_imagination.py --quick --seed 42",
    "uv run mkdocs build --strict",
    "bandit -r src/worldflux/ scripts/parity/ -ll",
    "pip-audit",
    "scripts/check_critical_coverage.py --report coverage.xml",
    "scripts/check_parity_suite_coverage.py",
)

REQUIRED_CHECKLIST_SNIPPETS: tuple[str, ...] = (
    "Version and tag are aligned",
    "release-ready notes for the tag",
    "Build succeeds",
    "scripts/validate_parity_artifacts.py",
    "scripts/check_docs_domain_tls.py",
    "additive-only",
    "suite_policy.json",
)

REQUIRED_RELEASE_SNIPPETS: tuple[str, ...] = (
    "scripts/validate_parity_artifacts.py",
    "scripts/check_docs_domain_tls.py",
    "reports/parity/runs/dreamer_atari100k.json",
    "reports/parity/runs/tdmpc2_dmcontrol39.json",
    "reports/parity/aggregate.json",
    "scripts/check_parity_suite_coverage.py",
    "reports/parity/suite_policy.json",
    "--check docs_domain_tls=pass",
    "--check parity_release_gate=pass",
    "--check parity_suite_policy=pass",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci", type=Path, default=Path(".github/workflows/ci.yml"))
    parser.add_argument(
        "--checklist",
        type=Path,
        default=Path("docs/reference/release-checklist.md"),
    )
    parser.add_argument(
        "--release-workflow",
        type=Path,
        default=Path(".github/workflows/release.yml"),
    )
    args = parser.parse_args()

    ci_text = args.ci.read_text(encoding="utf-8")
    checklist_text = args.checklist.read_text(encoding="utf-8")
    release_text = args.release_workflow.read_text(encoding="utf-8")

    failures: list[str] = []
    for snippet in REQUIRED_CI_SNIPPETS:
        if snippet not in ci_text:
            failures.append(f"CI missing required gate snippet: {snippet}")

    for snippet in REQUIRED_CHECKLIST_SNIPPETS:
        if snippet not in checklist_text:
            failures.append(f"Release checklist missing snippet: {snippet}")

    for snippet in REQUIRED_RELEASE_SNIPPETS:
        if snippet not in release_text:
            failures.append(f"Release workflow missing required gate snippet: {snippet}")

    if failures:
        print("[release-checklist-gate] failed")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("[release-checklist-gate] passed")
    print(f"  - validated CI workflow: {args.ci}")
    print(f"  - validated checklist doc: {args.checklist}")
    print(f"  - validated release workflow: {args.release_workflow}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
