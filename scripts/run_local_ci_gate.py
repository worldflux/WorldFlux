#!/usr/bin/env python3
"""Run local commands that mirror repository CI gates."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class GateCommand:
    """A single CI gate command."""

    name: str
    argv: tuple[str, ...]


def _is_docs_domain_reachable(timeout_seconds: float = 5.0) -> bool:
    request = urllib.request.Request("https://worldflux.ai/", method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds):
            return True
    except (urllib.error.URLError, TimeoutError):
        return False


def _lychee_command() -> GateCommand:
    args = [
        "uvx",
        "lychee",
        "--verbose",
        "--no-progress",
        "--max-concurrency",
        "8",
        "--max-retries",
        "2",
        "--retry-wait-time",
        "2",
        "--accept",
        "200,206,301,302,303,307,308,403,429",
        "README.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "SECURITY.md",
        "CODE_OF_CONDUCT.md",
        "docs",
        "spaces",
    ]
    if not _is_docs_domain_reachable():
        args.extend(["--exclude", r"worldflux\.ai"])
    return GateCommand(name="Check markdown links", argv=tuple(args))


def _twine_check_command() -> GateCommand:
    command = (
        "import pathlib, subprocess, sys; "
        "files = sorted(str(p) for p in pathlib.Path('dist').glob('*')); "
        "sys.exit(subprocess.call(['twine', 'check', *files]) if files else 1)"
    )
    return GateCommand(
        name="Check built artifacts",
        argv=("uv", "run", "--with", "twine", "python", "-c", command),
    )


def build_gate_commands() -> tuple[GateCommand, ...]:
    """Build an ordered list of local commands that mirror CI jobs."""
    return (
        GateCommand(
            name="Sync dependencies",
            argv=(
                "uv",
                "sync",
                "--extra",
                "dev",
                "--extra",
                "training",
                "--extra",
                "docs",
                "--extra",
                "cli",
            ),
        ),
        GateCommand(
            name="Ruff lint",
            argv=("uvx", "ruff", "check", "src/", "tests/", "examples/", "benchmarks/", "scripts/"),
        ),
        GateCommand(
            name="Ruff format check",
            argv=(
                "uvx",
                "ruff",
                "format",
                "--check",
                "src/",
                "tests/",
                "examples/",
                "benchmarks/",
                "scripts/",
            ),
        ),
        GateCommand(name="Mypy", argv=("uv", "run", "mypy", "src/worldflux/")),
        GateCommand(
            name="Pytest (3.10)",
            argv=(
                "uv",
                "run",
                "--python",
                "3.10",
                "--extra",
                "dev",
                "python",
                "-m",
                "pytest",
                "tests/",
                "-v",
                "--tb=short",
            ),
        ),
        GateCommand(
            name="Pytest (3.11)",
            argv=(
                "uv",
                "run",
                "--python",
                "3.11",
                "--extra",
                "dev",
                "python",
                "-m",
                "pytest",
                "tests/",
                "-v",
                "--tb=short",
            ),
        ),
        GateCommand(
            name="Pytest (3.12)",
            argv=(
                "uv",
                "run",
                "--python",
                "3.12",
                "--extra",
                "dev",
                "python",
                "-m",
                "pytest",
                "tests/",
                "-v",
                "--tb=short",
            ),
        ),
        GateCommand(
            name="Coverage report",
            argv=(
                "uv",
                "run",
                "--python",
                "3.11",
                "--extra",
                "dev",
                "python",
                "-m",
                "pytest",
                "tests/",
                "--cov=worldflux",
                "--cov-report=xml",
                "--cov-report=term-missing",
            ),
        ),
        GateCommand(
            name="Critical coverage thresholds",
            argv=(
                "uv",
                "run",
                "python",
                "scripts/check_critical_coverage.py",
                "--report",
                "coverage.xml",
            ),
        ),
        GateCommand(
            name="Dreamer example smoke",
            argv=("uv", "run", "python", "examples/train_dreamer.py", "--test"),
        ),
        GateCommand(
            name="TD-MPC2 example smoke",
            argv=("uv", "run", "python", "examples/train_tdmpc2.py", "--test"),
        ),
        GateCommand(
            name="CPU success example smoke",
            argv=("uv", "run", "python", "examples/quickstart_cpu_success.py", "--quick"),
        ),
        GateCommand(
            name="Unified comparison example smoke",
            argv=("uv", "run", "python", "examples/compare_unified_training.py", "--quick"),
        ),
        GateCommand(
            name="JEPA example smoke",
            argv=(
                "uv",
                "run",
                "python",
                "examples/train_jepa.py",
                "--steps",
                "5",
                "--batch-size",
                "4",
                "--obs-dim",
                "8",
            ),
        ),
        GateCommand(
            name="Token example smoke",
            argv=(
                "uv",
                "run",
                "python",
                "examples/train_token_model.py",
                "--steps",
                "5",
                "--batch-size",
                "4",
                "--seq-len",
                "8",
                "--vocab-size",
                "32",
            ),
        ),
        GateCommand(
            name="Diffusion example smoke",
            argv=(
                "uv",
                "run",
                "python",
                "examples/train_diffusion_model.py",
                "--steps",
                "5",
                "--batch-size",
                "4",
                "--obs-dim",
                "4",
                "--action-dim",
                "2",
            ),
        ),
        GateCommand(
            name="CEM planner example smoke",
            argv=(
                "uv",
                "run",
                "python",
                "examples/plan_cem.py",
                "--horizon",
                "3",
                "--action-dim",
                "2",
            ),
        ),
        GateCommand(
            name="Install minimal plugin example",
            argv=("uv", "pip", "install", "-e", "examples/plugins/minimal_plugin"),
        ),
        GateCommand(
            name="Minimal plugin smoke",
            argv=("uv", "run", "python", "examples/plugins/smoke_minimal_plugin.py"),
        ),
        GateCommand(
            name="Dreamer benchmark quick smoke",
            argv=(
                "uv",
                "run",
                "python",
                "benchmarks/benchmark_dreamerv3_atari.py",
                "--quick",
                "--seed",
                "42",
            ),
        ),
        GateCommand(
            name="TD-MPC2 benchmark quick smoke",
            argv=(
                "uv",
                "run",
                "python",
                "benchmarks/benchmark_tdmpc2_mujoco.py",
                "--quick",
                "--seed",
                "42",
            ),
        ),
        GateCommand(
            name="Diffusion benchmark quick smoke",
            argv=(
                "uv",
                "run",
                "python",
                "benchmarks/benchmark_diffusion_imagination.py",
                "--quick",
                "--seed",
                "42",
            ),
        ),
        GateCommand(
            name="Parity smoke",
            argv=(
                "uv",
                "run",
                "--extra",
                "dev",
                "python",
                "-m",
                "pytest",
                "-q",
                "tests/test_scripts/test_end_to_end_parity_smoke.py",
            ),
        ),
        GateCommand(
            name="Bandit security linter",
            argv=("uv", "run", "--with", "bandit", "bandit", "-r", "src/worldflux/", "-ll"),
        ),
        GateCommand(
            name="pip-audit",
            argv=("uv", "run", "--with", "pip-audit", "pip-audit"),
        ),
        GateCommand(
            name="API v0.2 tests",
            argv=(
                "uv",
                "run",
                "--extra",
                "dev",
                "python",
                "-m",
                "pytest",
                "-q",
                "tests/test_core/test_payloads.py",
                "tests/test_core/test_interfaces.py",
                "tests/test_core/test_contract_v2.py",
            ),
        ),
        GateCommand(
            name="Legacy bridge tests",
            argv=(
                "uv",
                "run",
                "--extra",
                "dev",
                "python",
                "-m",
                "pytest",
                "-q",
                "tests/test_integration/test_legacy_bridge_v02.py",
            ),
        ),
        GateCommand(
            name="Build docs (strict)",
            argv=("uv", "run", "--extra", "docs", "mkdocs", "build", "--strict"),
        ),
        _lychee_command(),
        GateCommand(
            name="Critical hardening tests",
            argv=(
                "uv",
                "run",
                "--extra",
                "dev",
                "python",
                "-m",
                "pytest",
                "-q",
                "tests/test_main_entrypoint.py",
                "tests/test_cli.py",
                "tests/test_samplers/test_token_sampler.py",
                "tests/test_samplers/test_diffusion_sampler.py",
                "tests/test_training.py",
                "tests/test_training/test_callbacks_validation.py",
                "tests/test_training/test_data_provider_guards.py",
                "tests/test_training/test_trainer_safety_guards.py",
                "tests/test_training/test_replay_buffer_schema_validation.py",
                "tests/test_utils.py",
                "tests/test_docs/test_docs_information_architecture.py",
                "--cov=worldflux",
                "--cov-report=xml",
                "--cov-report=term-missing",
            ),
        ),
        GateCommand(
            name="Planner boundary tests",
            argv=(
                "uv",
                "run",
                "--extra",
                "dev",
                "python",
                "-m",
                "pytest",
                "-q",
                "tests/test_planners/test_cem.py",
                "tests/test_integration/test_planner_dynamics_decoupling.py",
            ),
        ),
        GateCommand(
            name="Parity suite policy check",
            argv=(
                "uv",
                "run",
                "python",
                "scripts/check_parity_suite_coverage.py",
                "--policy",
                "reports/parity/suite_policy.json",
                "--lock",
                "reports/parity/upstream_lock.json",
            ),
        ),
        GateCommand(
            name="v3 migration checks",
            argv=(
                "uv",
                "run",
                "--extra",
                "dev",
                "python",
                "-m",
                "pytest",
                "-q",
                "tests/test_core/test_payloads.py::test_normalize_planned_action_missing_horizon_errors_in_v3",
                "tests/test_core/test_payloads.py::test_normalize_planned_action_legacy_sequence_key_errors_in_v3",
                "tests/test_integration/test_legacy_bridge_v02.py::test_planner_horizon_is_required_in_v3",
                "tests/test_integration/test_legacy_bridge_v02.py::test_planner_sequence_field_errors_in_v3",
                "tests/test_factory.py::TestCreateWorldModel::test_create_with_hybrid_action_rejected_in_v3",
            ),
        ),
        GateCommand(
            name="Update public contract snapshot",
            argv=(
                "uv",
                "run",
                "python",
                "scripts/update_public_contract_snapshot.py",
                "--snapshot",
                "tests/fixtures/public_contract_snapshot.json",
                "--result-json",
                "reports/public-contract/update-result.json",
            ),
        ),
        GateCommand(
            name="Public contract and parity tests",
            argv=(
                "uv",
                "run",
                "--extra",
                "dev",
                "python",
                "-m",
                "pytest",
                "-q",
                "tests/test_public_contract_freeze.py",
                "tests/test_parity/",
            ),
        ),
        GateCommand(
            name="Release checklist gate wiring",
            argv=("python", "scripts/check_release_checklist_gate.py"),
        ),
        GateCommand(
            name="Build package",
            argv=("uv", "run", "--with", "build", "python", "-m", "build"),
        ),
        _twine_check_command(),
    )


def _format_command(argv: tuple[str, ...]) -> str:
    return shlex.join(argv)


def run_commands(
    commands: tuple[GateCommand, ...],
    *,
    dry_run: bool = False,
    keep_going: bool = False,
) -> int:
    failures: list[tuple[str, int]] = []
    total = len(commands)
    for index, command in enumerate(commands, start=1):
        print(f"[ci-gate] ({index}/{total}) {command.name}")
        print(f"[ci-gate] command: {_format_command(command.argv)}")
        if dry_run:
            continue

        completed = subprocess.run(command.argv, check=False)
        if completed.returncode == 0:
            continue

        failures.append((command.name, completed.returncode))
        if not keep_going:
            break

    if failures:
        print("[ci-gate] failed")
        for name, returncode in failures:
            print(f"  - {name} (exit {returncode})")
        return 1

    print("[ci-gate] passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue running remaining commands after failures.",
    )
    args = parser.parse_args()

    commands = build_gate_commands()
    return run_commands(commands, dry_run=args.dry_run, keep_going=args.keep_going)


if __name__ == "__main__":
    sys.exit(main())
