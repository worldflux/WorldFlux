#!/usr/bin/env python3
"""Run the repository release gates locally.

This script mirrors the public release workflow closely enough to catch
metadata drift, broken docs builds, missing parity artifacts, and packaging
regressions before a tag is published.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib


@dataclass(frozen=True)
class ReleaseCommand:
    """A single release validation command."""

    name: str
    argv: tuple[str, ...]


REPO_ROOT = Path(__file__).resolve().parents[1]


def _current_tag(repo_root: Path) -> str:
    payload = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    version = str(payload["project"]["version"]).strip()
    return f"v{version}"


def _twine_check_command() -> ReleaseCommand:
    command = (
        "import pathlib, subprocess, sys; "
        "files = sorted(str(p) for p in pathlib.Path('dist').glob('*')); "
        "sys.exit(subprocess.call(['twine', 'check', *files]) if files else 1)"
    )
    return ReleaseCommand(
        name="Check built artifacts",
        argv=("uv", "run", "--with", "twine", "python", "-c", command),
    )


def build_release_commands(*, tag: str, profile: str) -> tuple[ReleaseCommand, ...]:
    """Return the ordered release validation commands for a profile."""
    commands: list[ReleaseCommand] = [
        ReleaseCommand(
            name="Sync Python dependencies",
            argv=("uv", "sync", "--extra", "dev", "--extra", "training", "--extra", "cli"),
        ),
        ReleaseCommand(
            name="Install docs dependencies",
            argv=("npm", "--prefix", "website", "ci"),
        ),
        ReleaseCommand(
            name="Docs dependency audit",
            argv=("npm", "--prefix", "website", "audit", "--audit-level=high"),
        ),
        ReleaseCommand(
            name="Validate docs domain TLS and HTTPS",
            argv=(
                "uv",
                "run",
                "python",
                "scripts/check_docs_domain_tls.py",
                "--host",
                "worldflux.ai",
                "--url",
                "https://worldflux.ai/",
                "--expected-san",
                "worldflux.ai",
            ),
        ),
        ReleaseCommand(
            name="Validate release metadata",
            argv=("uv", "run", "python", "scripts/check_release_metadata.py", "--tag", tag),
        ),
        ReleaseCommand(
            name="Validate release checklist gate wiring",
            argv=("uv", "run", "python", "scripts/check_release_checklist_gate.py"),
        ),
        ReleaseCommand(
            name="Regenerate release parity fixtures",
            argv=("uv", "run", "python", "scripts/generate_release_parity_fixtures.py"),
        ),
        ReleaseCommand(
            name="Run public contract freeze and parity tests",
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
        ReleaseCommand(
            name="Validate fixed parity artifacts",
            argv=(
                "uv",
                "run",
                "python",
                "scripts/validate_parity_artifacts.py",
                "--run",
                "reports/parity/runs/dreamer_atari100k.json",
                "--run",
                "reports/parity/runs/tdmpc2_dmcontrol39.json",
                "--aggregate",
                "reports/parity/aggregate.json",
                "--lock",
                "reports/parity/upstream_lock.json",
                "--required-suite",
                "dreamer_atari100k",
                "--required-suite",
                "tdmpc2_dmcontrol39",
                "--max-missing-pairs",
                "0",
            ),
        ),
        ReleaseCommand(
            name="Validate parity suite policy coverage",
            argv=(
                "uv",
                "run",
                "python",
                "scripts/check_parity_suite_coverage.py",
                "--policy",
                "reports/parity/suite_policy.json",
                "--lock",
                "reports/parity/upstream_lock.json",
                "--aggregate",
                "reports/parity/aggregate.json",
                "--enforce-pass",
            ),
        ),
        ReleaseCommand(
            name="Build docs site",
            argv=("npm", "--prefix", "website", "run", "build"),
        ),
        ReleaseCommand(
            name="Run Bandit",
            argv=("uv", "run", "--with", "bandit", "bandit", "-r", "src/worldflux/", "-ll"),
        ),
        ReleaseCommand(
            name="Run pip-audit",
            argv=("uv", "run", "--with", "pip-audit", "pip-audit"),
        ),
    ]

    if profile == "full":
        commands.extend(
            [
                ReleaseCommand(
                    name="Build package",
                    argv=("uv", "run", "--with", "build", "python", "-m", "build"),
                ),
                _twine_check_command(),
            ]
        )

    return tuple(commands)


def _projectize_uv_command(argv: tuple[str, ...], *, repo_root: Path) -> tuple[str, ...]:
    if not argv or argv[0] != "uv" or "--project" in argv:
        return argv
    if len(argv) >= 2 and argv[1] in {"run", "sync"}:
        updated = list(argv)
        updated[2:2] = ["--project", str(repo_root)]
        return tuple(updated)
    return argv


def run_commands(
    commands: tuple[ReleaseCommand, ...],
    *,
    repo_root: Path,
    dry_run: bool,
) -> int:
    resolved_root = repo_root.resolve()
    for index, command in enumerate(commands, start=1):
        argv = _projectize_uv_command(command.argv, repo_root=resolved_root)
        print(f"[release-dry-run] ({index}/{len(commands)}) {command.name}")
        print(f"[release-dry-run] command: {shlex.join(argv)}")
        if dry_run:
            continue
        completed = subprocess.run(argv, cwd=str(resolved_root), check=False)
        if completed.returncode != 0:
            print(f"[release-dry-run] failed: {command.name} (exit {completed.returncode})")
            return 1
    print("[release-dry-run] passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tag",
        default=None,
        help="Release tag to validate. Defaults to the current pyproject version as vX.Y.Z.",
    )
    parser.add_argument(
        "--profile",
        choices=("verify", "full"),
        default="full",
        help="Run the workflow-equivalent verify gates or the full release dry-run including package build.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root used as the working directory and uv --project target.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    tag = args.tag or _current_tag(repo_root)
    commands = build_release_commands(tag=tag, profile=args.profile)
    return run_commands(commands, repo_root=repo_root, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
