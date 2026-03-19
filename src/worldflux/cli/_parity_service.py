# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Execution services used by parity CLI commands."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from worldflux.parity import parse_seed_csv
from worldflux.parity.errors import ParityError


def resolve_parity_script_path(script_name: str) -> Path:
    candidates = (
        Path(__file__).resolve().parents[3] / "scripts" / "parity" / script_name,
        Path.cwd().resolve() / "scripts" / "parity" / script_name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ParityError(
        f"Unable to locate scripts/parity/{script_name}. "
        "Run this command from a WorldFlux source checkout."
    )


def run_parity_proof_script(script_name: str, args: list[str]) -> str:
    script_path = resolve_parity_script_path(script_name)
    completed = subprocess.run(
        [sys.executable, str(script_path), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        details = "\n".join(part for part in (stdout, stderr) if part)
        raise ParityError(
            f"Proof pipeline step failed ({script_name}, exit={completed.returncode}).\n{details}"
        )
    return stdout


def resolve_campaign_seeds(
    spec_default: tuple[int, ...], seeds_option: str | None
) -> tuple[int, ...]:
    parsed = parse_seed_csv(seeds_option)
    if parsed:
        return parsed
    if spec_default:
        return spec_default
    raise ParityError("No seeds provided. Pass --seeds or define campaign.default_seeds.")
