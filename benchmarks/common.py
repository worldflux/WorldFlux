"""Shared utilities for WorldFlux benchmark scripts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

from worldflux.telemetry.wasr import make_run_id, write_event


def write_reward_heatmap_ppm(rewards, output_path: str | Path) -> Path:
    """Reuse examples/_shared/viz.py without requiring package installation."""
    viz_path = Path(__file__).resolve().parents[1] / "examples" / "_shared" / "viz.py"
    spec = importlib.util.spec_from_file_location("worldflux_examples_viz", viz_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load visualization helper from {viz_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    writer = getattr(module, "write_reward_heatmap_ppm")
    return writer(rewards, output_path)


def add_common_cli(parser: argparse.ArgumentParser, *, default_output_dir: str) -> None:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="Run short, CI-safe configuration")
    mode.add_argument("--full", action="store_true", help="Run full benchmark configuration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=default_output_dir)


def resolve_mode(args: argparse.Namespace) -> str:
    if bool(args.full):
        return "full"
    return "quick"


def build_run_context(*, scenario: str, mode: str) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "mode": mode,
        "run_id": make_run_id(),
        "started_at": time.time(),
    }


def emit_failure(
    context: dict[str, Any], *, error: str, artifacts: dict[str, str] | None = None
) -> None:
    write_event(
        event="run_complete",
        scenario=str(context["scenario"]),
        run_id=str(context["run_id"]),
        success=False,
        duration_sec=float(time.time() - float(context["started_at"])),
        ttfi_sec=0.0,
        artifacts=artifacts or {},
        error=error,
    )


def emit_success(
    context: dict[str, Any],
    *,
    ttfi_sec: float,
    artifacts: dict[str, str],
) -> None:
    write_event(
        event="run_complete",
        scenario=str(context["scenario"]),
        run_id=str(context["run_id"]),
        success=True,
        duration_sec=float(time.time() - float(context["started_at"])),
        ttfi_sec=float(ttfi_sec),
        artifacts=artifacts,
        error="",
    )


def write_summary(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
