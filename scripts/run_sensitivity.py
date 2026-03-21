#!/usr/bin/env python3
"""Launch hyperparameter sensitivity analysis for DreamerV3.

Generates one-at-a-time sweep configurations and can execute them through the
Dreamer native runner. Produces a JSON report and Markdown summary.

Usage:
    # List all configurations without running
    python scripts/run_sensitivity.py --dry-run

    # Execute a stub-backed campaign
    python scripts/run_sensitivity.py \
        --task-id atari100k_pong \
        --env-backend stub \
        --model-profile wf12m \
        --seeds 0 \
        --steps 12 \
        --output reports/parity/sensitivity/dreamerv3_sensitivity.json

    # Generate Markdown report from existing JSON
    python scripts/run_sensitivity.py \
        --report-from reports/parity/sensitivity/dreamerv3_sensitivity.json \
        --output-md docs/reference/hyperparameter-sensitivity.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from worldflux.parity.sensitivity import (  # noqa: E402
    DREAMERV3_SWEEPS,
    SensitivityAnalysis,
    SensitivityReport,
    render_sensitivity_markdown,
)
from worldflux.parity.sensitivity_runner import (  # noqa: E402
    DEFAULT_ENV_BACKEND,
    DEFAULT_MODEL_PROFILE,
    DEFAULT_RUN_ROOT,
    DEFAULT_TASK_ID,
    run_sensitivity_campaign,
)

SMOKE_LANE = "smoke"
PUBLISH_LANE = "publish"
DEFAULT_TRACKED_REPORT = Path("reports/parity/sensitivity/dreamerv3_sensitivity.json")
DEFAULT_TRACKED_DOC = Path("docs/reference/hyperparameter-sensitivity.md")
CANONICAL_PUBLISH_SEEDS = [0, 1, 2]
CANONICAL_PUBLISH_STEPS = 200
CANONICAL_PUBLISH_MODEL_PROFILE = "wf12m"
CANONICAL_PUBLISH_ENV_BACKEND = "gymnasium"


def _parse_seeds(raw: str | None) -> list[int]:
    if raw is None:
        return [0, 1, 2]
    return sorted({int(s.strip()) for s in raw.split(",") if s.strip()})


def _report_is_degenerate(report: SensitivityReport) -> bool:
    values: list[float] = []
    for param in report.parameters:
        values.extend(float(value) for value in param.mean_rewards)
    if not values:
        return True
    first = values[0]
    return all(abs(value - first) <= 1e-12 for value in values)


def _validate_lane_policy(
    *,
    lane: str,
    task_id: str,
    env_backend: str,
    model_profile: str,
    seeds: list[int],
    steps: int,
    output: Path,
    output_md: Path | None,
) -> str | None:
    if lane == SMOKE_LANE:
        if output == DEFAULT_TRACKED_REPORT:
            return "tracked sensitivity JSON is reserved for publish lane."
        if output_md == DEFAULT_TRACKED_DOC:
            return "tracked sensitivity Markdown is reserved for publish lane."
        return None

    if env_backend == "stub":
        return "publish lane requires a real environment backend."
    if env_backend != CANONICAL_PUBLISH_ENV_BACKEND:
        return f"publish lane requires env_backend='{CANONICAL_PUBLISH_ENV_BACKEND}'."
    if model_profile != CANONICAL_PUBLISH_MODEL_PROFILE:
        return f"publish lane requires model_profile='{CANONICAL_PUBLISH_MODEL_PROFILE}'."
    if seeds != CANONICAL_PUBLISH_SEEDS:
        return (
            "publish lane requires canonical seeds "
            f"{','.join(str(seed) for seed in CANONICAL_PUBLISH_SEEDS)}."
        )
    if steps != CANONICAL_PUBLISH_STEPS:
        return f"publish lane requires steps={CANONICAL_PUBLISH_STEPS}."
    if task_id != DEFAULT_TASK_ID:
        return f"publish lane requires task_id='{DEFAULT_TASK_ID}'."
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lane",
        default=SMOKE_LANE,
        choices=[SMOKE_LANE, PUBLISH_LANE],
        help="Execution lane: smoke or publish.",
    )
    parser.add_argument(
        "--task-id",
        default=DEFAULT_TASK_ID,
        help="Canonical Dreamer task id (default: atari100k_pong).",
    )
    parser.add_argument(
        "--environment",
        default=None,
        help="Deprecated alias for --task-id.",
    )
    parser.add_argument(
        "--env-backend",
        default=DEFAULT_ENV_BACKEND,
        choices=["stub", "gymnasium"],
        help="Environment backend for the Dreamer native runner.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Execution device for the Dreamer native runner.",
    )
    parser.add_argument(
        "--model-profile",
        default=DEFAULT_MODEL_PROFILE,
        help="Dreamer native model profile (default: wf12m).",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds (default: 0,1,2).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
        help="Environment steps per run (default: 100000).",
    )
    parser.add_argument(
        "--family",
        default="dreamerv3",
        help="Model family (default: dreamerv3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/sensitivity/dreamerv3_sensitivity.json"),
        help="Path for aggregated JSON output.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Path for Markdown output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print configurations without executing.",
    )
    parser.add_argument(
        "--report-from",
        type=Path,
        default=None,
        help="Generate report from existing JSON results instead of running.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
        help="Root directory for per-run artifacts.",
    )

    args = parser.parse_args(argv)
    task_id = str(args.environment).strip() or str(args.task_id).strip()
    if str(args.task_id).strip():
        task_id = str(args.task_id).strip()

    seeds = _parse_seeds(args.seeds)

    # Report-from mode: just render existing results
    if args.report_from is not None:
        raw = json.loads(args.report_from.read_text(encoding="utf-8"))
        report = SensitivityReport.from_json_payload(raw)
        md = render_sensitivity_markdown(report)
        if args.output_md:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            args.output_md.write_text(md, encoding="utf-8")
            print(f"[sensitivity] Markdown written to {args.output_md}")
        else:
            print(md)
        return 0

    lane_policy_error = _validate_lane_policy(
        lane=str(args.lane),
        task_id=task_id,
        env_backend=str(args.env_backend),
        model_profile=str(args.model_profile),
        seeds=seeds,
        steps=int(args.steps),
        output=args.output,
        output_md=args.output_md,
    )
    if lane_policy_error is not None:
        print(f"[sensitivity] {lane_policy_error}", file=sys.stderr)
        return 2

    analysis = SensitivityAnalysis(
        sweeps=DREAMERV3_SWEEPS,
        family=args.family,
        environment=task_id,
        seeds=seeds,
        total_steps=args.steps,
    )

    configs = analysis.generate_run_configs()
    print(f"[sensitivity] Total configurations: {len(configs)}")
    print(f"[sensitivity] Parameters: {[s.name for s in analysis.sweeps]}")
    print(f"[sensitivity] Seeds: {seeds}")
    print(f"[sensitivity] Steps: {args.steps}")
    print(f"[sensitivity] Task: {task_id}")
    print(f"[sensitivity] Backend: {args.env_backend}")
    print(f"[sensitivity] Model profile: {args.model_profile}")
    print(f"[sensitivity] Lane: {args.lane}")

    if args.dry_run:
        print("\n[dry-run] Configurations:")
        for i, cfg in enumerate(configs):
            print(f"  [{i+1:3d}] {cfg['param_name']}={cfg['param_value']} " f"seed={cfg['seed']}")
        print(f"\n[dry-run] Would run {len(configs)} experiments. Exiting.")
        return 0

    def _progress(index: int, total: int, config: dict[str, object]) -> None:
        print(
            "[sensitivity] "
            f"[{index}/{total}] {config['param_name']}={config['param_value']} seed={config['seed']}"
        )

    report = run_sensitivity_campaign(
        analysis=analysis,
        lane=str(args.lane),
        task_id=task_id,
        env_backend=str(args.env_backend),
        device=str(args.device),
        model_profile=str(args.model_profile),
        run_root=args.run_root,
        progress_callback=_progress,
    )
    if str(args.lane) == PUBLISH_LANE and _report_is_degenerate(report):
        print(
            "[sensitivity] publish lane produced a degenerate report; refusing to overwrite tracked artifacts.",
            file=sys.stderr,
        )
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_json_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[sensitivity] Aggregated report written to {args.output}")

    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_sensitivity_markdown(report), encoding="utf-8")
        print(f"[sensitivity] Markdown written to {args.output_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
