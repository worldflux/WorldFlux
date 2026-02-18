#!/usr/bin/env python3
"""Render a parity equivalence JSON report as Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _fmt_float(value: Any, digits: int = 4) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.{digits}f}"
    return "-"


def _fmt_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    return "-"


def _metric_row(task_id: str, metric_name: str, metric: dict[str, Any]) -> str:
    if metric.get("status") != "ok":
        return (
            f"| {task_id} | {metric_name} | - | - | - | - | - | - | - | - | "
            f"{metric.get('status', 'missing')} |"
        )

    ci_ratio = metric.get("ci90_ratio", [None, None])
    ci_ratio_str = f"[{_fmt_float(ci_ratio[0])}, {_fmt_float(ci_ratio[1])}]"

    holm_primary = metric.get("holm_primary") or {}
    holm_all = metric.get("holm_all_metrics") or {}

    return (
        f"| {task_id} | {metric_name} | {metric.get('n_pairs', '-')} | "
        f"{_fmt_float(metric.get('official_mean'))} | {_fmt_float(metric.get('worldflux_mean'))} | "
        f"{_fmt_float(metric.get('ratio_mean'))} | {ci_ratio_str} | "
        f"{_fmt_float(metric.get('tost', {}).get('p_value'))} | "
        f"{_fmt_float(holm_primary.get('adjusted_p'))} | "
        f"{_fmt_float(holm_all.get('adjusted_p'))} | "
        f"{_fmt_bool(metric.get('noninferiority', {}).get('pass_raw'))} |"
    )


def _render(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# WorldFlux Parity Equivalence Report")
    lines.append("")
    lines.append(f"- Schema: `{report.get('schema_version', '-')}`")
    lines.append(f"- Generated: `{report.get('generated_at', '-')}`")
    lines.append(f"- Input: `{report.get('input', '-')}`")
    lines.append("")

    global_block = report.get("global", {})
    lines.append("## Global Verdict")
    lines.append("")
    lines.append(
        f"- Primary metric parity: **{_fmt_bool(global_block.get('parity_pass_primary'))}**"
    )
    lines.append(
        f"- All metrics parity: **{_fmt_bool(global_block.get('parity_pass_all_metrics'))}**"
    )
    lines.append(
        f"- Final verdict (all-metrics + completeness): **{_fmt_bool(global_block.get('parity_pass_final'))}**"
    )
    lines.append(f"- Validity gate: **{_fmt_bool(global_block.get('validity_pass'))}**")
    lines.append(
        f"- Missing pairs: `{global_block.get('missing_pairs', '-')}` "
        f"(strict-failed: `{global_block.get('strict_mode_failed', '-')}`)"
    )
    lines.append(
        "- Task pass counts: "
        f"primary `{global_block.get('tasks_pass_primary', 0)}` / `{global_block.get('tasks_total', 0)}`, "
        f"all-metrics `{global_block.get('tasks_pass_all_metrics', 0)}` / `{global_block.get('tasks_total', 0)}`"
    )
    lines.append("")

    completeness = report.get("completeness", {})
    if isinstance(completeness, dict):
        lines.append("## Matrix Completeness")
        lines.append("")
        lines.append(
            f"- Expected pairs: `{completeness.get('expected_pairs', '-')}`, "
            f"missing: `{completeness.get('missing_pairs', '-')}`"
        )
        lines.append(
            f"- Task-seed pairs: `{completeness.get('task_seed_count', '-')}`, "
            f"complete task-seed pairs: `{completeness.get('complete_task_seed_pairs', '-')}`"
        )
        lines.append("")

    validity = report.get("validity", {})
    if isinstance(validity, dict):
        lines.append("## Validity Gate")
        lines.append("")
        lines.append(
            f"- Proof mode: `{validity.get('proof_mode', '-')}`, pass: **{_fmt_bool(validity.get('pass'))}**"
        )
        lines.append(
            f"- Required policy mode: `{validity.get('required_policy_mode', '-')}`, "
            f"issues: `{validity.get('issue_count', '-')}`"
        )
        lines.append("")

    lines.append("## Per-Metric Results")
    lines.append("")
    lines.append(
        "| Task | Metric | N pairs | OFF mean | WF mean | Ratio mean | CI90 ratio | "
        "TOST p | Holm p (primary) | Holm p (all) | Non-inf |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|")

    for task in report.get("tasks", []):
        task_id = str(task.get("task_id", "-"))
        metrics = task.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for metric_name, metric in metrics.items():
            lines.append(_metric_row(task_id, metric_name, metric))

    lines.append("")
    lines.append("## Task Summary")
    lines.append("")
    lines.append("| Task | Primary Pass | All Metrics Pass |")
    lines.append("|---|---:|---:|")
    for task in report.get("tasks", []):
        lines.append(
            f"| {task.get('task_id', '-')} | {_fmt_bool(task.get('task_pass_primary'))} | "
            f"{_fmt_bool(task.get('task_pass_all_metrics'))} |"
        )

    lines.append("")
    lines.append("## Decision Rule")
    lines.append("")
    lines.append("- TOST on paired log-ratio with equivalence bounds ±5%.")
    lines.append("- One-sided non-inferiority bound at -5%.")
    lines.append("- Holm correction applied to multiple hypotheses.")
    lines.append("- Validity gate must pass in proof mode (no mock/random shortcuts).")
    lines.append("- Final pass requires all-metrics pass, zero missing pairs, and validity pass.")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = _parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    markdown = _render(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
