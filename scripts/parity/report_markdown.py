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


def _bayes_metric_row(task_id: str, metric_name: str, metric: dict[str, Any]) -> str:
    bayesian = metric.get("bayesian", {})
    if not isinstance(bayesian, dict) or bayesian.get("status") != "ok":
        status = "-"
        if isinstance(bayesian, dict):
            status = str(bayesian.get("status", "missing"))
        return f"| {task_id} | {metric_name} | - | - | - | {status} |"

    posterior_ci90 = bayesian.get("posterior_ci90", [None, None])
    posterior_ci90_str = f"[{_fmt_float(posterior_ci90[0])}, {_fmt_float(posterior_ci90[1])}]"
    return (
        f"| {task_id} | {metric_name} | "
        f"{_fmt_float(bayesian.get('p_equivalence'))} | "
        f"{_fmt_float(bayesian.get('p_noninferior'))} | "
        f"{posterior_ci90_str} | "
        f"{_fmt_bool(bayesian.get('pass_all'))} |"
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
        f"- All metrics parity (frequentist): **{_fmt_bool(global_block.get('parity_pass_all_metrics'))}**"
    )
    if "parity_pass_bayesian" in global_block:
        lines.append(
            f"- All metrics parity (Bayesian): **{_fmt_bool(global_block.get('parity_pass_bayesian'))}**"
        )
    if "parity_pass_frequentist" in global_block:
        lines.append(
            f"- Dual-pass frequentist gate: **{_fmt_bool(global_block.get('parity_pass_frequentist'))}**"
        )
    bayes_cfg = report.get("config", {}).get("bayesian", {})
    if isinstance(bayes_cfg, dict):
        lines.append(
            f"- Dual-pass required: `{bayes_cfg.get('dual_pass_required', False)}` "
            f"(Bayesian enabled: `{bayes_cfg.get('enabled', False)}`)"
        )
        if bayes_cfg.get("enabled", False):
            lines.append(
                "- Bayesian thresholds: "
                f"equivalence `{_fmt_float(bayes_cfg.get('probability_threshold_equivalence'))}`, "
                f"noninferiority `{_fmt_float(bayes_cfg.get('probability_threshold_noninferiority'))}`"
            )
            lines.append(
                f"- Bayesian draws/seed: `{bayes_cfg.get('draws', '-')}` / `{bayes_cfg.get('seed', '-')}`"
            )
    lines.append(
        f"- Final verdict (all gates): **{_fmt_bool(global_block.get('parity_pass_final'))}**"
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

    bayesian_block = report.get("bayesian", {})
    if isinstance(bayesian_block, dict):
        lines.append("## Bayesian Summary")
        lines.append("")
        lines.append(f"- Enabled: `{bayesian_block.get('enabled', False)}`")
        lines.append(
            f"- Task pass count: `{bayesian_block.get('tasks_pass_all_metrics', '-')}` "
            f"/ `{bayesian_block.get('tasks_total', '-')}`"
        )
        lines.append(
            f"- Bayesian all-metrics parity: **{_fmt_bool(bayesian_block.get('parity_pass_all_metrics'))}**"
        )
        posterior = bayesian_block.get("posterior_summary", {})
        if isinstance(posterior, dict) and posterior:
            lines.append(
                "- Posterior probability summary: "
                f"mean p(eq) `{_fmt_float(posterior.get('mean_p_equivalence'))}`, "
                f"min p(eq) `{_fmt_float(posterior.get('min_p_equivalence'))}`, "
                f"mean p(non-inf) `{_fmt_float(posterior.get('mean_p_noninferior'))}`, "
                f"min p(non-inf) `{_fmt_float(posterior.get('min_p_noninferior'))}`"
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
    if isinstance(bayesian_block, dict) and bool(bayesian_block.get("enabled", False)):
        lines.append("## Bayesian Per-Metric Results")
        lines.append("")
        lines.append("| Task | Metric | P(eq) | P(non-inf) | Posterior CI90 | Bayes Pass |")
        lines.append("|---|---|---:|---:|---|---:|")
        for task in report.get("tasks", []):
            task_id = str(task.get("task_id", "-"))
            metrics = task.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            for metric_name, metric in metrics.items():
                lines.append(_bayes_metric_row(task_id, metric_name, metric))
        lines.append("")

    lines.append("## Task Summary")
    lines.append("")
    lines.append("| Task | Primary Pass | All Metrics Pass | Bayesian Pass |")
    lines.append("|---|---:|---:|---:|")
    for task in report.get("tasks", []):
        lines.append(
            f"| {task.get('task_id', '-')} | {_fmt_bool(task.get('task_pass_primary'))} | "
            f"{_fmt_bool(task.get('task_pass_all_metrics'))} | "
            f"{_fmt_bool(task.get('task_pass_bayesian'))} |"
        )

    paper_cmp = report.get("paper_comparison")
    if isinstance(paper_cmp, dict):
        lines.append("")
        lines.append("## Paper Baseline Comparison")
        lines.append("")
        lines.append(f"Suite: `{paper_cmp.get('suite_id', '-')}`")
        lines.append("")
        lines.append("| Task | Paper Score | Run Score | Delta | Relative % |")
        lines.append("|---|---:|---:|---:|---:|")
        for delta in paper_cmp.get("deltas", []):
            if not isinstance(delta, dict):
                continue
            lines.append(
                f"| {delta.get('task', '-')} | {float(delta.get('paper_score', 0)):.1f} "
                f"| {float(delta.get('run_score', 0)):.1f} "
                f"| {float(delta.get('absolute_delta', 0)):+.1f} "
                f"| {float(delta.get('relative_delta_pct', 0)):+.1f}% |"
            )
        lines.append("")
        lines.append(
            f"Mean relative delta: **{float(paper_cmp.get('mean_relative_delta_pct', 0)):+.1f}%**"
        )
        lines.append(
            f"Tasks within 5%: **{paper_cmp.get('tasks_within_5pct', 0)}** "
            f"/ {len(paper_cmp.get('deltas', []))}"
        )

    lines.append("")
    lines.append("## Decision Rule")
    lines.append("")
    lines.append("- TOST on paired log-ratio with equivalence bounds ±5%.")
    lines.append("- One-sided non-inferiority bound at -5%.")
    lines.append("- Holm correction applied to multiple hypotheses.")
    lines.append("- Bayesian bootstrap estimates posterior P(eq) and P(non-inferior).")
    lines.append("- Validity gate must pass in proof mode (no mock/random shortcuts).")
    lines.append(
        "- Final pass requires frequentist pass, optional Bayesian pass (when dual-pass is enabled), "
        "zero missing pairs, and validity pass."
    )

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
