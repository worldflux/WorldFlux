#!/usr/bin/env python3
"""Compute statistical equivalence between official and WorldFlux parity runs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any


@dataclass(frozen=True)
class PairSample:
    seed: int
    official: float
    worldflux: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=str, default="final_return_mean,auc_return")
    parser.add_argument("--primary-metric", type=str, default="final_return_mean")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--equivalence-margin", type=float, default=0.05)
    parser.add_argument("--noninferiority-margin", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--min-pairs", type=int, default=2)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                entries.append(parsed)
    return entries


def _collect_pairs(
    entries: list[dict[str, Any]],
    *,
    metric: str,
) -> dict[str, list[PairSample]]:
    by_key: dict[tuple[str, int], dict[str, float]] = {}
    for entry in entries:
        if entry.get("status") != "success":
            continue
        task_id = str(entry.get("task_id", ""))
        system = str(entry.get("system", ""))
        seed = int(entry.get("seed", -1))
        metrics = entry.get("metrics", {})
        if system not in {"official", "worldflux"}:
            continue
        if not isinstance(metrics, dict):
            continue
        value = metrics.get(metric)
        if not isinstance(value, int | float):
            continue
        pair = by_key.setdefault((task_id, seed), {})
        pair[system] = float(value)

    grouped: dict[str, list[PairSample]] = defaultdict(list)
    for (task_id, seed), payload in by_key.items():
        if "official" not in payload or "worldflux" not in payload:
            continue
        grouped[task_id].append(
            PairSample(seed=seed, official=payload["official"], worldflux=payload["worldflux"])
        )

    return {task_id: sorted(samples, key=lambda s: s.seed) for task_id, samples in grouped.items()}


def _sample_mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = _sample_mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(max(0.0, var))


def _paired_log_ratios(samples: list[PairSample], eps: float) -> list[float]:
    ratios: list[float] = []
    for sample in samples:
        floor = min(sample.official, sample.worldflux)
        shift = (-floor + eps) if floor <= 0 else 0.0
        numerator = sample.worldflux + shift + eps
        denominator = sample.official + shift + eps
        ratios.append(math.log(numerator / denominator))
    return ratios


def _z_value(alpha: float) -> float:
    return NormalDist().inv_cdf(1.0 - alpha)


def _one_sided_p_lower(mean: float, se: float, lower: float) -> float:
    """H0: mean <= lower, H1: mean > lower."""
    if se <= 0:
        return 0.0 if mean > lower else 1.0
    z = (mean - lower) / se
    return 1.0 - NormalDist().cdf(z)


def _one_sided_p_upper(mean: float, se: float, upper: float) -> float:
    """H0: mean >= upper, H1: mean < upper."""
    if se <= 0:
        return 0.0 if mean < upper else 1.0
    z = (mean - upper) / se
    return NormalDist().cdf(z)


def _holm_adjustments(
    p_values: dict[str, float], alpha: float
) -> dict[str, dict[str, float | bool]]:
    if not p_values:
        return {}

    ranked = sorted(p_values.items(), key=lambda item: item[1])
    m = len(ranked)

    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (hypothesis_id, p_value) in enumerate(ranked):
        factor = m - rank
        running_max = max(running_max, min(1.0, p_value * factor))
        adjusted[hypothesis_id] = running_max

    rejected: dict[str, bool] = {}
    active = True
    for rank, (hypothesis_id, p_value) in enumerate(ranked):
        threshold = alpha / (m - rank)
        passed = bool(active and (p_value <= threshold))
        rejected[hypothesis_id] = passed
        if not passed:
            active = False

    return {
        hid: {
            "raw_p": p_values[hid],
            "adjusted_p": adjusted[hid],
            "pass": rejected[hid],
        }
        for hid in p_values
    }


def _metric_report(
    *,
    task_id: str,
    metric: str,
    samples: list[PairSample],
    alpha: float,
    eps: float,
    equivalence_margin: float,
    noninferiority_margin: float,
    min_pairs: int,
) -> dict[str, Any]:
    n_pairs = len(samples)
    official_values = [s.official for s in samples]
    worldflux_values = [s.worldflux for s in samples]

    if n_pairs < min_pairs:
        return {
            "task_id": task_id,
            "metric": metric,
            "n_pairs": n_pairs,
            "seed_ids": [s.seed for s in samples],
            "status": "insufficient_pairs",
            "reason": f"Need >= {min_pairs} paired seeds",
            "official_mean": _sample_mean(official_values) if official_values else 0.0,
            "worldflux_mean": _sample_mean(worldflux_values) if worldflux_values else 0.0,
        }

    lower_equiv = math.log(1.0 - equivalence_margin)
    upper_equiv = math.log(1.0 + equivalence_margin)
    lower_noninf = math.log(1.0 - noninferiority_margin)

    ratios = _paired_log_ratios(samples, eps)
    mean_lr = _sample_mean(ratios)
    std_lr = _sample_std(ratios)
    se_lr = std_lr / math.sqrt(max(1, n_pairs))

    p_lower = _one_sided_p_lower(mean_lr, se_lr, lower_equiv)
    p_upper = _one_sided_p_upper(mean_lr, se_lr, upper_equiv)
    tost_p = max(p_lower, p_upper)
    tost_pass = bool(p_lower < alpha and p_upper < alpha)

    p_noninf = _one_sided_p_lower(mean_lr, se_lr, lower_noninf)
    noninf_pass = bool(p_noninf < alpha)

    z = _z_value(alpha)
    ci_low = mean_lr - z * se_lr
    ci_high = mean_lr + z * se_lr

    return {
        "task_id": task_id,
        "metric": metric,
        "n_pairs": n_pairs,
        "seed_ids": [s.seed for s in samples],
        "status": "ok",
        "official_mean": _sample_mean(official_values),
        "worldflux_mean": _sample_mean(worldflux_values),
        "mean_log_ratio": mean_lr,
        "std_log_ratio": std_lr,
        "se_log_ratio": se_lr,
        "ratio_mean": math.exp(mean_lr),
        "ci90_log": [ci_low, ci_high],
        "ci90_ratio": [math.exp(ci_low), math.exp(ci_high)],
        "equivalence_bounds_log": [lower_equiv, upper_equiv],
        "equivalence_bounds_ratio": [1.0 - equivalence_margin, 1.0 + equivalence_margin],
        "noninferiority_bound_log": lower_noninf,
        "noninferiority_bound_ratio": 1.0 - noninferiority_margin,
        "tost": {
            "alpha": alpha,
            "p_lower": p_lower,
            "p_upper": p_upper,
            "p_value": tost_p,
            "pass_raw": tost_pass,
        },
        "noninferiority": {
            "alpha": alpha,
            "p_value": p_noninf,
            "pass_raw": noninf_pass,
        },
    }


def main() -> int:
    args = _parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise SystemExit("--metrics must include at least one metric")

    entries = _load_jsonl(args.input)

    task_set: set[str] = set()
    metric_reports: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for metric in metrics:
        paired = _collect_pairs(entries, metric=metric)
        task_set.update(paired.keys())
        for task_id, samples in paired.items():
            metric_reports[task_id][metric] = _metric_report(
                task_id=task_id,
                metric=metric,
                samples=samples,
                alpha=args.alpha,
                eps=args.eps,
                equivalence_margin=args.equivalence_margin,
                noninferiority_margin=args.noninferiority_margin,
                min_pairs=args.min_pairs,
            )

    ordered_tasks = sorted(task_set)

    primary_p_values: dict[str, float] = {}
    all_p_values: dict[str, float] = {}

    for task_id in ordered_tasks:
        for metric in metrics:
            report = metric_reports.get(task_id, {}).get(metric)
            if not report or report.get("status") != "ok":
                continue
            hypothesis_id = f"{task_id}::{metric}"
            p_value = float(report["tost"]["p_value"])
            all_p_values[hypothesis_id] = p_value
            if metric == args.primary_metric:
                primary_p_values[hypothesis_id] = p_value

    holm_primary = _holm_adjustments(primary_p_values, args.alpha)
    holm_all = _holm_adjustments(all_p_values, args.alpha)

    task_reports: list[dict[str, Any]] = []
    for task_id in ordered_tasks:
        per_metric: dict[str, Any] = {}
        primary_pass = True
        all_metrics_pass = True

        for metric in metrics:
            report = metric_reports.get(task_id, {}).get(metric)
            if not report:
                report = {
                    "task_id": task_id,
                    "metric": metric,
                    "status": "missing",
                    "reason": "No paired runs for metric",
                }
                primary_pass = False
                all_metrics_pass = False
            elif report.get("status") == "ok":
                hid = f"{task_id}::{metric}"
                report["holm_primary"] = holm_primary.get(hid)
                report["holm_all_metrics"] = holm_all.get(hid)

                if metric == args.primary_metric:
                    pass_primary = bool(holm_primary.get(hid, {}).get("pass", False))
                    report["pass_with_holm_primary"] = pass_primary
                    primary_pass = primary_pass and pass_primary

                pass_all = bool(holm_all.get(hid, {}).get("pass", False))
                report["pass_with_holm_all_metrics"] = pass_all
                all_metrics_pass = all_metrics_pass and pass_all
            else:
                primary_pass = False
                all_metrics_pass = False

            per_metric[metric] = report

        if args.primary_metric not in per_metric:
            primary_pass = False

        task_reports.append(
            {
                "task_id": task_id,
                "metrics": per_metric,
                "task_pass_primary": bool(primary_pass),
                "task_pass_all_metrics": bool(all_metrics_pass),
            }
        )

    tasks_pass_primary = sum(1 for t in task_reports if t["task_pass_primary"])
    tasks_pass_all = sum(1 for t in task_reports if t["task_pass_all_metrics"])

    output: dict[str, Any] = {
        "schema_version": "parity.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "input": str(args.input),
        "config": {
            "metrics": metrics,
            "primary_metric": args.primary_metric,
            "alpha": args.alpha,
            "equivalence_margin": args.equivalence_margin,
            "noninferiority_margin": args.noninferiority_margin,
            "eps": args.eps,
            "min_pairs": args.min_pairs,
        },
        "holm": {
            "primary": holm_primary,
            "all_metrics": holm_all,
        },
        "tasks": task_reports,
        "global": {
            "tasks_total": len(task_reports),
            "tasks_pass_primary": tasks_pass_primary,
            "tasks_pass_all_metrics": tasks_pass_all,
            "parity_pass_primary": bool(task_reports) and tasks_pass_primary == len(task_reports),
            "parity_pass_all_metrics": bool(task_reports) and tasks_pass_all == len(task_reports),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
