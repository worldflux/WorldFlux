#!/usr/bin/env python3
# ruff: noqa: E402
"""Compute statistical equivalence between official and WorldFlux parity runs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from contract_schema import SuiteContract, load_suite_contract
from metric_transforms import equivalence_bounds, transform_pair
from validity_gate import evaluate_validity


@dataclass(frozen=True)
class PairSample:
    seed: int
    official: float
    worldflux: float


@dataclass(frozen=True)
class TaskMetricConfig:
    primary_metric: str
    effect_transform: str
    higher_is_better: bool
    equivalence_margin: float
    noninferiority_margin: float
    alpha: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional parity manifest/suite contract used for per-task metric config.",
    )
    parser.add_argument("--metrics", type=str, default="final_return_mean,auc_return")
    parser.add_argument("--primary-metric", type=str, default="final_return_mean")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--equivalence-margin", type=float, default=0.05)
    parser.add_argument("--noninferiority-margin", type=float, default=0.05)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--min-pairs", type=int, default=2)
    parser.add_argument(
        "--systems",
        type=str,
        default="official,worldflux",
        help="Comma-separated system IDs required for each task/seed pair.",
    )
    parser.add_argument(
        "--strict-completeness",
        action="store_true",
        help="Fail fast when any expected task/seed/system result is missing or unsuccessful.",
    )
    parser.add_argument(
        "--strict-validity",
        action="store_true",
        help="Fail fast when validity gate detects proof-incompatible runs.",
    )
    parser.add_argument(
        "--proof-mode",
        action="store_true",
        help="Enable proof-mode validity checks (random policy/mock/protocol mismatch checks).",
    )
    parser.add_argument(
        "--policy-mode-required",
        type=str,
        default="parity_candidate",
        help="Required worldflux policy mode in proof-mode validity checks.",
    )
    parser.add_argument(
        "--validity-report",
        type=Path,
        default=None,
        help="Optional path for validity report JSON output.",
    )
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


def _load_manifest_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Manifest must be valid JSON (YAML optional parser unavailable: install pyyaml)."
            ) from exc
        loaded = yaml.safe_load(text)

    if not isinstance(loaded, dict):
        raise RuntimeError("Manifest root must be object.")
    return loaded


def _derive_task_configs(
    *,
    entries: list[dict[str, Any]],
    suite: SuiteContract | None,
    default_primary_metric: str,
    default_effect_transform: str,
    default_higher_is_better: bool,
    default_equivalence_margin: float,
    default_noninferiority_margin: float,
    default_alpha: float,
) -> dict[str, TaskMetricConfig]:
    out: dict[str, TaskMetricConfig] = {}

    if suite is not None:
        for task in suite.tasks:
            out[task.task_id] = TaskMetricConfig(
                primary_metric=task.primary_metric,
                effect_transform=task.effect_transform,
                higher_is_better=task.higher_is_better,
                equivalence_margin=task.equivalence_margin,
                noninferiority_margin=task.noninferiority_margin,
                alpha=task.alpha,
            )

    for entry in entries:
        task_id = str(entry.get("task_id", "")).strip()
        if not task_id or task_id in out:
            continue
        out[task_id] = TaskMetricConfig(
            primary_metric=str(entry.get("primary_metric", default_primary_metric)),
            effect_transform=str(entry.get("effect_transform", default_effect_transform)),
            higher_is_better=bool(entry.get("higher_is_better", default_higher_is_better)),
            equivalence_margin=float(entry.get("equivalence_margin", default_equivalence_margin)),
            noninferiority_margin=float(
                entry.get("noninferiority_margin", default_noninferiority_margin)
            ),
            alpha=float(entry.get("alpha", default_alpha)),
        )

    return out


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


def _record_priority(entry: dict[str, Any]) -> tuple[int, int, str]:
    status = str(entry.get("status", ""))
    status_rank = 3 if status == "success" else (2 if status == "failed" else 1)
    attempt = int(entry.get("attempt", 0) or 0)
    timestamp = str(entry.get("timestamp", ""))
    return (status_rank, attempt, timestamp)


def _completeness_summary(
    entries: list[dict[str, Any]],
    *,
    metrics: list[str],
    systems: list[str],
) -> dict[str, Any]:
    latest: dict[tuple[str, int, str], dict[str, Any]] = {}
    task_seed_keys: set[tuple[str, int]] = set()

    for entry in entries:
        task_id = str(entry.get("task_id", ""))
        system = str(entry.get("system", ""))
        seed = int(entry.get("seed", -1))
        if not task_id or seed < 0 or system not in systems:
            continue
        key = (task_id, seed, system)
        task_seed_keys.add((task_id, seed))
        current = latest.get(key)
        if current is None or _record_priority(entry) > _record_priority(current):
            latest[key] = entry

    expected_keys = {
        (task_id, seed, system) for task_id, seed in task_seed_keys for system in systems
    }

    missing_entries: list[dict[str, Any]] = []
    for task_id, seed, system in sorted(expected_keys):
        payload = latest.get((task_id, seed, system))
        if payload is None:
            missing_entries.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "system": system,
                    "reason": "missing_record",
                }
            )
            continue
        if payload.get("status") != "success":
            missing_entries.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "system": system,
                    "reason": f"status={payload.get('status')}",
                }
            )
            continue
        payload_metrics = payload.get("metrics", {})
        if not isinstance(payload_metrics, dict):
            missing_entries.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "system": system,
                    "reason": "metrics_missing",
                }
            )
            continue
        missing_metric_names = [
            metric for metric in metrics if not isinstance(payload_metrics.get(metric), int | float)
        ]
        if missing_metric_names:
            missing_entries.append(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "system": system,
                    "reason": f"missing_metrics={','.join(missing_metric_names)}",
                }
            )

    missing_keyset = {
        (str(item["task_id"]), int(item["seed"]), str(item["system"])) for item in missing_entries
    }
    complete_task_seeds = {
        (task_id, seed)
        for (task_id, seed) in task_seed_keys
        if all((task_id, seed, system) not in missing_keyset for system in systems)
    }

    return {
        "systems": systems,
        "task_seed_count": len(task_seed_keys),
        "expected_pairs": len(expected_keys),
        "complete_task_seed_pairs": len(complete_task_seeds),
        "missing_pairs": len(missing_entries),
        "missing_entries": missing_entries,
    }


def _sample_mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = _sample_mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(max(0.0, var))


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
    effect_transform: str,
    higher_is_better: bool,
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

    bounds = equivalence_bounds(
        transform=effect_transform,
        equivalence_margin=equivalence_margin,
        noninferiority_margin=noninferiority_margin,
    )

    effects: list[float] = []
    for sample in samples:
        effects.append(
            transform_pair(
                transform=effect_transform,
                official=sample.official,
                worldflux=sample.worldflux,
                higher_is_better=higher_is_better,
                eps=eps,
            )
        )

    mean_effect = _sample_mean(effects)
    std_effect = _sample_std(effects)
    se_effect = std_effect / math.sqrt(max(1, n_pairs))

    p_lower = _one_sided_p_lower(mean_effect, se_effect, bounds.lower_equivalence)
    p_upper = _one_sided_p_upper(mean_effect, se_effect, bounds.upper_equivalence)
    tost_p = max(p_lower, p_upper)
    tost_pass = bool(p_lower < alpha and p_upper < alpha)

    p_noninf = _one_sided_p_lower(mean_effect, se_effect, bounds.lower_noninferiority)
    noninf_pass = bool(p_noninf < alpha)

    z = _z_value(alpha)
    ci_low = mean_effect - z * se_effect
    ci_high = mean_effect + z * se_effect

    report: dict[str, Any] = {
        "task_id": task_id,
        "metric": metric,
        "effect_transform": effect_transform,
        "higher_is_better": bool(higher_is_better),
        "n_pairs": n_pairs,
        "seed_ids": [s.seed for s in samples],
        "status": "ok",
        "official_mean": _sample_mean(official_values),
        "worldflux_mean": _sample_mean(worldflux_values),
        "mean_effect": mean_effect,
        "std_effect": std_effect,
        "se_effect": se_effect,
        "ci90_effect": [ci_low, ci_high],
        "equivalence_bounds_effect": [bounds.lower_equivalence, bounds.upper_equivalence],
        "noninferiority_bound_effect": bounds.lower_noninferiority,
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

    if effect_transform == "paired_log_ratio":
        report["mean_log_ratio"] = mean_effect
        report["std_log_ratio"] = std_effect
        report["se_log_ratio"] = se_effect
        report["ratio_mean"] = math.exp(mean_effect)
        report["ci90_log"] = [ci_low, ci_high]
        report["ci90_ratio"] = [math.exp(ci_low), math.exp(ci_high)]
        report["equivalence_bounds_log"] = [bounds.lower_equivalence, bounds.upper_equivalence]
        report["equivalence_bounds_ratio"] = [1.0 - equivalence_margin, 1.0 + equivalence_margin]
        report["noninferiority_bound_log"] = bounds.lower_noninferiority
        report["noninferiority_bound_ratio"] = 1.0 - noninferiority_margin

    return report


def main() -> int:
    args = _parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        raise SystemExit("--metrics must include at least one metric")
    systems = [item.strip() for item in args.systems.split(",") if item.strip()]
    if not systems:
        raise SystemExit("--systems must include at least one system")

    suite_contract: SuiteContract | None = None
    suite_requirements: dict[str, Any] = {}
    if args.manifest is not None:
        suite_contract = load_suite_contract(_load_manifest_payload(args.manifest))
        suite_requirements = dict(suite_contract.validity_requirements)
        for metric in (suite_contract.primary_metric, *suite_contract.secondary_metrics):
            if metric not in metrics:
                metrics.append(metric)

    entries = _load_jsonl(args.input)
    task_configs = _derive_task_configs(
        entries=entries,
        suite=suite_contract,
        default_primary_metric=args.primary_metric,
        default_effect_transform="paired_log_ratio",
        default_higher_is_better=True,
        default_equivalence_margin=args.equivalence_margin,
        default_noninferiority_margin=args.noninferiority_margin,
        default_alpha=args.alpha,
    )

    completeness = _completeness_summary(entries, metrics=metrics, systems=systems)
    validity = evaluate_validity(
        entries,
        proof_mode=bool(args.proof_mode or args.strict_validity),
        required_policy_mode=args.policy_mode_required,
        requirements=suite_requirements,
    )
    if args.validity_report is not None:
        args.validity_report.parent.mkdir(parents=True, exist_ok=True)
        args.validity_report.write_text(
            json.dumps(validity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    strict_failure = bool(
        (args.strict_completeness and completeness["missing_pairs"] > 0)
        or (args.strict_validity and not bool(validity.get("pass", False)))
    )
    if strict_failure:
        output: dict[str, Any] = {
            "schema_version": "parity.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input": str(args.input),
            "config": {
                "metrics": metrics,
                "primary_metric": args.primary_metric,
                "alpha": args.alpha,
                "equivalence_margin": args.equivalence_margin,
                "noninferiority_margin": args.noninferiority_margin,
                "eps": args.eps,
                "min_pairs": args.min_pairs,
                "systems": systems,
                "strict_completeness": bool(args.strict_completeness),
                "strict_validity": bool(args.strict_validity),
                "proof_mode": bool(args.proof_mode or args.strict_validity),
                "policy_mode_required": args.policy_mode_required,
                "manifest": str(args.manifest) if args.manifest else None,
            },
            "completeness": completeness,
            "validity": validity,
            "holm": {
                "primary": {},
                "all_metrics": {},
            },
            "tasks": [],
            "global": {
                "tasks_total": 0,
                "tasks_pass_primary": 0,
                "tasks_pass_all_metrics": 0,
                "parity_pass_primary": False,
                "parity_pass_all_metrics": False,
                "parity_pass_final": False,
                "strict_mode_failed": True,
                "missing_pairs": int(completeness["missing_pairs"]),
                "validity_pass": bool(validity.get("pass", False)),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(output, indent=2, sort_keys=True))
        return 1

    task_set: set[str] = set()
    metric_reports: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)

    for metric in metrics:
        paired = _collect_pairs(entries, metric=metric)
        task_set.update(paired.keys())
        for task_id, samples in paired.items():
            task_cfg = task_configs.get(
                task_id,
                TaskMetricConfig(
                    primary_metric=args.primary_metric,
                    effect_transform="paired_log_ratio",
                    higher_is_better=True,
                    equivalence_margin=args.equivalence_margin,
                    noninferiority_margin=args.noninferiority_margin,
                    alpha=args.alpha,
                ),
            )
            metric_reports[task_id][metric] = _metric_report(
                task_id=task_id,
                metric=metric,
                samples=samples,
                alpha=task_cfg.alpha,
                eps=args.eps,
                equivalence_margin=task_cfg.equivalence_margin,
                noninferiority_margin=task_cfg.noninferiority_margin,
                effect_transform=task_cfg.effect_transform,
                higher_is_better=task_cfg.higher_is_better,
                min_pairs=args.min_pairs,
            )

    discovered_tasks = {
        str(entry.get("task_id", "")) for entry in entries if str(entry.get("task_id", "")).strip()
    }
    ordered_tasks = sorted(task_set | discovered_tasks)

    primary_p_values: dict[str, float] = {}
    all_p_values: dict[str, float] = {}

    for task_id in ordered_tasks:
        task_cfg = task_configs.get(
            task_id,
            TaskMetricConfig(
                primary_metric=args.primary_metric,
                effect_transform="paired_log_ratio",
                higher_is_better=True,
                equivalence_margin=args.equivalence_margin,
                noninferiority_margin=args.noninferiority_margin,
                alpha=args.alpha,
            ),
        )
        for metric in metrics:
            report = metric_reports.get(task_id, {}).get(metric)
            if not report or report.get("status") != "ok":
                continue
            hypothesis_id = f"{task_id}::{metric}"
            p_value = float(report["tost"]["p_value"])
            all_p_values[hypothesis_id] = p_value
            if metric == task_cfg.primary_metric:
                primary_p_values[hypothesis_id] = p_value

    holm_primary = _holm_adjustments(primary_p_values, args.alpha)
    holm_all = _holm_adjustments(all_p_values, args.alpha)

    task_reports: list[dict[str, Any]] = []
    for task_id in ordered_tasks:
        task_cfg = task_configs.get(
            task_id,
            TaskMetricConfig(
                primary_metric=args.primary_metric,
                effect_transform="paired_log_ratio",
                higher_is_better=True,
                equivalence_margin=args.equivalence_margin,
                noninferiority_margin=args.noninferiority_margin,
                alpha=args.alpha,
            ),
        )
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

                if metric == task_cfg.primary_metric:
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

        if task_cfg.primary_metric not in per_metric:
            primary_pass = False

        task_reports.append(
            {
                "task_id": task_id,
                "task_primary_metric": task_cfg.primary_metric,
                "metrics": per_metric,
                "task_pass_primary": bool(primary_pass),
                "task_pass_all_metrics": bool(all_metrics_pass),
            }
        )

    tasks_pass_primary = sum(1 for t in task_reports if t["task_pass_primary"])
    tasks_pass_all = sum(1 for t in task_reports if t["task_pass_all_metrics"])

    output: dict[str, Any] = {
        "schema_version": "parity.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "config": {
            "metrics": metrics,
            "primary_metric": args.primary_metric,
            "alpha": args.alpha,
            "equivalence_margin": args.equivalence_margin,
            "noninferiority_margin": args.noninferiority_margin,
            "eps": args.eps,
            "min_pairs": args.min_pairs,
            "systems": systems,
            "strict_completeness": bool(args.strict_completeness),
            "strict_validity": bool(args.strict_validity),
            "proof_mode": bool(args.proof_mode or args.strict_validity),
            "policy_mode_required": args.policy_mode_required,
            "manifest": str(args.manifest) if args.manifest else None,
        },
        "completeness": completeness,
        "validity": validity,
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
            "parity_pass_final": bool(task_reports)
            and tasks_pass_all == len(task_reports)
            and int(completeness["missing_pairs"]) == 0
            and bool(validity.get("pass", False)),
            "strict_mode_failed": False,
            "missing_pairs": int(completeness["missing_pairs"]),
            "validity_pass": bool(validity.get("pass", False)),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
