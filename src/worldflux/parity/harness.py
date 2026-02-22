"""Parity harness orchestration for run/aggregate/report flows."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ParityError
from .loaders import SourceSpec, load_score_points, load_suite_spec
from .stats import non_inferiority_test
from .types import ScorePoint

RUN_SCHEMA_VERSION = "worldflux.parity.run.v1"
AGGREGATE_SCHEMA_VERSION = "worldflux.parity.aggregate.v1"
DEFAULT_UPSTREAM_LOCK_PATH = Path("reports/parity/upstream_lock.json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _round_float(value: float) -> float:
    return float(f"{value:.10f}")


def _dedupe_latest(points: list[ScorePoint]) -> dict[tuple[str, int], ScorePoint]:
    latest: dict[tuple[str, int], ScorePoint] = {}
    for point in points:
        current = latest.get(point.key)
        if current is None or point.step >= current.step:
            latest[point.key] = point
    return latest


def _resolve_output_path(suite_id: str, output_path: Path | None) -> Path:
    if output_path is not None:
        return output_path
    return Path("reports/parity/runs") / f"{suite_id}.json"


def _override_source(
    source: SourceSpec,
    *,
    path_override: Path | None = None,
    format_override: str | None = None,
) -> SourceSpec:
    path = path_override.resolve() if path_override is not None else source.path
    format_name = format_override if format_override is not None else source.format
    return replace(source, path=path, format=format_name)


def _relative_drop_ratio(
    upstream_score: float,
    worldflux_score: float,
    *,
    higher_is_better: bool,
) -> float:
    """Return positive values when WorldFlux underperforms upstream."""
    if higher_is_better:
        gap = upstream_score - worldflux_score
    else:
        gap = worldflux_score - upstream_score
    denom = max(abs(upstream_score), 1.0)
    return float(gap / denom)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_path(path: Path) -> str:
    if path.is_file():
        return _hash_file(path)
    if not path.is_dir():
        raise ParityError(f"Cannot hash non-file, non-directory path: {path}")

    digest = hashlib.sha256()
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        rel = file_path.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\n")
        digest.update(_hash_file(file_path).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_upstream_lock(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ParityError(f"Upstream lock payload must be object: {path}")
    return payload


def _suite_lock_ref(
    lock_payload: dict[str, Any] | None,
    *,
    lock_path: Path,
    suite_id: str,
    upstream_commit: str | None,
) -> dict[str, Any]:
    lock_version = None
    locked_commit = None
    locked_repo = None
    lock_found = False

    if isinstance(lock_payload, dict):
        lock_version = lock_payload.get("schema_version")
        suites = lock_payload.get("suites")
        if isinstance(suites, dict):
            suite_entry = suites.get(suite_id)
            if isinstance(suite_entry, dict):
                lock_found = True
                commit_value = suite_entry.get("commit")
                repo_value = suite_entry.get("repo")
                if commit_value is not None:
                    locked_commit = str(commit_value)
                if repo_value is not None:
                    locked_repo = str(repo_value)

    matches = (
        locked_commit == upstream_commit
        if lock_found and locked_commit is not None and upstream_commit is not None
        else None
    )

    return {
        "suite_id": suite_id,
        "lock_path": str(lock_path),
        "lock_version": lock_version,
        "lock_found": lock_found,
        "locked_upstream_commit": locked_commit,
        "locked_upstream_repo": locked_repo,
        "resolved_upstream_commit": upstream_commit,
        "matches_lock": matches,
    }


def _evaluation_manifest(generated_at_utc: str) -> dict[str, Any]:
    torch_version: str | None = None
    cuda_version: str | None = None
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        torch_version = None
        cuda_version = None
    else:  # pragma: no branch - tiny branch for optional metadata
        torch_version = str(torch.__version__)
        cuda_raw = getattr(torch.version, "cuda", None)
        cuda_version = str(cuda_raw) if cuda_raw is not None else None

    return {
        "runner": "worldflux.parity.run_suite",
        "python": platform.python_version(),
        "python_impl": platform.python_implementation(),
        "platform": sys.platform,
        "torch": torch_version,
        "cuda": cuda_version,
        "seed_policy": "deterministic_sort+bootstrap_seed_0",
        "generated_at_utc": generated_at_utc,
    }


def _build_verdict_reason(*, ci_upper_ratio: float, margin_ratio: float) -> tuple[bool, str]:
    passed = bool(ci_upper_ratio <= margin_ratio)
    if passed:
        return (
            True,
            f"PASS: one-sided upper CI {ci_upper_ratio:.6f} <= margin {margin_ratio:.6f}",
        )
    return (
        False,
        f"FAIL: one-sided upper CI {ci_upper_ratio:.6f} > margin {margin_ratio:.6f}",
    )


def run_suite(
    suite_path: Path,
    *,
    output_path: Path | None = None,
    upstream_path: Path | None = None,
    upstream_format: str | None = None,
    worldflux_path: Path | None = None,
    worldflux_format: str | None = None,
    upstream_lock_path: Path | None = None,
) -> dict[str, Any]:
    """Run one parity suite and write normalized comparison artifact."""
    suite_path_resolved = suite_path.resolve()
    suite = load_suite_spec(suite_path_resolved)
    upstream_source = _override_source(
        suite.upstream,
        path_override=upstream_path,
        format_override=upstream_format,
    )
    worldflux_source = _override_source(
        suite.worldflux,
        path_override=worldflux_path,
        format_override=worldflux_format,
    )

    upstream_points = load_score_points(upstream_source.path, upstream_source.format)
    worldflux_points = load_score_points(worldflux_source.path, worldflux_source.format)

    if suite.tasks:
        allowed_tasks = set(suite.tasks)
        upstream_points = [point for point in upstream_points if point.task in allowed_tasks]
        worldflux_points = [point for point in worldflux_points if point.task in allowed_tasks]

    upstream_latest = _dedupe_latest(upstream_points)
    worldflux_latest = _dedupe_latest(worldflux_points)

    upstream_keys = set(upstream_latest)
    worldflux_keys = set(worldflux_latest)
    paired_keys = sorted(upstream_keys & worldflux_keys)
    if not paired_keys:
        raise ParityError(
            "No paired task/seed data found between upstream and worldflux sources for suite "
            f"{suite.suite_id!r}."
        )

    pairs: list[dict[str, Any]] = []
    drop_ratios: list[float] = []
    upstream_sum = 0.0
    worldflux_sum = 0.0

    for task, seed in paired_keys:
        upstream_point = upstream_latest[(task, seed)]
        worldflux_point = worldflux_latest[(task, seed)]
        drop_ratio = _relative_drop_ratio(
            upstream_point.score,
            worldflux_point.score,
            higher_is_better=suite.higher_is_better,
        )
        rounded_drop_ratio = _round_float(drop_ratio)
        drop_ratios.append(rounded_drop_ratio)
        upstream_sum += upstream_point.score
        worldflux_sum += worldflux_point.score

        pairs.append(
            {
                "task": task,
                "seed": int(seed),
                "upstream_score": _round_float(upstream_point.score),
                "worldflux_score": _round_float(worldflux_point.score),
                "upstream_step": int(upstream_point.step),
                "worldflux_step": int(worldflux_point.step),
                "drop_ratio": rounded_drop_ratio,
            }
        )

    stats = non_inferiority_test(
        drop_ratios,
        margin_ratio=suite.margin_ratio,
        confidence=suite.confidence,
    )
    passed, verdict_reason = _build_verdict_reason(
        ci_upper_ratio=stats.ci_upper_ratio,
        margin_ratio=stats.margin_ratio,
    )

    upstream_only = sorted(upstream_keys - worldflux_keys)
    worldflux_only = sorted(worldflux_keys - upstream_keys)

    generated_at_utc = _utc_now_iso()
    lock_path = (upstream_lock_path or DEFAULT_UPSTREAM_LOCK_PATH).resolve()
    lock_payload = _load_upstream_lock(lock_path)

    result = {
        "schema_version": RUN_SCHEMA_VERSION,
        "generated_at_utc": generated_at_utc,
        "suite": {
            "suite_id": suite.suite_id,
            "family": suite.family,
            "metric": suite.metric,
            "higher_is_better": suite.higher_is_better,
            "margin_ratio": _round_float(suite.margin_ratio),
            "confidence": _round_float(suite.confidence),
            "tasks": list(suite.tasks),
        },
        "sources": {
            "upstream": {
                "path": str(upstream_source.path),
                "format": upstream_source.format,
                "repo": upstream_source.repo,
                "commit": upstream_source.commit,
            },
            "worldflux": {
                "path": str(worldflux_source.path),
                "format": worldflux_source.format,
                "repo": worldflux_source.repo,
                "commit": worldflux_source.commit,
            },
        },
        "evaluation_manifest": _evaluation_manifest(generated_at_utc),
        "artifact_integrity": {
            "suite_sha256": _hash_path(suite_path_resolved),
            "upstream_input_sha256": _hash_path(upstream_source.path),
            "worldflux_input_sha256": _hash_path(worldflux_source.path),
            "upstream_lock_sha256": _hash_path(lock_path) if lock_payload is not None else None,
        },
        "suite_lock_ref": _suite_lock_ref(
            lock_payload,
            lock_path=lock_path,
            suite_id=suite.suite_id,
            upstream_commit=upstream_source.commit,
        ),
        "counts": {
            "upstream_points": len(upstream_latest),
            "worldflux_points": len(worldflux_latest),
            "paired": len(pairs),
            "upstream_only": len(upstream_only),
            "worldflux_only": len(worldflux_only),
        },
        "stats": {
            "sample_size": stats.sample_size,
            "mean_upstream_score": _round_float(upstream_sum / len(pairs)),
            "mean_worldflux_score": _round_float(worldflux_sum / len(pairs)),
            "mean_drop_ratio": stats.mean_drop_ratio,
            "ci_lower_ratio": stats.ci_lower_ratio,
            "ci_upper_ratio": stats.ci_upper_ratio,
            "confidence": stats.confidence,
            "margin_ratio": stats.margin_ratio,
            "pass_non_inferiority": passed,
            "verdict_reason": verdict_reason,
        },
        "missing": {
            "upstream_only": [{"task": task, "seed": seed} for task, seed in upstream_only],
            "worldflux_only": [{"task": task, "seed": seed} for task, seed in worldflux_only],
        },
        "pairs": pairs,
        "pass": passed,
    }

    try:
        from .paper_comparison import compare_against_paper

        task_scores: dict[str, list[float]] = {}
        for pair in pairs:
            task_scores.setdefault(pair["task"], []).append(float(pair["worldflux_score"]))
        mean_scores = {task: sum(s) / len(s) for task, s in task_scores.items()}
        comparison = compare_against_paper(suite.suite_id, mean_scores)
        if comparison is not None:
            result["paper_comparison"] = {
                "suite_id": comparison.suite_id,
                "mean_relative_delta_pct": comparison.mean_relative_delta_pct,
                "tasks_within_5pct": comparison.tasks_within_5pct,
                "deltas": [
                    {
                        "task": d.task,
                        "paper_score": d.paper_score,
                        "run_score": d.run_score,
                        "absolute_delta": d.absolute_delta,
                        "relative_delta_pct": d.relative_delta_pct,
                    }
                    for d in comparison.deltas
                ],
            }
    except (ImportError, TypeError, ValueError, KeyError):  # pragma: no cover - optional enrichment
        pass

    out_path = _resolve_output_path(suite.suite_id, output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def aggregate_runs(
    run_paths: list[Path],
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Aggregate multiple parity run artifacts into one top-level verdict."""
    if not run_paths:
        raise ParityError("aggregate_runs requires at least one run artifact")

    runs: list[dict[str, Any]] = []
    for path in run_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ParityError(f"Run payload must be object: {path}")
        runs.append(payload)

    suite_rows: list[dict[str, Any]] = []
    family_rollup: dict[str, dict[str, int]] = {}
    all_drop_ratios: list[float] = []
    margins: list[float] = []
    confidences: list[float] = []

    for run, path in zip(runs, run_paths, strict=True):
        suite = run.get("suite", {})
        stats = run.get("stats", {})
        pairs = run.get("pairs", [])

        suite_id = str(suite.get("suite_id", "unknown"))
        family = str(suite.get("family", "unknown"))

        margin_ratio = _round_float(
            float(stats.get("margin_ratio", suite.get("margin_ratio", 0.05)))
        )
        confidence = _round_float(float(stats.get("confidence", suite.get("confidence", 0.95))))
        margins.append(margin_ratio)
        confidences.append(confidence)

        for pair in pairs:
            if not isinstance(pair, dict):
                continue
            if "drop_ratio" not in pair:
                continue
            all_drop_ratios.append(float(pair["drop_ratio"]))

        ci_upper_ratio = _round_float(float(stats.get("ci_upper_ratio", 0.0)))
        passed, verdict_reason = _build_verdict_reason(
            ci_upper_ratio=ci_upper_ratio,
            margin_ratio=margin_ratio,
        )

        suite_rows.append(
            {
                "suite_id": suite_id,
                "family": family,
                "sample_size": int(stats.get("sample_size", len(pairs))),
                "margin_ratio": margin_ratio,
                "confidence": confidence,
                "mean_drop_ratio": _round_float(float(stats.get("mean_drop_ratio", 0.0))),
                "ci_upper_ratio": ci_upper_ratio,
                "pass_non_inferiority": passed,
                "verdict_reason": str(stats.get("verdict_reason", verdict_reason)),
                "run_path": str(path),
            }
        )

        rollup = family_rollup.setdefault(family, {"total": 0, "pass": 0})
        rollup["total"] += 1
        if passed:
            rollup["pass"] += 1

    suite_pass_count = sum(1 for row in suite_rows if row["pass_non_inferiority"])
    suite_fail_count = len(suite_rows) - suite_pass_count
    all_suites_pass = suite_fail_count == 0

    global_stats: dict[str, Any] | None = None
    if all_drop_ratios:
        global_margin = min(margins) if margins else 0.05
        global_confidence = min(confidences) if confidences else 0.95
        global_result = non_inferiority_test(
            all_drop_ratios,
            margin_ratio=global_margin,
            confidence=global_confidence,
        )
        global_passed, global_reason = _build_verdict_reason(
            ci_upper_ratio=global_result.ci_upper_ratio,
            margin_ratio=global_result.margin_ratio,
        )
        global_stats = {
            "sample_size": global_result.sample_size,
            "mean_drop_ratio": global_result.mean_drop_ratio,
            "ci_lower_ratio": global_result.ci_lower_ratio,
            "ci_upper_ratio": global_result.ci_upper_ratio,
            "confidence": global_result.confidence,
            "margin_ratio": global_result.margin_ratio,
            "pass_non_inferiority": global_passed,
            "verdict_reason": global_reason,
        }

    families = [
        {
            "family": family,
            "runs": counts["total"],
            "pass": counts["pass"],
            "fail": counts["total"] - counts["pass"],
        }
        for family, counts in sorted(family_rollup.items())
    ]

    aggregate = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "run_count": len(runs),
        "suite_pass_count": suite_pass_count,
        "suite_fail_count": suite_fail_count,
        "all_suites_pass": all_suites_pass,
        "global_non_inferiority": global_stats,
        "families": families,
        "suites": sorted(suite_rows, key=lambda row: str(row["suite_id"])),
        "run_paths": [str(path) for path in run_paths],
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return aggregate


def render_markdown_report(aggregate: dict[str, Any]) -> str:
    """Render aggregate parity payload as a concise markdown report."""
    suites = aggregate.get("suites", [])
    if not isinstance(suites, list):
        raise ParityError("aggregate.suites must be a list")

    overall = "PASS" if bool(aggregate.get("all_suites_pass", False)) else "FAIL"
    lines = [
        "# WorldFlux Parity Report",
        "",
        f"Generated at: {aggregate.get('generated_at_utc', 'unknown')}",
        f"Overall suite status: **{overall}**",
        "",
        "## Suite Results",
        "",
        "| Suite | Family | Samples | Mean Drop | Upper CI (one-sided) | Margin | Verdict | Reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    for row in suites:
        if not isinstance(row, dict):
            continue
        verdict = "PASS" if bool(row.get("pass_non_inferiority", False)) else "FAIL"
        lines.append(
            "| {suite} | {family} | {samples} | {mean:.4f} | {upper:.4f} | {margin:.4f} | {verdict} | {reason} |".format(
                suite=row.get("suite_id", "unknown"),
                family=row.get("family", "unknown"),
                samples=int(row.get("sample_size", 0)),
                mean=float(row.get("mean_drop_ratio", 0.0)),
                upper=float(row.get("ci_upper_ratio", 0.0)),
                margin=float(row.get("margin_ratio", 0.0)),
                verdict=verdict,
                reason=str(row.get("verdict_reason", "")).replace("|", "\\|"),
            )
        )

    global_stats = aggregate.get("global_non_inferiority")
    if isinstance(global_stats, dict):
        global_verdict = "PASS" if bool(global_stats.get("pass_non_inferiority", False)) else "FAIL"
        lines.extend(
            [
                "",
                "## Global Rollup",
                "",
                f"Sample size: {int(global_stats.get('sample_size', 0))}",
                f"Mean drop ratio: {float(global_stats.get('mean_drop_ratio', 0.0)):.4f}",
                f"Upper CI (one-sided): {float(global_stats.get('ci_upper_ratio', 0.0)):.4f}",
                f"Margin ratio: {float(global_stats.get('margin_ratio', 0.0)):.4f}",
                f"Verdict: **{global_verdict}**",
                f"Reason: {global_stats.get('verdict_reason', '')}",
            ]
        )

    return "\n".join(lines) + "\n"
