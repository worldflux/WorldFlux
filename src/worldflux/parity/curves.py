"""Curve data loading and aggregation for parity JSONL results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CurvePoint:
    """A single (step, value) point on a learning curve."""

    step: float
    value: float


@dataclass(frozen=True)
class CurveData:
    """Learning curve for one (task, seed, system) combination."""

    task: str
    seed: int
    system: str
    points: list[CurvePoint] = field(default_factory=list)


def load_curves_from_parity_jsonl(path: Path) -> list[CurveData]:
    """Parse parity JSONL files and extract learning curves.

    Each JSON line must have ``schema_version == "parity.v1"`` and contain
    ``adapter``, ``task_id``, ``seed``, and ``curve`` fields.  The ``curve``
    field is a list of ``{"step": float, "return": float}`` dicts.

    Parameters
    ----------
    path:
        Path to a single JSONL file **or** a directory of ``*.jsonl`` files.

    Returns
    -------
    list[CurveData]
        One entry per (task, seed, system) found in the input.
    """
    paths: list[Path]
    if path.is_dir():
        paths = sorted(path.glob("*.jsonl"))
    else:
        paths = [path]

    results: list[CurveData] = []
    for file_path in paths:
        with file_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record: dict[str, Any] = json.loads(line)
                if record.get("schema_version") != "parity.v1":
                    continue
                adapter = record.get("adapter", "")
                task_id = record.get("task_id", "")
                seed = int(record.get("seed", 0))
                raw_curve = record.get("curve")
                if not isinstance(raw_curve, list):
                    continue

                points = [
                    CurvePoint(step=float(pt["step"]), value=float(pt["return"]))
                    for pt in raw_curve
                    if isinstance(pt, dict) and "step" in pt and "return" in pt
                ]
                results.append(CurveData(task=task_id, seed=seed, system=adapter, points=points))
    return results


def aggregate_curves(
    curves: list[CurveData],
    *,
    num_interp_points: int = 200,
    ci_percentile: float = 95.0,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Group curves by (task, system), interpolate, and compute mean + CI.

    Parameters
    ----------
    curves:
        Raw curve data as returned by :func:`load_curves_from_parity_jsonl`.
    num_interp_points:
        Number of evenly-spaced points for the interpolation grid.
    ci_percentile:
        Confidence interval width (e.g. 95 → 2.5th and 97.5th percentiles).

    Returns
    -------
    dict
        Mapping ``(task, system)`` to a dict with keys ``steps`` (1-D array),
        ``mean`` (1-D array), ``ci_low`` (1-D array), ``ci_high`` (1-D array).
    """
    groups: dict[tuple[str, str], list[CurveData]] = {}
    for cd in curves:
        key = (cd.task, cd.system)
        groups.setdefault(key, []).append(cd)

    lo_pct = (100.0 - ci_percentile) / 2.0
    hi_pct = 100.0 - lo_pct

    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, group in groups.items():
        # Determine common step range across all seeds in this group.
        all_min: list[float] = []
        all_max: list[float] = []
        for cd in group:
            if len(cd.points) < 2:
                continue
            steps = [p.step for p in cd.points]
            all_min.append(min(steps))
            all_max.append(max(steps))

        if not all_min:
            continue

        grid_lo = max(all_min)  # start where all curves have data
        grid_hi = min(all_max)  # end where all curves have data
        if grid_lo >= grid_hi:
            continue

        common_steps = np.linspace(grid_lo, grid_hi, num_interp_points)

        # Interpolate each seed curve onto the common grid.
        interp_matrix: list[np.ndarray] = []
        for cd in group:
            if len(cd.points) < 2:
                continue
            xs = np.array([p.step for p in cd.points])
            ys = np.array([p.value for p in cd.points])
            interp_vals = np.interp(common_steps, xs, ys)
            interp_matrix.append(interp_vals)

        if not interp_matrix:
            continue

        stacked = np.stack(interp_matrix, axis=0)  # (n_seeds, n_points)
        mean_curve = np.mean(stacked, axis=0)
        ci_low = np.percentile(stacked, lo_pct, axis=0)
        ci_high = np.percentile(stacked, hi_pct, axis=0)

        out[key] = {
            "steps": common_steps,
            "mean": mean_curve,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    return out
