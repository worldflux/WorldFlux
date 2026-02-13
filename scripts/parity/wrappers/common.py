#!/usr/bin/env python3
"""Common helpers for parity adapter wrapper scripts."""

from __future__ import annotations

import csv
import json
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CurvePoint:
    step: float
    value: float


def run_command(
    command: str | list[str],
    *,
    cwd: Path,
    timeout_sec: int | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command and return the completed process."""
    return subprocess.run(
        command,
        cwd=str(cwd),
        shell=isinstance(command, str),
        text=True,
        capture_output=True,
        timeout=timeout_sec,
        env=env,
        check=False,
    )


def find_latest_file(search_roots: list[Path], names: list[str]) -> Path | None:
    """Find newest file matching any candidate name under given roots."""
    candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for name in names:
            candidates.extend(root.rglob(name))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _first_numeric(mapping: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, int | float):
            return float(value)
    for key, value in mapping.items():
        key_l = str(key).lower()
        if not isinstance(value, int | float):
            continue
        if "score" in key_l or "reward" in key_l or "return" in key_l:
            return float(value)
    return None


def _step_value(mapping: dict[str, Any], keys: list[str], fallback: int) -> float:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, int | float):
            return float(value)
    return float(fallback)


def load_jsonl_curve(
    path: Path,
    *,
    value_keys: list[str],
    step_keys: list[str] | None = None,
) -> list[CurvePoint]:
    """Load step/value curve points from a JSONL file."""
    step_keys = step_keys or ["step", "env_step", "env_steps", "total_steps", "global_step"]
    points: list[CurvePoint] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                continue
            value = _first_numeric(parsed, value_keys)
            if value is None:
                continue
            step = _step_value(parsed, step_keys, idx)
            points.append(CurvePoint(step=step, value=value))
    return sorted(points, key=lambda p: p.step)


def load_csv_curve(
    path: Path,
    *,
    value_keys: list[str],
    step_keys: list[str] | None = None,
) -> list[CurvePoint]:
    """Load step/value curve points from a CSV file."""
    step_keys = step_keys or ["step", "env_step", "env_steps", "global_step"]
    points: list[CurvePoint] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            value = _first_numeric(row, value_keys)
            if value is None:
                continue
            step = _step_value(row, step_keys, idx)
            points.append(CurvePoint(step=step, value=value))
    return sorted(points, key=lambda p: p.step)


def curve_final_mean(points: list[CurvePoint], window: int) -> float:
    """Compute the average return over the tail window."""
    if not points:
        return 0.0
    tail = points[-max(1, int(window)) :]
    return float(sum(p.value for p in tail) / len(tail))


def curve_auc(points: list[CurvePoint]) -> float:
    """Compute normalized trapezoidal AUC over the return curve."""
    if not points:
        return 0.0
    if len(points) == 1:
        return float(points[0].value)
    area = 0.0
    for left, right in zip(points[:-1], points[1:], strict=False):
        dx = max(0.0, right.step - left.step)
        area += 0.5 * dx * (left.value + right.value)
    span = max(1e-8, points[-1].step - points[0].step)
    return float(area / span)


def write_metrics(
    *,
    metrics_out: Path,
    adapter: str,
    task_id: str,
    seed: int,
    device: str,
    points: list[CurvePoint],
    final_return_mean: float,
    auc_return: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write normalized parity metrics JSON."""
    payload: dict[str, Any] = {
        "schema_version": "parity.v1",
        "adapter": adapter,
        "task_id": task_id,
        "seed": int(seed),
        "device": device,
        "final_return_mean": float(final_return_mean),
        "auc_return": float(auc_return),
        "num_curve_points": len(points),
        "curve": [{"step": float(p.step), "return": float(p.value)} for p in points],
        "success": True,
    }
    if metadata:
        payload["metadata"] = metadata

    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def deterministic_mock_curve(
    *,
    seed: int,
    steps: int,
    family: str,
    system: str,
    points: int = 16,
) -> list[CurvePoint]:
    """Generate deterministic synthetic curves for CI/smoke tests."""
    rng = random.Random((seed + 17) * 997)
    family_scale = 40.0 if family == "dreamerv3" else 180.0
    progress_scale = 55.0 if family == "dreamerv3" else 320.0
    system_bias = 0.0 if system == "official" else -0.4

    out: list[CurvePoint] = []
    for idx in range(points):
        frac = idx / max(1, points - 1)
        step = float(frac * max(1, steps))
        trend = family_scale + progress_scale * math.log1p(3.0 * frac)
        noise = rng.uniform(-2.5, 2.5)
        value = trend + system_bias + noise
        out.append(CurvePoint(step=step, value=float(value)))
    return out
