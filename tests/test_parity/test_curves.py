"""Tests for parity curve loading and aggregation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from worldflux.parity.curves import (
    CurveData,
    CurvePoint,
    aggregate_curves,
    load_curves_from_parity_jsonl,
)


# ── helpers ────────────────────────────────────────────────────────────
def _make_record(
    *,
    adapter: str = "official_dreamerv3",
    task_id: str = "walker_walk",
    seed: int = 0,
    curve: list[dict[str, float]] | None = None,
) -> str:
    """Return a single parity.v1 JSON line."""
    if curve is None:
        curve = [{"step": float(i), "return": float(i * 10)} for i in range(5)]
    record = {
        "schema_version": "parity.v1",
        "adapter": adapter,
        "task_id": task_id,
        "seed": seed,
        "device": "cpu",
        "final_return_mean": 0.0,
        "auc_return": 0.0,
        "num_curve_points": len(curve),
        "curve": curve,
        "success": True,
    }
    return json.dumps(record)


def _write_jsonl(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ── load_curves_from_parity_jsonl ──────────────────────────────────────
class TestLoadCurves:
    def test_loads_single_file(self, tmp_path: Path) -> None:
        jsonl = _write_jsonl(tmp_path / "data.jsonl", [_make_record()])
        curves = load_curves_from_parity_jsonl(jsonl)
        assert len(curves) == 1
        assert curves[0].task == "walker_walk"
        assert curves[0].seed == 0
        assert curves[0].system == "official_dreamerv3"
        assert len(curves[0].points) == 5

    def test_loads_directory(self, tmp_path: Path) -> None:
        _write_jsonl(tmp_path / "a.jsonl", [_make_record(seed=1)])
        _write_jsonl(tmp_path / "b.jsonl", [_make_record(seed=2)])
        curves = load_curves_from_parity_jsonl(tmp_path)
        assert len(curves) == 2
        seeds = {c.seed for c in curves}
        assert seeds == {1, 2}

    def test_skips_non_parity_v1(self, tmp_path: Path) -> None:
        bad_line = json.dumps({"schema_version": "other", "adapter": "x"})
        good_line = _make_record()
        jsonl = _write_jsonl(tmp_path / "mixed.jsonl", [bad_line, good_line])
        curves = load_curves_from_parity_jsonl(jsonl)
        assert len(curves) == 1

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        jsonl = _write_jsonl(tmp_path / "blanks.jsonl", ["", _make_record(), "", ""])
        curves = load_curves_from_parity_jsonl(jsonl)
        assert len(curves) == 1

    def test_point_values(self, tmp_path: Path) -> None:
        curve = [{"step": 0.0, "return": 5.0}, {"step": 1.0, "return": 15.0}]
        jsonl = _write_jsonl(tmp_path / "pts.jsonl", [_make_record(curve=curve)])
        curves = load_curves_from_parity_jsonl(jsonl)
        assert curves[0].points[0] == CurvePoint(step=0.0, value=5.0)
        assert curves[0].points[1] == CurvePoint(step=1.0, value=15.0)

    def test_missing_curve_field(self, tmp_path: Path) -> None:
        record = {
            "schema_version": "parity.v1",
            "adapter": "x",
            "task_id": "t",
            "seed": 0,
        }
        jsonl = _write_jsonl(tmp_path / "no_curve.jsonl", [json.dumps(record)])
        curves = load_curves_from_parity_jsonl(jsonl)
        assert len(curves) == 0


# ── aggregate_curves ───────────────────────────────────────────────────
def _make_curve_data(
    *,
    task: str = "walker_walk",
    system: str = "official",
    seed: int = 0,
    steps: list[float] | None = None,
    values: list[float] | None = None,
) -> CurveData:
    if steps is None:
        steps = [0.0, 1.0, 2.0, 3.0, 4.0]
    if values is None:
        values = [float(s * 10) for s in steps]
    points = [CurvePoint(step=s, value=v) for s, v in zip(steps, values)]
    return CurveData(task=task, seed=seed, system=system, points=points)


class TestAggregateCurves:
    def test_single_seed_mean_equals_curve(self) -> None:
        cd = _make_curve_data(steps=[0.0, 10.0], values=[100.0, 200.0])
        agg = aggregate_curves([cd], num_interp_points=5)
        key = ("walker_walk", "official")
        assert key in agg
        np.testing.assert_allclose(agg[key]["mean"][0], 100.0, atol=1.0)
        np.testing.assert_allclose(agg[key]["mean"][-1], 200.0, atol=1.0)

    def test_multiple_seeds_averaged(self) -> None:
        c1 = _make_curve_data(seed=0, steps=[0.0, 10.0], values=[100.0, 200.0])
        c2 = _make_curve_data(seed=1, steps=[0.0, 10.0], values=[200.0, 300.0])
        agg = aggregate_curves([c1, c2], num_interp_points=3)
        key = ("walker_walk", "official")
        # Mean of [100,200] and [200,300] at endpoints
        np.testing.assert_allclose(agg[key]["mean"][0], 150.0, atol=1.0)
        np.testing.assert_allclose(agg[key]["mean"][-1], 250.0, atol=1.0)

    def test_ci_bounds_present(self) -> None:
        c1 = _make_curve_data(seed=0, steps=[0.0, 10.0], values=[100.0, 200.0])
        c2 = _make_curve_data(seed=1, steps=[0.0, 10.0], values=[200.0, 300.0])
        agg = aggregate_curves([c1, c2], num_interp_points=3)
        key = ("walker_walk", "official")
        assert "ci_low" in agg[key]
        assert "ci_high" in agg[key]
        # CI low should be <= mean and CI high >= mean everywhere
        assert np.all(agg[key]["ci_low"] <= agg[key]["mean"] + 1e-10)
        assert np.all(agg[key]["ci_high"] >= agg[key]["mean"] - 1e-10)

    def test_groups_by_task_and_system(self) -> None:
        c1 = _make_curve_data(task="t1", system="A", steps=[0.0, 10.0], values=[1.0, 2.0])
        c2 = _make_curve_data(task="t2", system="B", steps=[0.0, 10.0], values=[3.0, 4.0])
        agg = aggregate_curves([c1, c2])
        assert ("t1", "A") in agg
        assert ("t2", "B") in agg
        assert len(agg) == 2

    def test_interpolation_grid_length(self) -> None:
        cd = _make_curve_data(steps=[0.0, 5.0, 10.0], values=[0.0, 50.0, 100.0])
        n = 50
        agg = aggregate_curves([cd], num_interp_points=n)
        key = ("walker_walk", "official")
        assert len(agg[key]["steps"]) == n
        assert len(agg[key]["mean"]) == n

    def test_skips_single_point_curves(self) -> None:
        cd = _make_curve_data(steps=[5.0], values=[99.0])
        agg = aggregate_curves([cd])
        assert len(agg) == 0

    def test_ci_percentile_custom(self) -> None:
        curves = [
            _make_curve_data(seed=i, steps=[0.0, 10.0], values=[float(i * 10), float(i * 20)])
            for i in range(20)
        ]
        agg90 = aggregate_curves(curves, num_interp_points=5, ci_percentile=90.0)
        agg99 = aggregate_curves(curves, num_interp_points=5, ci_percentile=99.0)
        key = ("walker_walk", "official")
        # 99% CI should be wider than 90% CI
        width90 = agg90[key]["ci_high"] - agg90[key]["ci_low"]
        width99 = agg99[key]["ci_high"] - agg99[key]["ci_low"]
        assert np.all(width99 >= width90 - 1e-10)
