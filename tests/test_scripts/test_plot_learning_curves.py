"""Tests for the plot_learning_curves.py CLI script."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent / "scripts" / "parity"


def _ensure_path():
    src = str(_SCRIPT_DIR.parent.parent / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    if str(_SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPT_DIR))


def _load_module():
    _ensure_path()
    spec = importlib.util.spec_from_file_location(
        "plot_learning_curves", _SCRIPT_DIR / "plot_learning_curves.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plot_learning_curves"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_jsonl_record(
    *,
    adapter: str = "official_dreamerv3",
    task_id: str = "walker_walk",
    seed: int = 0,
) -> str:
    curve = [{"step": float(i * 100), "return": float(i * 10 + seed)} for i in range(10)]
    return json.dumps(
        {
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
    )


def _write_test_jsonl(tmp_path: Path) -> Path:
    lines = [
        _make_jsonl_record(adapter="official_dreamerv3", task_id="t1", seed=0),
        _make_jsonl_record(adapter="official_dreamerv3", task_id="t1", seed=1),
        _make_jsonl_record(adapter="worldflux_dreamerv3", task_id="t1", seed=0),
        _make_jsonl_record(adapter="worldflux_dreamerv3", task_id="t1", seed=1),
    ]
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jsonl


class TestPlotLearningCurvesCLI:
    def test_build_parser(self) -> None:
        mod = _load_module()
        parser = mod._build_parser()
        args = parser.parse_args(["--input", "/tmp/data.jsonl"])
        assert args.input == Path("/tmp/data.jsonl")
        assert args.fmt == "png"
        assert args.interactive is False

    def test_parser_format_choices(self) -> None:
        mod = _load_module()
        parser = mod._build_parser()
        for fmt in ("png", "pdf", "svg"):
            args = parser.parse_args(["--input", "/tmp/x", "--format", fmt])
            assert args.fmt == fmt

    def test_system_color(self) -> None:
        mod = _load_module()
        assert mod._system_color("official_dreamerv3") == "#1f77b4"
        assert mod._system_color("worldflux_tdmpc2") == "#ff7f0e"

    def test_system_label(self) -> None:
        mod = _load_module()
        assert mod._system_label("official_dreamerv3") == "official"
        assert mod._system_label("worldflux_tdmpc2") == "worldflux"

    @pytest.mark.skipif(
        not importlib.util.find_spec("matplotlib"),
        reason="matplotlib not installed",
    )
    def test_plot_per_task_creates_files(self, tmp_path: Path) -> None:
        mod = _load_module()
        from worldflux.parity.curves import aggregate_curves, load_curves_from_parity_jsonl

        jsonl = _write_test_jsonl(tmp_path)
        curves = load_curves_from_parity_jsonl(jsonl)
        agg = aggregate_curves(curves, num_interp_points=10)

        out_dir = tmp_path / "plots"
        out_dir.mkdir()
        saved = mod.plot_per_task(agg, output_dir=out_dir, fmt="png")
        assert len(saved) >= 1
        for path in saved:
            assert path.exists()
            assert path.suffix == ".png"

    @pytest.mark.skipif(
        not importlib.util.find_spec("matplotlib"),
        reason="matplotlib not installed",
    )
    def test_plot_suite_grid_creates_file(self, tmp_path: Path) -> None:
        mod = _load_module()
        from worldflux.parity.curves import aggregate_curves, load_curves_from_parity_jsonl

        jsonl = _write_test_jsonl(tmp_path)
        curves = load_curves_from_parity_jsonl(jsonl)
        agg = aggregate_curves(curves, num_interp_points=10)

        out_dir = tmp_path / "grid_plots"
        out_dir.mkdir()
        result = mod.plot_suite_grid(agg, output_dir=out_dir, fmt="png")
        assert result is not None
        assert result.exists()
