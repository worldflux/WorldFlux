"""Tests for paper baseline data and comparison logic."""

from __future__ import annotations

from worldflux.parity.paper_baselines import (
    DREAMERV3_ATARI100K_BASELINES,
    SUITE_BASELINES,
    TDMPC2_DMCONTROL39_BASELINES,
)
from worldflux.parity.paper_comparison import (
    compare_against_paper,
)


class TestPaperBaselineData:
    def test_dreamerv3_baseline_count(self) -> None:
        assert len(DREAMERV3_ATARI100K_BASELINES) >= 24

    def test_tdmpc2_baseline_count(self) -> None:
        assert len(TDMPC2_DMCONTROL39_BASELINES) >= 30

    def test_all_scores_positive(self) -> None:
        for name, baseline in DREAMERV3_ATARI100K_BASELINES.items():
            assert baseline.score > 0, f"DreamerV3 {name} score must be positive"
        for name, baseline in TDMPC2_DMCONTROL39_BASELINES.items():
            assert baseline.score > 0, f"TD-MPC2 {name} score must be positive"

    def test_all_sources_non_empty(self) -> None:
        for name, baseline in DREAMERV3_ATARI100K_BASELINES.items():
            assert len(baseline.source) > 10, f"DreamerV3 {name} source too short"
        for name, baseline in TDMPC2_DMCONTROL39_BASELINES.items():
            assert len(baseline.source) > 10, f"TD-MPC2 {name} source too short"

    def test_task_name_matches_key(self) -> None:
        for key, baseline in DREAMERV3_ATARI100K_BASELINES.items():
            assert baseline.task == key
        for key, baseline in TDMPC2_DMCONTROL39_BASELINES.items():
            assert baseline.task == key

    def test_suite_baselines_registry(self) -> None:
        assert "dreamerv3_atari100k" in SUITE_BASELINES
        assert "tdmpc2_dmcontrol39" in SUITE_BASELINES


class TestCompareAgainstPaper:
    def test_returns_none_for_unknown_suite(self) -> None:
        result = compare_against_paper("nonexistent_suite", {"t1": 100.0})
        assert result is None

    def test_returns_none_for_no_matching_tasks(self) -> None:
        result = compare_against_paper("dreamerv3_atari100k", {"fake_task": 100.0})
        assert result is None

    def test_basic_comparison(self) -> None:
        scores = {"alien": 650.0, "pong": 14.0}
        report = compare_against_paper("dreamerv3_atari100k", scores)
        assert report is not None
        assert len(report.deltas) == 2
        assert report.suite_id == "dreamerv3_atari100k"

    def test_delta_computation(self) -> None:
        scores = {"alien": 700.0}
        report = compare_against_paper("dreamerv3_atari100k", scores)
        assert report is not None
        delta = report.deltas[0]
        assert delta.task == "alien"
        assert delta.paper_score == 600.0
        assert delta.run_score == 700.0
        assert delta.absolute_delta == 100.0
        expected_rel = 100.0 / 600.0 * 100.0
        assert abs(delta.relative_delta_pct - expected_rel) < 0.01

    def test_within_5pct_count(self) -> None:
        scores = {"alien": 625.0, "pong": 11.0}
        report = compare_against_paper("dreamerv3_atari100k", scores)
        assert report is not None
        assert report.tasks_within_5pct == 1

    def test_render_markdown(self) -> None:
        scores = {"alien": 650.0}
        report = compare_against_paper("dreamerv3_atari100k", scores)
        assert report is not None
        md = report.render_markdown()
        assert "## Paper Baseline Comparison" in md
        assert "alien" in md
        assert "650.0" in md
        assert "Mean relative delta" in md

    def test_tdmpc2_comparison(self) -> None:
        scores = {"cheetah-run": 790.0, "walker-walk": 940.0}
        report = compare_against_paper("tdmpc2_dmcontrol39", scores)
        assert report is not None
        assert len(report.deltas) == 2
