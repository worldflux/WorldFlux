"""Tests for TrainingReportCallback."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from worldflux.training import TrainingConfig
from worldflux.training import callbacks as callbacks_module
from worldflux.training.callbacks import TrainingReportCallback
from worldflux.training.report import HealthSignal, LossCurveSummary


class _TrainerStub:
    def __init__(self, output_dir: Path):
        self.config = TrainingConfig(total_steps=10, output_dir=str(output_dir))
        self.model = torch.nn.Linear(2, 1)
        self.state = SimpleNamespace(
            global_step=0,
            epoch=0,
            metrics={},
            best_loss=float("inf"),
            should_stop=False,
            train_start_time=None,
            ttfi_sec=0.0,
        )
        self._profile: dict[str, float | None] = {"throughput_steps_per_sec": None}

    def runtime_profile(self) -> dict[str, float | None]:
        return dict(self._profile)


# -- end-to-end flow --


def test_report_callback_saves_json(tmp_path: Path) -> None:
    """Full lifecycle: begin → step (with loss) → end produces a valid JSON report."""
    out = tmp_path / "outputs"
    trainer = _TrainerStub(out)
    cb = TrainingReportCallback(output_dir=str(out))

    cb.on_train_begin(trainer)

    # Simulate a few steps with loss
    for step in range(1, 6):
        trainer.state.global_step = step
        trainer.state.metrics = {"loss": 5.0 - step * 0.5}
        cb.on_step_end(trainer)

    cb.on_train_end(trainer)

    report_path = out / "training_report.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text())
    assert data["model_id"] == "Linear"
    assert data["total_steps"] == 5
    assert data["final_loss"] == pytest.approx(2.5)
    assert data["best_loss"] == pytest.approx(2.5)
    assert data["health_score"] >= 0.0


def test_report_callback_emits_wasr_event(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """on_train_end writes a WASR run.summary event when metrics_path is set."""
    out = tmp_path / "outputs"
    metrics_path = tmp_path / "metrics.jsonl"
    trainer = _TrainerStub(out)
    cb = TrainingReportCallback(
        output_dir=str(out),
        metrics_path=metrics_path,
        run_id="report-run",
        scenario="test",
    )

    events: list[dict[str, object]] = []
    monkeypatch.setattr(callbacks_module, "write_event", lambda **kw: events.append(kw))

    cb.on_train_begin(trainer)
    trainer.state.global_step = 1
    trainer.state.metrics = {"loss": 1.0}
    cb.on_step_end(trainer)
    cb.on_train_end(trainer)

    assert len(events) == 1
    assert events[0]["event"] == "run.summary"
    assert events[0]["run_id"] == "report-run"


def test_report_callback_no_wasr_without_metrics_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No WASR event when metrics_path is None."""
    out = tmp_path / "outputs"
    trainer = _TrainerStub(out)
    cb = TrainingReportCallback(output_dir=str(out), metrics_path=None)

    events: list[dict[str, object]] = []
    monkeypatch.setattr(callbacks_module, "write_event", lambda **kw: events.append(kw))

    cb.on_train_begin(trainer)
    cb.on_train_end(trainer)
    assert events == []


# -- _compute_loss_summary --


def test_loss_summary_empty_history() -> None:
    cb = TrainingReportCallback()
    summary = cb._compute_loss_summary()
    assert summary.initial_loss == 0.0
    assert summary.convergence_slope == 0.0
    assert summary.plateau_detected is False


def test_loss_summary_single_value() -> None:
    cb = TrainingReportCallback()
    cb._loss_history = [3.14]
    summary = cb._compute_loss_summary()
    assert summary.initial_loss == pytest.approx(3.14)
    assert summary.final_loss == pytest.approx(3.14)
    assert summary.convergence_slope == 0.0


def test_loss_summary_decreasing() -> None:
    cb = TrainingReportCallback()
    cb._loss_history = [float(10 - i) for i in range(10)]
    summary = cb._compute_loss_summary()
    assert summary.convergence_slope < 0
    assert summary.best_loss == pytest.approx(1.0)
    assert summary.best_step == 9


def test_loss_summary_plateau_detected() -> None:
    """Plateau detection when recent and previous windows have similar loss."""
    cb = TrainingReportCallback()
    # 40 identical values — relative change is 0 → plateau
    cb._loss_history = [1.0] * 40
    summary = cb._compute_loss_summary()
    assert summary.plateau_detected is True


def test_loss_summary_no_plateau_when_diverging() -> None:
    cb = TrainingReportCallback()
    # Linearly increasing loss → window comparison shows large relative change → no plateau
    cb._loss_history = [float(i) for i in range(1, 41)]
    summary = cb._compute_loss_summary()
    assert summary.plateau_detected is False


# -- _compute_health_signals --


def test_health_signals_no_loss_is_critical() -> None:
    cb = TrainingReportCallback()
    summary = LossCurveSummary(
        initial_loss=0,
        final_loss=0,
        best_loss=0,
        best_step=0,
        convergence_slope=0,
        plateau_detected=False,
    )
    signals = cb._compute_health_signals(summary)
    assert signals["loss_convergence"].status == "critical"


def test_health_signals_increasing_loss_is_warning() -> None:
    cb = TrainingReportCallback()
    cb._loss_history = [1.0, 2.0, 3.0]
    summary = LossCurveSummary(
        initial_loss=1,
        final_loss=3,
        best_loss=1,
        best_step=0,
        convergence_slope=1.0,
        plateau_detected=False,
    )
    signals = cb._compute_health_signals(summary)
    assert signals["loss_convergence"].status == "warning"


def test_health_signals_decreasing_loss_is_healthy() -> None:
    cb = TrainingReportCallback()
    cb._loss_history = [3.0, 2.0, 1.0]
    summary = LossCurveSummary(
        initial_loss=3,
        final_loss=1,
        best_loss=1,
        best_step=2,
        convergence_slope=-1.0,
        plateau_detected=False,
    )
    signals = cb._compute_health_signals(summary)
    assert signals["loss_convergence"].status == "healthy"


def test_health_signals_non_finite_critical() -> None:
    cb = TrainingReportCallback()
    cb._loss_history = [1.0]
    cb._non_finite_count = 5  # >10% of total (6)
    summary = LossCurveSummary(
        initial_loss=1,
        final_loss=1,
        best_loss=1,
        best_step=0,
        convergence_slope=0,
        plateau_detected=False,
    )
    signals = cb._compute_health_signals(summary)
    assert signals["numerical_stability"].status == "critical"


def test_health_signals_non_finite_warning() -> None:
    cb = TrainingReportCallback()
    cb._loss_history = [1.0] * 100
    cb._non_finite_count = 1  # >0 but <=10%
    summary = LossCurveSummary(
        initial_loss=1,
        final_loss=1,
        best_loss=1,
        best_step=0,
        convergence_slope=0,
        plateau_detected=False,
    )
    signals = cb._compute_health_signals(summary)
    assert signals["numerical_stability"].status == "warning"


def test_health_signals_throughput_stable() -> None:
    cb = TrainingReportCallback()
    cb._loss_history = [1.0]
    cb._throughput_history = [10.0, 10.1, 9.9]
    summary = LossCurveSummary(
        initial_loss=1,
        final_loss=1,
        best_loss=1,
        best_step=0,
        convergence_slope=-0.1,
        plateau_detected=False,
    )
    signals = cb._compute_health_signals(summary)
    assert signals["throughput_stability"].status == "healthy"


def test_health_signals_throughput_unstable() -> None:
    cb = TrainingReportCallback()
    cb._loss_history = [1.0]
    cb._throughput_history = [1.0, 100.0]  # CV >> 0.5
    summary = LossCurveSummary(
        initial_loss=1,
        final_loss=1,
        best_loss=1,
        best_step=0,
        convergence_slope=-0.1,
        plateau_detected=False,
    )
    signals = cb._compute_health_signals(summary)
    assert signals["throughput_stability"].status == "warning"


def test_health_signals_insufficient_throughput_data() -> None:
    cb = TrainingReportCallback()
    cb._loss_history = [1.0]
    cb._throughput_history = [10.0]  # < 2 samples
    summary = LossCurveSummary(
        initial_loss=1,
        final_loss=1,
        best_loss=1,
        best_step=0,
        convergence_slope=-0.1,
        plateau_detected=False,
    )
    signals = cb._compute_health_signals(summary)
    assert signals["throughput_stability"].status == "healthy"
    assert "Insufficient" in signals["throughput_stability"].message


# -- _compute_health_score --


def test_health_score_empty_signals() -> None:
    assert TrainingReportCallback._compute_health_score({}) == 0.0


def test_health_score_all_healthy() -> None:
    signals = {
        "loss_convergence": HealthSignal("loss_convergence", "healthy", 1.0, "ok"),
        "numerical_stability": HealthSignal("numerical_stability", "healthy", 1.0, "ok"),
    }
    score = TrainingReportCallback._compute_health_score(signals)
    assert score == pytest.approx(1.0)


def test_health_score_mixed() -> None:
    signals = {
        "loss_convergence": HealthSignal("loss_convergence", "critical", 0.0, "bad"),
    }
    score = TrainingReportCallback._compute_health_score(signals)
    assert score == pytest.approx(0.0)


# -- _generate_recommendations --


def test_recommendations_from_critical_and_plateau() -> None:
    signals = {
        "stability": HealthSignal("stability", "critical", 0.0, "Non-finite values"),
        "throughput": HealthSignal("throughput", "warning", 0.5, "High CV"),
    }
    summary = LossCurveSummary(
        initial_loss=1,
        final_loss=1,
        best_loss=1,
        best_step=0,
        convergence_slope=0,
        plateau_detected=True,
    )
    recs = TrainingReportCallback._generate_recommendations(signals, summary)
    assert any("[CRITICAL]" in r for r in recs)
    assert any("[WARNING]" in r for r in recs)
    assert any("plateau" in r.lower() for r in recs)


def test_recommendations_empty_when_healthy() -> None:
    signals = {
        "loss_convergence": HealthSignal("loss_convergence", "healthy", 1.0, "ok"),
    }
    summary = LossCurveSummary(
        initial_loss=1,
        final_loss=0.5,
        best_loss=0.5,
        best_step=9,
        convergence_slope=-0.1,
        plateau_detected=False,
    )
    recs = TrainingReportCallback._generate_recommendations(signals, summary)
    assert recs == []


# -- on_step_end non-finite loss --


def test_step_end_tracks_non_finite_loss(tmp_path: Path) -> None:
    trainer = _TrainerStub(tmp_path)
    cb = TrainingReportCallback(output_dir=str(tmp_path))

    cb.on_train_begin(trainer)

    trainer.state.global_step = 1
    trainer.state.metrics = {"loss": float("inf")}
    cb.on_step_end(trainer)

    assert cb._non_finite_count == 1
    assert len(cb._loss_history) == 0


# -- on_step_end collects throughput at step % 100 == 0 --


def test_step_end_collects_throughput(tmp_path: Path) -> None:
    trainer = _TrainerStub(tmp_path)
    trainer._profile = {"throughput_steps_per_sec": 42.0}
    cb = TrainingReportCallback(output_dir=str(tmp_path))

    cb.on_train_begin(trainer)

    trainer.state.global_step = 100
    trainer.state.metrics = {"loss": 1.0}
    cb.on_step_end(trainer)

    assert cb._throughput_history == [42.0]


def test_step_end_skips_zero_throughput(tmp_path: Path) -> None:
    trainer = _TrainerStub(tmp_path)
    trainer._profile = {"throughput_steps_per_sec": 0.0}
    cb = TrainingReportCallback(output_dir=str(tmp_path))

    cb.on_train_begin(trainer)

    trainer.state.global_step = 100
    trainer.state.metrics = {"loss": 1.0}
    cb.on_step_end(trainer)

    assert cb._throughput_history == []


# -- _build_report throughput fallback --


def test_build_report_throughput_fallback(tmp_path: Path) -> None:
    """When no throughput_history, falls back to total_steps / wall_time."""
    trainer = _TrainerStub(tmp_path)
    cb = TrainingReportCallback(output_dir=str(tmp_path))
    cb._start_time = 0.0  # will have positive wall_time
    cb._loss_history = [2.0, 1.0]
    trainer.state.global_step = 2

    report = cb._build_report(trainer)
    assert report.throughput_steps_per_sec > 0
