"""Tests for training report generation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from worldflux.telemetry.wasr import read_events
from worldflux.training import TrainingConfig
from worldflux.training.callbacks import TrainingReportCallback
from worldflux.training.report import HealthSignal, LossCurveSummary, TrainingReport


class TestHealthSignal:
    def test_creation(self):
        signal = HealthSignal(
            name="loss_convergence",
            status="healthy",
            value=0.95,
            message="Loss is decreasing.",
        )
        assert signal.name == "loss_convergence"
        assert signal.status == "healthy"
        assert signal.value == 0.95
        assert signal.message == "Loss is decreasing."

    def test_to_dict(self):
        signal = HealthSignal(
            name="numerical_stability",
            status="warning",
            value=0.8,
            message="Some non-finite values.",
        )
        d = signal.to_dict()
        assert d == {
            "name": "numerical_stability",
            "status": "warning",
            "value": 0.8,
            "message": "Some non-finite values.",
        }

    def test_frozen(self):
        signal = HealthSignal(name="test", status="healthy", value=1.0, message="ok")
        try:
            signal.name = "changed"  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass


class TestLossCurveSummary:
    def test_creation(self):
        summary = LossCurveSummary(
            initial_loss=2.0,
            final_loss=0.5,
            best_loss=0.4,
            best_step=800,
            convergence_slope=-0.002,
            plateau_detected=False,
        )
        assert summary.initial_loss == 2.0
        assert summary.final_loss == 0.5
        assert summary.best_loss == 0.4
        assert summary.best_step == 800
        assert summary.convergence_slope == -0.002
        assert summary.plateau_detected is False

    def test_to_dict(self):
        summary = LossCurveSummary(
            initial_loss=1.0,
            final_loss=0.1,
            best_loss=0.1,
            best_step=500,
            convergence_slope=-0.001,
            plateau_detected=True,
        )
        d = summary.to_dict()
        assert d["initial_loss"] == 1.0
        assert d["final_loss"] == 0.1
        assert d["plateau_detected"] is True


class TestTrainingReport:
    def test_creation_and_to_dict(self):
        signal = HealthSignal(name="test", status="healthy", value=1.0, message="ok")
        summary = LossCurveSummary(
            initial_loss=2.0,
            final_loss=0.5,
            best_loss=0.4,
            best_step=800,
            convergence_slope=-0.002,
            plateau_detected=False,
        )
        report = TrainingReport(
            model_id="DummyModel",
            total_steps=1000,
            wall_time_sec=60.0,
            final_loss=0.5,
            best_loss=0.4,
            ttfi_sec=0.1,
            throughput_steps_per_sec=16.7,
            health_score=0.9,
            health_signals={"test": signal},
            loss_curve_summary=summary,
            recommendations=["Keep training."],
        )
        d = report.to_dict()
        assert d["model_id"] == "DummyModel"
        assert d["total_steps"] == 1000
        assert d["health_score"] == 0.9
        assert "test" in d["health_signals"]
        assert d["loss_curve_summary"]["best_step"] == 800
        assert d["recommendations"] == ["Keep training."]

    def test_save(self, tmp_path: Path):
        report = TrainingReport(
            model_id="TestModel",
            total_steps=100,
            wall_time_sec=10.0,
            final_loss=1.0,
            best_loss=0.8,
            ttfi_sec=0.05,
            throughput_steps_per_sec=10.0,
            health_score=1.0,
        )
        path = tmp_path / "sub" / "report.json"
        report.save(path)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["model_id"] == "TestModel"
        assert loaded["total_steps"] == 100


class _TrainerStub:
    """Minimal trainer stub for callback testing."""

    def __init__(self, output_dir: Path):
        self.config = TrainingConfig(total_steps=10, output_dir=str(output_dir))
        self.state = SimpleNamespace(
            global_step=0,
            epoch=0,
            metrics={},
            best_loss=float("inf"),
            should_stop=False,
            train_start_time=None,
            ttfi_sec=0.0,
        )
        self.model = torch.nn.Linear(2, 1)

    def runtime_profile(self) -> dict[str, float | None]:
        return {
            "ttfi_sec": float(self.state.ttfi_sec),
            "elapsed_sec": 1.0,
            "throughput_steps_per_sec": 10.0,
        }


class TestTrainingReportCallback:
    def test_data_collection(self, tmp_path: Path):
        trainer = _TrainerStub(tmp_path)
        callback = TrainingReportCallback(output_dir=tmp_path, run_id="test-run")

        callback.on_train_begin(trainer)

        # Simulate steps with loss values
        for i in range(1, 6):
            trainer.state.global_step = i
            trainer.state.metrics = {"loss": 2.0 - i * 0.2}
            callback.on_step_end(trainer)

        assert len(callback._loss_history) == 5
        assert callback._non_finite_count == 0

    def test_non_finite_loss_tracking(self, tmp_path: Path):
        trainer = _TrainerStub(tmp_path)
        callback = TrainingReportCallback(output_dir=tmp_path)

        callback.on_train_begin(trainer)

        trainer.state.global_step = 1
        trainer.state.metrics = {"loss": float("inf")}
        callback.on_step_end(trainer)

        trainer.state.global_step = 2
        trainer.state.metrics = {"loss": float("nan")}
        callback.on_step_end(trainer)

        trainer.state.global_step = 3
        trainer.state.metrics = {"loss": 1.0}
        callback.on_step_end(trainer)

        assert callback._non_finite_count == 2
        assert len(callback._loss_history) == 1

    def test_report_generation(self, tmp_path: Path):
        trainer = _TrainerStub(tmp_path)
        callback = TrainingReportCallback(output_dir=tmp_path, run_id="test-report")

        callback.on_train_begin(trainer)

        # Simulate decreasing loss over 10 steps
        for i in range(1, 11):
            trainer.state.global_step = i
            trainer.state.metrics = {"loss": 2.0 - i * 0.1}
            callback.on_step_end(trainer)

        callback.on_train_end(trainer)

        report_path = tmp_path / "training_report.json"
        assert report_path.exists()

        data = json.loads(report_path.read_text())
        assert data["model_id"] == "Linear"
        assert data["total_steps"] == 10
        assert data["final_loss"] == 1.0
        assert data["best_loss"] == 1.0
        assert data["health_score"] > 0.0
        assert "loss_convergence" in data["health_signals"]
        assert "numerical_stability" in data["health_signals"]

    def test_wasr_event_emitted(self, tmp_path: Path):
        metrics_path = tmp_path / "metrics.jsonl"
        trainer = _TrainerStub(tmp_path)
        callback = TrainingReportCallback(
            output_dir=tmp_path,
            metrics_path=metrics_path,
            run_id="wasr-test",
            scenario="test",
        )

        callback.on_train_begin(trainer)
        trainer.state.global_step = 5
        trainer.state.metrics = {"loss": 1.0}
        callback.on_step_end(trainer)
        callback.on_train_end(trainer)

        events = read_events(metrics_path)
        assert len(events) == 1
        assert events[0]["event"] == "run.summary"
        assert events[0]["run_id"] == "wasr-test"
        assert events[0]["success"] is True

    def test_no_wasr_event_without_metrics_path(self, tmp_path: Path):
        trainer = _TrainerStub(tmp_path)
        callback = TrainingReportCallback(output_dir=tmp_path)

        callback.on_train_begin(trainer)
        trainer.state.global_step = 1
        trainer.state.metrics = {"loss": 1.0}
        callback.on_step_end(trainer)
        callback.on_train_end(trainer)

        # Report should exist but no WASR event
        report_path = tmp_path / "training_report.json"
        assert report_path.exists()
        # No metrics.jsonl should be created
        assert not (tmp_path / "metrics.jsonl").exists()

    def test_empty_loss_history(self, tmp_path: Path):
        trainer = _TrainerStub(tmp_path)
        callback = TrainingReportCallback(output_dir=tmp_path)

        callback.on_train_begin(trainer)
        # No steps with loss
        trainer.state.global_step = 0
        callback.on_train_end(trainer)

        report_path = tmp_path / "training_report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert data["final_loss"] == 0.0
        assert data["health_signals"]["loss_convergence"]["status"] == "critical"

    def test_throughput_collection(self, tmp_path: Path):
        trainer = _TrainerStub(tmp_path)
        callback = TrainingReportCallback(output_dir=tmp_path)

        callback.on_train_begin(trainer)

        # Step at multiples of 100 to trigger throughput collection
        for step in (100, 200, 300):
            trainer.state.global_step = step
            trainer.state.metrics = {"loss": 1.0}
            callback.on_step_end(trainer)

        assert len(callback._throughput_history) == 3

    def test_health_score_range(self, tmp_path: Path):
        trainer = _TrainerStub(tmp_path)
        callback = TrainingReportCallback(output_dir=tmp_path)

        callback.on_train_begin(trainer)
        for i in range(1, 6):
            trainer.state.global_step = i
            trainer.state.metrics = {"loss": 2.0 - i * 0.2}
            callback.on_step_end(trainer)
        callback.on_train_end(trainer)

        data = json.loads((tmp_path / "training_report.json").read_text())
        assert 0.0 <= data["health_score"] <= 1.0
