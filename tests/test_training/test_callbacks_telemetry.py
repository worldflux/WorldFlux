"""Tests for telemetry-oriented training callbacks."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from worldflux.telemetry.wasr import read_events
from worldflux.training import TrainingConfig
from worldflux.training.callbacks import DiagnosisCallback, HeartbeatCallback


class _TrainerStub:
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


def test_heartbeat_callback_emits_jsonl_event(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    trainer = _TrainerStub(tmp_path)
    callback = HeartbeatCallback(
        interval_steps=2,
        metrics_path=metrics_path,
        run_id="heartbeat-run",
        scenario="trainer",
    )

    callback.on_train_begin(trainer)
    trainer.state.global_step = 1
    trainer.state.metrics = {"flops_estimate": 100.0, "power_watts": 50.0}
    callback.on_step_end(trainer)
    assert read_events(metrics_path) == []

    trainer.state.global_step = 2
    callback.on_step_end(trainer)
    events = read_events(metrics_path)
    assert len(events) == 1
    event = events[0]
    assert event["event"] == "heartbeat"
    assert event["run_id"] == "heartbeat-run"
    assert event["step"] == 2
    assert event["flops_per_watt"] == 2.0


def test_diagnosis_callback_detects_vanishing_gradients_and_latent_collapse(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "metrics.jsonl"
    trainer = _TrainerStub(tmp_path)
    callback = DiagnosisCallback(
        check_interval=1,
        gradient_min_norm=1e-4,
        latent_std_min=1e-6,
        metrics_path=metrics_path,
        run_id="diagnostic-run",
        scenario="trainer",
    )

    callback.on_train_begin(trainer)
    trainer.state.global_step = 1
    trainer.state.metrics = {"loss": 1.0, "latent_std": 0.0}

    for param in trainer.model.parameters():
        param.grad = torch.zeros_like(param)

    callback.on_step_end(trainer)
    assert callback.last_suggestions
    assert any("vanishing gradients" in s.lower() for s in callback.last_suggestions)
    assert any("latent collapse" in s.lower() for s in callback.last_suggestions)

    events = read_events(metrics_path)
    assert len(events) == 1
    event = events[0]
    assert event["event"] == "diagnostic"
    assert event["run_id"] == "diagnostic-run"
    assert event["suggestions"]
