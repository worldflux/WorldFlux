"""Validation-focused tests for training callbacks."""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from worldflux.training import TrainingConfig
from worldflux.training import callbacks as callbacks_module
from worldflux.training.callbacks import (
    Callback,
    CallbackList,
    CheckpointCallback,
    DiagnosisCallback,
    EarlyStoppingCallback,
    HeartbeatCallback,
    LoggingCallback,
    ProgressCallback,
)


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
            ttfi_sec=None,
        )
        self._profile: dict[str, float | None] = {"throughput_steps_per_sec": None}
        self.saved_checkpoints: list[str] = []

    def save_checkpoint(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("checkpoint", encoding="utf-8")
        self.saved_checkpoints.append(path)

    def runtime_profile(self) -> dict[str, float | None]:
        return dict(self._profile)


def test_callback_base_and_callback_list_methods_are_noops(tmp_path: Path) -> None:
    trainer = _TrainerStub(tmp_path)
    callback = Callback()

    callback.on_train_begin(trainer)
    callback.on_train_end(trainer)
    callback.on_epoch_begin(trainer)
    callback.on_epoch_end(trainer)
    callback.on_step_begin(trainer)
    callback.on_step_end(trainer)

    callbacks = CallbackList([])
    callbacks.on_train_begin(trainer)
    callbacks.on_train_end(trainer)
    callbacks.on_epoch_begin(trainer)
    callbacks.on_epoch_end(trainer)
    callbacks.on_step_begin(trainer)
    callbacks.on_step_end(trainer)


def test_logging_callback_rejects_non_positive_interval() -> None:
    with pytest.raises(ValueError, match="log_interval"):
        LoggingCallback(log_interval=0)


def test_checkpoint_callback_rejects_invalid_intervals() -> None:
    with pytest.raises(ValueError, match="save_interval"):
        CheckpointCallback(save_interval=0)
    with pytest.raises(ValueError, match="max_checkpoints"):
        CheckpointCallback(save_interval=1, max_checkpoints=0)


def test_early_stopping_callback_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="patience"):
        EarlyStoppingCallback(patience=0)
    with pytest.raises(ValueError, match="min_delta"):
        EarlyStoppingCallback(min_delta=-1e-3)


def test_heartbeat_callback_rejects_invalid_interval() -> None:
    with pytest.raises(ValueError, match="interval_steps"):
        HeartbeatCallback(interval_steps=0)


def test_diagnosis_callback_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="check_interval"):
        DiagnosisCallback(check_interval=0)
    with pytest.raises(ValueError, match="gradient_min_norm"):
        DiagnosisCallback(gradient_min_norm=-1.0)
    with pytest.raises(ValueError, match="latent_std_min"):
        DiagnosisCallback(latent_std_min=-1.0)


def test_logging_callback_logs_metrics_without_wandb(tmp_path: Path) -> None:
    trainer = _TrainerStub(tmp_path)
    callback = LoggingCallback(log_interval=1, use_wandb=False)

    callback.on_train_begin(trainer)
    trainer.state.global_step = 1
    trainer.state.metrics = {"loss": 1.23}
    callback.on_step_end(trainer)
    callback.on_train_end(trainer)


def test_logging_callback_disables_wandb_if_import_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trainer = _TrainerStub(tmp_path)
    callback = LoggingCallback(log_interval=1, use_wandb=True)
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "wandb":
            raise ImportError("wandb missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    callback.on_train_begin(trainer)
    assert callback.use_wandb is False


def test_logging_callback_wandb_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    trainer = _TrainerStub(tmp_path)
    calls: dict[str, object] = {}

    class _Run:
        def finish(self) -> None:
            calls["finished"] = True

    run = _Run()

    def _init(*, project: str, name: str | None, config: dict[str, object]) -> _Run:
        calls["init"] = {"project": project, "name": name, "config": config}
        return run

    def _log(payload: dict[str, float]) -> None:
        calls["log"] = payload

    wandb_mod = types.ModuleType("wandb")
    wandb_mod.init = _init  # type: ignore[attr-defined]
    wandb_mod.log = _log  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wandb", wandb_mod)

    callback = LoggingCallback(
        log_interval=1,
        use_wandb=True,
        wandb_project="worldflux-tests",
        wandb_run_name="callbacks",
    )
    callback.on_train_begin(trainer)
    trainer.state.global_step = 1
    trainer.state.metrics = {"loss": 1.5}
    callback.on_step_end(trainer)
    callback.on_train_end(trainer)

    assert calls["finished"] is True
    assert calls["init"] == {
        "project": "worldflux-tests",
        "name": "callbacks",
        "config": trainer.config.to_dict(),
    }
    log_payload = calls["log"]
    assert isinstance(log_payload, dict)
    assert log_payload["step"] == 1
    assert isinstance(log_payload["speed"], float)
    assert log_payload["loss"] == 1.5


def test_checkpoint_callback_saves_and_cleans_up_old_checkpoints(tmp_path: Path) -> None:
    trainer = _TrainerStub(tmp_path)
    callback = CheckpointCallback(
        save_interval=2,
        output_dir=str(tmp_path / "ckpts"),
        save_best=True,
        max_checkpoints=1,
    )

    callback.on_train_begin(trainer)
    trainer.state.global_step = 2
    trainer.state.metrics = {"loss": 2.0}
    callback.on_step_end(trainer)

    trainer.state.global_step = 4
    trainer.state.metrics = {"loss": 1.0}
    callback.on_step_end(trainer)

    ckpt_dir = tmp_path / "ckpts"
    assert (ckpt_dir / "checkpoint_4.pt").exists()
    assert not (ckpt_dir / "checkpoint_2.pt").exists()
    assert (ckpt_dir / "checkpoint_best.pt").exists()


def test_checkpoint_callback_cleanup_is_noop_when_unbounded(tmp_path: Path) -> None:
    callback = CheckpointCallback(
        save_interval=1,
        output_dir=str(tmp_path / "ckpts"),
        save_best=False,
        max_checkpoints=None,
    )
    callback._checkpoint_paths = [tmp_path / "ckpts" / "checkpoint_1.pt"]
    callback._cleanup_old_checkpoints()
    assert len(callback._checkpoint_paths) == 1


def test_early_stopping_callback_sets_should_stop(tmp_path: Path) -> None:
    trainer = _TrainerStub(tmp_path)
    callback = EarlyStoppingCallback(patience=2, min_delta=0.1, monitor="loss")

    trainer.state.metrics = {"loss": 1.0}
    callback.on_step_end(trainer)
    assert trainer.state.should_stop is False

    trainer.state.metrics = {"loss": 0.95}
    callback.on_step_end(trainer)
    assert trainer.state.should_stop is False

    trainer.state.metrics = {"loss": 0.96}
    callback.on_step_end(trainer)
    assert trainer.state.should_stop is True


def test_early_stopping_callback_ignores_missing_metric(tmp_path: Path) -> None:
    trainer = _TrainerStub(tmp_path)
    callback = EarlyStoppingCallback(patience=1, min_delta=0.0, monitor="loss")
    trainer.state.metrics = {"other": 1.0}
    callback.on_step_end(trainer)
    assert trainer.state.should_stop is False


def test_progress_callback_updates_and_closes_progress_bar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: dict[str, object] = {"updated": 0, "closed": False, "postfix": None}

    class _DummyPbar:
        def update(self, amount: int) -> None:
            calls["updated"] = int(calls["updated"]) + amount

        def set_postfix(self, postfix) -> None:
            calls["postfix"] = postfix

        def close(self) -> None:
            calls["closed"] = True

    def _dummy_tqdm(*, total, initial, desc, unit):
        assert total == 10
        assert initial == 0
        assert desc == "Training"
        assert unit == "step"
        return _DummyPbar()

    tqdm_mod = types.ModuleType("tqdm")
    tqdm_mod.tqdm = _dummy_tqdm  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_mod)

    trainer = _TrainerStub(tmp_path)
    callback = ProgressCallback(desc="Training")
    callback.on_train_begin(trainer)

    trainer.state.metrics = {"loss": 0.5}
    callback.on_step_end(trainer)
    callback.on_train_end(trainer)

    assert calls["updated"] == 1
    assert calls["closed"] is True
    assert calls["postfix"] == {"loss": "0.5000"}


def test_heartbeat_safe_float_and_duration(tmp_path: Path) -> None:
    callback = HeartbeatCallback(
        interval_steps=2, metrics_path=tmp_path / "metrics.jsonl", run_id="hb"
    )
    assert callback._safe_float(None) is None
    assert callback._safe_float("not-a-number") is None
    assert callback._safe_float(float("nan")) is None
    assert callback._safe_float("3.5") == 3.5
    assert callback._current_duration() == 0.0

    trainer = _TrainerStub(tmp_path)
    trainer.state.train_start_time = 12.0
    callback.on_train_begin(trainer)
    assert callback._start_time == 12.0


def test_heartbeat_callback_writes_event_on_interval(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trainer = _TrainerStub(tmp_path)
    trainer.state.ttfi_sec = 0.25
    trainer.state.epoch = 3
    trainer._profile = {"throughput_steps_per_sec": 10.0}

    events: list[dict[str, object]] = []

    def _capture_event(**payload: object) -> dict[str, object]:
        events.append(payload)
        return payload

    monkeypatch.setattr(callbacks_module, "write_event", _capture_event)
    callback = HeartbeatCallback(
        interval_steps=2,
        scenario="trainer-test",
        metrics_path=tmp_path / "heartbeat.jsonl",
        run_id="run-hb",
    )
    callback.on_train_begin(trainer)

    trainer.state.global_step = 1
    trainer.state.metrics = {}
    callback.on_step_end(trainer)
    assert events == []

    trainer.state.global_step = 2
    trainer.state.metrics = {"flops_estimate": 200.0, "watts_estimate": 50.0}
    callback.on_step_end(trainer)

    assert len(events) == 1
    payload = events[0]
    assert payload["event"] == "heartbeat"
    assert payload["scenario"] == "trainer-test"
    assert payload["step"] == 2
    assert payload["epoch"] == 3
    assert payload["throughput_steps_per_sec"] == 10.0
    assert payload["flops_estimate"] == 200.0
    assert payload["watts_estimate"] == 50.0
    assert payload["flops_per_watt"] == 4.0


def test_diagnosis_callback_gradient_detection_paths(tmp_path: Path) -> None:
    trainer = _TrainerStub(tmp_path)
    callback = DiagnosisCallback(check_interval=1, gradient_min_norm=0.1, latent_std_min=1e-6)

    assert callback._detect_gradient_issues(trainer) == []

    trainer.model.weight.grad = torch.full_like(trainer.model.weight, 1e-6)
    vanishing = callback._detect_gradient_issues(trainer)
    assert any("vanishing gradients" in item for item in vanishing)

    trainer.model.weight.grad = torch.full_like(trainer.model.weight, float("nan"))
    nan_or_inf = callback._detect_gradient_issues(trainer)
    assert any("NaN/Inf gradients" in item for item in nan_or_inf)


def test_diagnosis_callback_metric_detection_paths(tmp_path: Path) -> None:
    trainer = _TrainerStub(tmp_path)
    callback = DiagnosisCallback(check_interval=1, latent_std_min=0.1)
    trainer.state.metrics = {"loss": float("inf"), "latent_std": 0.0}
    issues = callback._detect_metric_issues(trainer)
    assert any("Loss is non-finite" in item for item in issues)
    assert any("Latent collapse indicator" in item for item in issues)
    assert callback._current_duration() == 0.0


def test_diagnosis_callback_writes_event_when_issues_detected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trainer = _TrainerStub(tmp_path)
    trainer.state.train_start_time = 5.0
    trainer.state.ttfi_sec = 0.2
    trainer.state.epoch = 1
    callback = DiagnosisCallback(
        check_interval=2,
        scenario="diagnostics",
        metrics_path=tmp_path / "diagnostics.jsonl",
        run_id="run-diag",
    )
    callback.on_train_begin(trainer)

    events: list[dict[str, object]] = []

    def _capture_event(**payload: object) -> dict[str, object]:
        events.append(payload)
        return payload

    monkeypatch.setattr(callbacks_module, "write_event", _capture_event)

    trainer.state.global_step = 2
    trainer.state.metrics = {"loss": 1.0, "latent_std": 1.0}
    callback.on_step_end(trainer)
    assert callback.last_suggestions == []
    assert events == []

    trainer.state.global_step = 4
    trainer.state.metrics = {"loss": float("inf"), "latent_std": 0.0}
    callback.on_step_end(trainer)

    assert len(events) == 1
    payload = events[0]
    assert payload["event"] == "diagnostic"
    assert payload["scenario"] == "diagnostics"
    assert payload["success"] is False
    assert payload["step"] == 4
    assert any("Loss is non-finite" in item for item in callback.last_suggestions)


def test_progress_callback_gracefully_handles_missing_tqdm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trainer = _TrainerStub(tmp_path)
    callback = ProgressCallback(desc="Training")
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tqdm":
            raise ImportError("tqdm missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    callback.on_train_begin(trainer)
    callback.on_step_end(trainer)
    callback.on_train_end(trainer)
