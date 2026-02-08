"""Validation-focused tests for training callbacks."""

from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from worldflux.training import TrainingConfig
from worldflux.training.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    ProgressCallback,
)


class _TrainerStub:
    def __init__(self, output_dir: Path):
        self.config = TrainingConfig(total_steps=10, output_dir=str(output_dir))
        self.state = SimpleNamespace(
            global_step=0,
            metrics={},
            best_loss=float("inf"),
            should_stop=False,
        )
        self.saved_checkpoints: list[str] = []

    def save_checkpoint(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("checkpoint", encoding="utf-8")
        self.saved_checkpoints.append(path)


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
