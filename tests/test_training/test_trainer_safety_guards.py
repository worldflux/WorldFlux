"""Safety guard tests for Trainer error paths and boundary handling."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from worldflux.core.batch import Batch
from worldflux.core.exceptions import CheckpointError, TrainingError
from worldflux.core.model import WorldModel
from worldflux.core.output import LossOutput
from worldflux.core.state import State
from worldflux.training import Trainer, TrainingConfig


class _MiniModel(WorldModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def encode(self, obs, deterministic: bool = False) -> State:
        if isinstance(obs, dict):
            obs = obs["obs"]
        return State(tensors={"latent": obs.float()})

    def transition(self, state: State, action: torch.Tensor, deterministic: bool = False) -> State:
        return state

    def update(self, state: State, action: torch.Tensor, obs) -> State:
        return self.encode(obs)

    def decode(self, state: State):
        return None

    def loss(self, batch) -> LossOutput:
        obs = batch.obs
        if isinstance(obs, dict):
            obs = obs["obs"]
        if obs.dim() == 3:
            obs = obs[:, 0]
        pred = self.linear(obs.float())
        loss = pred.mean()
        return LossOutput(loss=loss, components={"mini": loss})


def _trainer(tmp_path: Path) -> Trainer:
    model = _MiniModel()
    config = TrainingConfig(
        total_steps=3,
        batch_size=2,
        sequence_length=1,
        output_dir=str(tmp_path),
        device="cpu",
    )
    return Trainer(model, config, callbacks=[])


def test_evaluate_rejects_non_positive_num_batches(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    with pytest.raises(TrainingError, match="num_batches must be positive"):
        trainer.evaluate(data=[], num_batches=0)


def test_load_checkpoint_rejects_missing_file(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    with pytest.raises(CheckpointError, match="not found"):
        trainer.load_checkpoint(str(tmp_path / "missing.pt"))


def test_load_checkpoint_rejects_corrupted_file(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    path = tmp_path / "broken.pt"
    path.write_text("not-a-checkpoint", encoding="utf-8")
    with pytest.raises(CheckpointError, match="Failed to load checkpoint"):
        trainer.load_checkpoint(str(path))


def test_load_checkpoint_rejects_missing_required_keys(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    path = tmp_path / "incomplete.pt"
    torch.save({"model_state_dict": trainer.model.state_dict()}, path)
    with pytest.raises(CheckpointError, match="missing required key"):
        trainer.load_checkpoint(str(path))


def test_check_for_nan_inf_raises_training_error(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    with pytest.raises(TrainingError, match="Numerical instability"):
        trainer._check_for_nan_inf(
            {
                "loss": torch.tensor(float("nan")),
                "reward": torch.tensor(float("inf")),
            },
            step=5,
        )


def test_check_gradients_raises_training_error_on_nan(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    trainer.model.linear.weight.grad = torch.full_like(trainer.model.linear.weight, float("nan"))
    with pytest.raises(TrainingError, match="NaN gradient"):
        trainer._check_gradients(step=1)


def test_next_batch_rejects_provider_with_invalid_return_type(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)

    class _BadProvider:
        def sample(self, batch_size: int, seq_len: int | None = None, device: str = "cpu"):
            return "not-a-batch"

    with pytest.raises(TrainingError, match="must return Batch or dict"):
        trainer._next_batch(_BadProvider())


def test_next_batch_restarts_iterable_after_stop_iteration(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    batches = [{"obs": torch.randn(2, 4)}]

    first = trainer._next_batch(batches)
    second = trainer._next_batch(batches)

    assert isinstance(first, Batch)
    assert isinstance(second, Batch)


def test_save_checkpoint_writes_and_validates_file(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    path = tmp_path / "checkpoint.pt"
    trainer.save_checkpoint(str(path))
    assert path.exists()


def test_sample_from_provider_supports_request_style(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)

    class _RequestProvider:
        def sample(self, request):
            return Batch(obs=torch.randn(request.batch_size, 4))

    batch = trainer._next_batch(_RequestProvider())
    assert isinstance(batch, Batch)


def test_train_step_reports_metrics(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)

    class _Provider:
        def sample(self, batch_size: int, seq_len: int | None = None, device: str = "cpu"):
            return Batch(obs=torch.randn(batch_size, 4, device=device))

    metrics = trainer._train_step(_Provider())
    assert "loss" in metrics
    assert isinstance(metrics["loss"], float)


def test_load_checkpoint_restores_global_step(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)

    class _Provider:
        def sample(self, batch_size: int, seq_len: int | None = None, device: str = "cpu"):
            return Batch(obs=torch.randn(batch_size, 4, device=device))

    trainer._train_step(_Provider())
    trainer.state.global_step = 3
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        checkpoint_path = Path(f.name)
    try:
        trainer.save_checkpoint(str(checkpoint_path))
        reloaded = _trainer(tmp_path)
        reloaded.load_checkpoint(str(checkpoint_path))
        assert reloaded.state.global_step == 3
    finally:
        checkpoint_path.unlink(missing_ok=True)


def test_checkpoint_roundtrip_preserves_scheduler_and_scaler_state(tmp_path: Path) -> None:
    config = TrainingConfig(
        total_steps=3,
        batch_size=2,
        sequence_length=1,
        output_dir=str(tmp_path),
        device="cpu",
        scheduler="constant",
        mixed_precision=True,
    )
    trainer = Trainer(_MiniModel(), config, callbacks=[])
    checkpoint_path = tmp_path / "checkpoint_with_states.pt"
    trainer.save_checkpoint(str(checkpoint_path))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert "scheduler_state_dict" in checkpoint
    assert "scaler_state_dict" in checkpoint

    restored = Trainer(_MiniModel(), config, callbacks=[])
    restored.load_checkpoint(str(checkpoint_path))
