"""Trainer for WorldLoom."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler

from worldloom.core.exceptions import CheckpointError, ConfigurationError, TrainingError

from .callbacks import Callback, CallbackList, CheckpointCallback, LoggingCallback
from .config import TrainingConfig
from .data import ReplayBuffer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class TrainingState:
    """Mutable state during training."""

    def __init__(self) -> None:
        self.global_step: int = 0
        self.epoch: int = 0
        self.best_loss: float = float("inf")
        self.total_loss: float = 0.0
        self.num_batches: int = 0
        self.should_stop: bool = False
        self.metrics: dict[str, float] = {}

    def reset_epoch(self) -> None:
        """Reset per-epoch statistics."""
        self.total_loss = 0.0
        self.num_batches = 0
        self.metrics = {}


class Trainer:
    """
    HuggingFace-style trainer for WorldLoom.

    Provides a simple interface for training world models with:
    - Automatic device placement
    - Gradient clipping
    - Checkpointing
    - Logging (console and optional wandb)
    - Learning rate scheduling

    Args:
        model: World model to train (must implement compute_loss).
        config: Training configuration.
        callbacks: List of callbacks for logging/checkpointing.
        optimizer: Optional custom optimizer.
        scheduler: Optional learning rate scheduler.

    Example:
        >>> from worldloom import create_world_model
        >>> from worldloom.training import Trainer, TrainingConfig, ReplayBuffer
        >>>
        >>> model = create_world_model("dreamerv3:size12m")
        >>> buffer = ReplayBuffer.load("data.npz")
        >>> config = TrainingConfig(total_steps=50_000, batch_size=32)
        >>>
        >>> trainer = Trainer(model, config)
        >>> trainer.train(buffer)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig | None = None,
        callbacks: list[Callback] | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
    ):
        self.config = config or TrainingConfig()
        self.device = torch.device(self.config.resolve_device())

        # Move model to device
        self.model = model.to(self.device)

        # Setup optimizer
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer

        # Setup scheduler
        self.scheduler = scheduler

        # Setup callbacks
        default_callbacks = [
            LoggingCallback(log_interval=self.config.log_interval),
            CheckpointCallback(
                save_interval=self.config.save_interval,
                output_dir=self.config.output_dir,
            ),
        ]
        if callbacks:
            default_callbacks.extend(callbacks)
        self.callbacks = CallbackList(default_callbacks)

        # Training state
        self.state = TrainingState()

        # Mixed precision
        self.scaler = torch.amp.GradScaler() if self.config.mixed_precision else None

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

    def _create_optimizer(self) -> Optimizer:
        """Create optimizer based on config."""
        params = self.model.parameters()

        if self.config.optimizer == "adamw":
            return AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adam":
            return torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
            )
        else:
            raise ConfigurationError(
                f"Unknown optimizer: '{self.config.optimizer}'. "
                f"Supported optimizers: 'adamw', 'adam', 'sgd'"
            )

    def train(
        self,
        data: ReplayBuffer,
        num_steps: int | None = None,
        resume_from: str | None = None,
    ) -> nn.Module:
        """
        Train the model.

        Args:
            data: ReplayBuffer containing training data.
            num_steps: Number of steps to train. If None, uses config.total_steps.
            resume_from: Path to checkpoint to resume from.

        Returns:
            Trained model.
        """
        total_steps = num_steps or self.config.total_steps

        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)

        logger.info(f"Starting training for {total_steps} steps")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Sequence length: {self.config.sequence_length}")

        self.callbacks.on_train_begin(self)

        try:
            while self.state.global_step < total_steps and not self.state.should_stop:
                try:
                    self._train_step(data)
                except RuntimeError as e:
                    raise TrainingError(
                        f"Training step failed at step {self.state.global_step}: {e}"
                    ) from e

                self.state.global_step += 1

                # Callbacks
                self.callbacks.on_step_end(self)

                # Check for early stopping
                if self.state.should_stop:
                    logger.info("Early stopping triggered")
                    break

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")

        self.callbacks.on_train_end(self)

        # Save final checkpoint
        self.save_checkpoint(os.path.join(self.config.output_dir, "checkpoint_final.pt"))

        return self.model

    def _train_step(self, data: ReplayBuffer) -> dict[str, float]:
        """Execute a single training step."""
        self.model.train()

        # Sample batch
        batch = data.sample(
            batch_size=self.config.batch_size,
            seq_len=self.config.sequence_length,
            device=self.device,
        )

        # Forward pass
        self.optimizer.zero_grad()

        if self.config.mixed_precision and self.scaler is not None:
            with torch.amp.autocast(device_type=self.device.type):
                losses = self.model.compute_loss(batch)  # type: ignore[attr-defined, operator]
                loss = losses["loss"]

            self.scaler.scale(loss).backward()  # type: ignore[union-attr]

            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses = self.model.compute_loss(batch)  # type: ignore[attr-defined, operator]
            loss = losses["loss"]

            loss.backward()

            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip,
                )

            self.optimizer.step()

        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        # Update state
        loss_value = loss.item()
        self.state.total_loss += loss_value
        self.state.num_batches += 1
        self.state.metrics = {k: v.item() for k, v in losses.items()}

        return self.state.metrics

    def evaluate(
        self,
        data: ReplayBuffer,
        num_batches: int = 10,
    ) -> dict[str, float]:
        """
        Evaluate the model on data.

        Args:
            data: ReplayBuffer containing evaluation data.
            num_batches: Number of batches to evaluate.

        Returns:
            Dictionary of average metrics.
        """
        self.model.eval()

        total_metrics: dict[str, float] = {}

        with torch.no_grad():
            for _ in range(num_batches):
                batch = data.sample(
                    batch_size=self.config.batch_size,
                    seq_len=self.config.sequence_length,
                    device=self.device,
                )
                losses = self.model.compute_loss(batch)  # type: ignore[attr-defined, operator]

                for k, v in losses.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v.item()

        # Average
        return {k: v / num_batches for k, v in total_metrics.items()}

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.state.global_step,
            "best_loss": self.state.best_loss,
            "config": self.config.to_dict(),
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save model config if available
        if hasattr(self.model, "config") and hasattr(self.model.config, "to_dict"):  # type: ignore[union-attr, operator]
            checkpoint["model_config"] = self.model.config.to_dict()  # type: ignore[union-attr, operator]

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to checkpoint file.

        Raises:
            CheckpointError: If checkpoint file is missing or corrupted.
        """
        if not Path(path).exists():
            raise CheckpointError(f"Checkpoint file not found: {path}")

        try:
            # Note: weights_only=False is required to load optimizer states.
            # Only load checkpoints from trusted sources.
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint from {path}: {e}") from e

        try:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.state.global_step = checkpoint["global_step"]
            self.state.best_loss = checkpoint.get("best_loss", float("inf"))

            if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            if self.scaler is not None and "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        except KeyError as e:
            raise CheckpointError(
                f"Checkpoint is missing required key: {e}. "
                f"The checkpoint may be corrupted or from an incompatible version."
            ) from e

        logger.info(f"Loaded checkpoint from {path} (step {self.state.global_step})")


def train(
    model: nn.Module,
    data: ReplayBuffer,
    total_steps: int | None = None,
    batch_size: int = 16,
    sequence_length: int = 50,
    learning_rate: float = 3e-4,
    grad_clip: float = 100.0,
    output_dir: str = "./outputs",
    device: str = "auto",
    **kwargs: Any,
) -> nn.Module:
    """
    One-liner training function for quick experimentation.

    Args:
        model: World model to train.
        data: ReplayBuffer containing training data.
        total_steps: Number of training steps.
        batch_size: Batch size.
        sequence_length: Sequence length for trajectory sampling.
        learning_rate: Learning rate.
        grad_clip: Gradient clipping value.
        output_dir: Directory for outputs.
        device: Device to train on.
        **kwargs: Additional config options.

    Returns:
        Trained model.

    Example:
        >>> from worldloom import create_world_model
        >>> from worldloom.training import train, ReplayBuffer
        >>>
        >>> model = create_world_model("dreamerv3:size12m")
        >>> buffer = ReplayBuffer.load("data.npz")
        >>> trained_model = train(model, buffer, total_steps=50_000)
    """
    config = TrainingConfig(
        total_steps=total_steps or 100_000,
        batch_size=batch_size,
        sequence_length=sequence_length,
        learning_rate=learning_rate,
        grad_clip=grad_clip,
        output_dir=output_dir,
        device=device,
        **kwargs,
    )

    trainer = Trainer(model, config)
    return trainer.train(data)
