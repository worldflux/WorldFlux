"""Trainer for WorldFlux."""

from __future__ import annotations

import inspect
import logging
import math
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler

from worldflux.core.batch import Batch, BatchProvider, BatchProviderV2, BatchRequest
from worldflux.core.exceptions import (
    BufferError,
    CheckpointError,
    ConfigurationError,
    ShapeMismatchError,
    TrainingError,
    ValidationError,
)
from worldflux.core.spec import ModelIOContract
from worldflux.utils import set_seed

from .callbacks import Callback, CallbackList, CheckpointCallback, LoggingCallback
from .config import TrainingConfig

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
    HuggingFace-style trainer for WorldFlux.

    Provides a simple interface for training world models with:
    - Automatic device placement
    - Gradient clipping
    - Checkpointing
    - Logging (console and optional wandb)
    - Learning rate scheduling

    Args:
        model: World model to train (must implement loss(batch)).
        config: Training configuration.
        callbacks: List of callbacks for logging/checkpointing.
        optimizer: Optional custom optimizer.
        scheduler: Optional learning rate scheduler.

    Example:
        >>> from worldflux import create_world_model
        >>> from worldflux.training import Trainer, TrainingConfig, ReplayBuffer
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
        set_seed(self.config.seed)
        self.device = torch.device(self.config.resolve_device())

        # Move model to device
        self.model = model.to(self.device)

        if self.config.model_overrides:
            raise ConfigurationError(
                "TrainingConfig.model_overrides is not applied by Trainer. "
                "Pass model overrides at model creation time."
            )
        if self.config.ema_decay is not None:
            raise ConfigurationError(
                "TrainingConfig.ema_decay is not implemented yet. Set ema_decay=None."
            )

        # Setup optimizer
        if optimizer is None:
            self.optimizer = self._create_optimizer()
        else:
            self.optimizer = optimizer

        # Setup scheduler
        self.scheduler: LRScheduler | None
        if scheduler is not None:
            self.scheduler = scheduler
        elif self.config.scheduler != "none":
            self.scheduler = self._create_scheduler()
        else:
            self.scheduler = None

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

        # Gradient accumulation counter
        self._accumulation_step = 0

        # Mixed precision
        self.scaler = torch.amp.GradScaler() if self.config.mixed_precision else None

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Data iterator cache (for iterable datasets)
        self._data_iter: Iterator[Any] | None = None

        # Optional model I/O contract for runtime validation.
        contract_fn = getattr(self.model, "io_contract", None)
        self.io_contract: ModelIOContract | None = None
        if callable(contract_fn):
            try:
                maybe_contract = contract_fn()
            except Exception as e:
                raise TrainingError(f"Failed to build model I/O contract: {e}") from e
            if not isinstance(maybe_contract, ModelIOContract):
                raise TrainingError(
                    "Invalid model I/O contract: io_contract() must return ModelIOContract"
                )
            try:
                maybe_contract.validate()
            except ValidationError as e:
                raise TrainingError(f"Invalid model I/O contract: {e}") from e
            self.io_contract = maybe_contract

    def _apply_contract_to_batch(self, batch: Batch, data: BatchProvider | Any) -> Batch:
        if hasattr(data, "batch_layout"):
            try:
                layout = data.batch_layout()  # type: ignore[attr-defined]
            except Exception:
                layout = None
            if isinstance(layout, dict) and layout:
                batch = batch.with_layouts(layout, strict=True)

        if self.io_contract is not None:
            if not batch.layouts and self.io_contract.sequence_layout.axes_by_field:
                batch = batch.with_layouts(
                    self.io_contract.sequence_layout.axes_by_field, strict=True
                )

            missing = []
            for key in self.io_contract.required_batch_keys:
                has_key = False
                has_batch_key = getattr(self.model, "has_batch_key", None)
                if callable(has_batch_key):
                    has_key = bool(has_batch_key(batch, key))
                else:
                    has_key = getattr(batch, key, None) is not None
                if not has_key:
                    missing.append(key)
            if missing:
                raise TrainingError(f"Batch is missing required keys for model contract: {missing}")
            validate_batch_contract = getattr(self.model, "validate_batch_contract", None)
            if callable(validate_batch_contract):
                try:
                    validate_batch_contract(batch)
                except ValidationError as e:
                    raise TrainingError(f"Batch violates model I/O contract: {e}") from e
        return batch

    def _next_batch(self, data: BatchProvider | Any) -> Batch:
        """Fetch the next batch from a provider or iterator."""
        if hasattr(data, "sample"):
            request = BatchRequest(
                batch_size=self.config.batch_size,
                seq_len=self.config.sequence_length,
                device=self.device,
            )
            batch = self._sample_from_provider(cast(BatchProvider | BatchProviderV2, data), request)
            if isinstance(batch, dict):
                batch = Batch.from_dict(batch)
            if not isinstance(batch, Batch):
                raise TrainingError(
                    f"BatchProvider.sample() must return Batch or dict, got {type(batch).__name__}"
                )
            batch = batch.to(self.device)
            batch = self._apply_contract_to_batch(batch, data)
            try:
                sequence_spec = (
                    self.io_contract.effective_sequence_field_spec
                    if self.io_contract is not None
                    else None
                )
                batch.validate(strict_time=True, sequence_field_spec=sequence_spec)
            except (ShapeMismatchError, ValidationError, BufferError) as e:
                raise TrainingError(f"Invalid batch for training: {e}") from e
            return batch

        if self._data_iter is None:
            iterable = cast(Iterable[Any], data)
            self._data_iter = iter(iterable)

        try:
            assert self._data_iter is not None
            batch = next(self._data_iter)
        except StopIteration:
            iterable = cast(Iterable[Any], data)
            self._data_iter = iter(iterable)
            batch = next(self._data_iter)

        if isinstance(batch, dict):
            batch = Batch.from_dict(batch)
        if not isinstance(batch, Batch):
            raise TrainingError(
                f"BatchProvider must yield Batch or dict, got {type(batch).__name__}"
            )
        batch = batch.to(self.device)
        batch = self._apply_contract_to_batch(batch, data)
        try:
            sequence_spec = (
                self.io_contract.effective_sequence_field_spec
                if self.io_contract is not None
                else None
            )
            batch.validate(strict_time=True, sequence_field_spec=sequence_spec)
        except (ShapeMismatchError, ValidationError, BufferError) as e:
            raise TrainingError(f"Invalid batch for training: {e}") from e
        return batch

    @staticmethod
    def _sample_from_provider(
        data: BatchProvider | BatchProviderV2,
        request: BatchRequest,
    ) -> Batch | dict[str, Any]:
        sample_fn = getattr(data, "sample")
        try:
            sig = inspect.signature(sample_fn)
        except (TypeError, ValueError):
            sig = None

        if sig is not None:
            params = list(sig.parameters.values())
            if len(params) == 1 and params[0].kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                return sample_fn(request)

        return sample_fn(
            batch_size=request.batch_size,
            seq_len=request.seq_len,
            device=request.device,
        )

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

    def _create_scheduler(self) -> LRScheduler:
        """Create an LR scheduler based on config.scheduler and warmup settings."""
        total_steps = max(1, self.config.total_steps)
        warmup_steps = min(max(0, self.config.warmup_steps), total_steps - 1)
        schedule_name = self.config.scheduler

        def _lambda(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)

            if schedule_name == "constant":
                return 1.0

            denom = max(1, total_steps - warmup_steps)
            progress = min(max((step - warmup_steps) / denom, 0.0), 1.0)

            if schedule_name == "linear":
                return max(0.0, 1.0 - progress)
            if schedule_name == "cosine":
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return 1.0

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=_lambda)

    def add_callback(self, callback: Callback) -> None:
        """Register a callback after trainer construction."""
        self.callbacks.append(callback)

    def train(
        self,
        data: BatchProvider | Any,
        num_steps: int | None = None,
        resume_from: str | None = None,
    ) -> nn.Module:
        """
        Train the model.

        Args:
            data: BatchProvider or iterable yielding Batch/dict.
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

    def _check_for_nan_inf(self, losses: dict[str, torch.Tensor], step: int) -> None:
        """
        Check for NaN or Inf values in losses and raise an error if detected.

        Args:
            losses: Dictionary of loss tensors to check.
            step: Current training step (for error message).

        Raises:
            TrainingError: If NaN or Inf values are detected.
        """
        nan_losses = []
        inf_losses = []

        for name, value in losses.items():
            if torch.isnan(value).any():
                nan_losses.append(name)
            if torch.isinf(value).any():
                inf_losses.append(name)

        if nan_losses or inf_losses:
            msg_parts = []
            if nan_losses:
                msg_parts.append(f"NaN in losses: {nan_losses}")
            if inf_losses:
                msg_parts.append(f"Inf in losses: {inf_losses}")
            raise TrainingError(
                f"Numerical instability at step {step}: {', '.join(msg_parts)}. "
                "Consider reducing learning rate, enabling gradient clipping, "
                "or checking input data for anomalies."
            )

    def _check_gradients(self, step: int) -> None:
        """
        Check for NaN or Inf values in gradients.

        Args:
            step: Current training step (for error message).

        Raises:
            TrainingError: If NaN or Inf gradients are detected.
        """
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    raise TrainingError(
                        f"NaN gradient detected in parameter '{name}' at step {step}. "
                        "Consider reducing learning rate or enabling gradient clipping."
                    )
                if torch.isinf(param.grad).any():
                    raise TrainingError(
                        f"Inf gradient detected in parameter '{name}' at step {step}. "
                        "Consider reducing learning rate or enabling gradient clipping."
                    )

    def _train_step(self, data: BatchProvider | Any) -> dict[str, float]:
        """
        Execute a single training step with gradient accumulation support.

        When gradient_accumulation_steps > 1, gradients are accumulated across
        multiple forward/backward passes before updating weights. This enables
        training with larger effective batch sizes while staying within memory limits.

        The effective batch size is: batch_size * gradient_accumulation_steps
        """
        self.model.train()
        accum_steps = self.config.gradient_accumulation_steps
        is_accumulating = self._accumulation_step < accum_steps - 1

        # Sample batch
        batch = self._next_batch(data)

        # Only zero gradients at the start of accumulation cycle
        if self._accumulation_step == 0:
            self.optimizer.zero_grad()

        if self.config.mixed_precision and self.scaler is not None:
            with torch.amp.autocast(device_type=self.device.type):
                loss_out = self.model.loss(batch)  # type: ignore[attr-defined, operator]
                # Scale loss by accumulation steps for proper averaging
                loss = loss_out.loss / accum_steps

            # Check for NaN/Inf in losses before backward
            self._check_for_nan_inf(
                {"loss": loss_out.loss, **loss_out.components}, self.state.global_step
            )

            self.scaler.scale(loss).backward()  # type: ignore[union-attr]

            # Only step optimizer when accumulation is complete
            if not is_accumulating:
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    # Check gradients after unscaling
                    self._check_gradients(self.state.global_step)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
        else:
            loss_out = self.model.loss(batch)  # type: ignore[attr-defined, operator]
            # Scale loss by accumulation steps for proper averaging
            loss = loss_out.loss / accum_steps

            # Check for NaN/Inf in losses before backward
            self._check_for_nan_inf(
                {"loss": loss_out.loss, **loss_out.components}, self.state.global_step
            )

            loss.backward()

            # Only step optimizer when accumulation is complete
            if not is_accumulating:
                # Check gradients after backward
                self._check_gradients(self.state.global_step)

                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip,
                    )

                self.optimizer.step()

        # Update accumulation counter
        self._accumulation_step = (self._accumulation_step + 1) % accum_steps

        # Update scheduler only when accumulation is complete
        if self.scheduler is not None and not is_accumulating:
            self.scheduler.step()

        # Update state (use unscaled loss for logging)
        loss_value = loss_out.loss.item()  # Use original loss, not scaled
        self.state.total_loss += loss_value
        self.state.num_batches += 1
        self.state.metrics = dict(loss_out.metrics)
        if not self.state.metrics:
            self.state.metrics = {k: v.item() for k, v in loss_out.components.items()}
        self.state.metrics["loss"] = loss_value

        return self.state.metrics

    def evaluate(
        self,
        data: BatchProvider | Any,
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
        if num_batches <= 0:
            raise TrainingError(f"num_batches must be positive, got {num_batches}")

        self.model.eval()

        total_metrics: dict[str, float] = {}

        with torch.no_grad():
            for _ in range(num_batches):
                batch = self._next_batch(data)
                loss_out = self.model.loss(batch)  # type: ignore[attr-defined, operator]

                total_metrics["loss"] = total_metrics.get("loss", 0.0) + loss_out.loss.item()
                for k, v in loss_out.components.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v.item()

        # Average
        return {k: v / num_batches for k, v in total_metrics.items()}

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint atomically with validation.

        Uses atomic write pattern: write to temp file, validate, then rename.
        This prevents corrupted checkpoints if disk fills or process is killed.
        """
        import tempfile

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

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Atomic save: write to temp file, validate, then rename
        temp_fd, temp_path = tempfile.mkstemp(
            suffix=".pt.tmp", dir=path_obj.parent, prefix=path_obj.stem
        )
        try:
            os.close(temp_fd)
            torch.save(checkpoint, temp_path)

            # Validate checkpoint by attempting to load it
            try:
                test_load = torch.load(temp_path, map_location="cpu", weights_only=True)
                # Verify essential keys exist
                required_keys = ["model_state_dict", "optimizer_state_dict", "global_step"]
                for key in required_keys:
                    if key not in test_load:
                        raise CheckpointError(f"Checkpoint validation failed: missing key '{key}'")
                del test_load
            except Exception as e:
                raise CheckpointError(f"Checkpoint validation failed: {e}") from e

            # Atomic rename (on POSIX systems)
            os.replace(temp_path, path)
            logger.info(f"Saved checkpoint to {path}")

        except Exception:
            # Clean up temp file on failure
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

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
            checkpoint = torch.load(  # nosec B614
                path, map_location=self.device, weights_only=False
            )
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
    data: BatchProvider | Any,
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
        data: BatchProvider or iterable yielding Batch/dict.
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
        >>> from worldflux import create_world_model
        >>> from worldflux.training import train, ReplayBuffer
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
