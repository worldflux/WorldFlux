"""Callbacks for training hooks and logging."""

from __future__ import annotations

import logging
import time
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .trainer import Trainer

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base class for training callbacks."""

    def on_train_begin(self, trainer: Trainer) -> None:
        """Called at the start of training."""
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer: Trainer) -> None:
        """Called at the start of each epoch."""
        pass

    def on_epoch_end(self, trainer: Trainer) -> None:
        """Called at the end of each epoch."""
        pass

    def on_step_begin(self, trainer: Trainer) -> None:
        """Called before each training step."""
        pass

    def on_step_end(self, trainer: Trainer) -> None:
        """Called after each training step."""
        pass


class CallbackList:
    """Container for managing multiple callbacks."""

    def __init__(self, callbacks: list[Callback] | None = None):
        self.callbacks = callbacks or []

    def append(self, callback: Callback) -> None:
        """Add a callback to the list."""
        self.callbacks.append(callback)

    def on_train_begin(self, trainer: Trainer) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(trainer)

    def on_train_end(self, trainer: Trainer) -> None:
        for cb in self.callbacks:
            cb.on_train_end(trainer)

    def on_epoch_begin(self, trainer: Trainer) -> None:
        for cb in self.callbacks:
            cb.on_epoch_begin(trainer)

    def on_epoch_end(self, trainer: Trainer) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(trainer)

    def on_step_begin(self, trainer: Trainer) -> None:
        for cb in self.callbacks:
            cb.on_step_begin(trainer)

    def on_step_end(self, trainer: Trainer) -> None:
        for cb in self.callbacks:
            cb.on_step_end(trainer)


class LoggingCallback(Callback):
    """
    Callback for logging training metrics.

    Logs to console and optionally to wandb.

    Args:
        log_interval: Steps between log outputs.
        use_wandb: Whether to log to wandb.
        wandb_project: wandb project name.
        wandb_run_name: wandb run name.

    Example:
        >>> callback = LoggingCallback(log_interval=100, use_wandb=True)
        >>> trainer = Trainer(model, config, callbacks=[callback])
    """

    def __init__(
        self,
        log_interval: int = 100,
        use_wandb: bool = False,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
    ):
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        self._start_time: float = 0.0
        self._last_log_time: float = 0.0
        self._last_log_step: int = 0
        self._wandb_run: Any = None

    def on_train_begin(self, trainer: Trainer) -> None:
        self._start_time = time.time()
        self._last_log_time = self._start_time
        self._last_log_step = trainer.state.global_step

        if self.use_wandb:
            try:
                import wandb

                self._wandb_run = wandb.init(
                    project=self.wandb_project or "worldmodels",
                    name=self.wandb_run_name,
                    config=trainer.config.to_dict(),
                )
            except ImportError:
                logger.warning("wandb not installed, disabling wandb logging")
                self.use_wandb = False

        logger.info("Training started")

    def on_train_end(self, trainer: Trainer) -> None:
        elapsed = time.time() - self._start_time
        logger.info(
            f"Training completed in {elapsed:.1f}s "
            f"({trainer.state.global_step} steps)"
        )

        if self._wandb_run is not None:
            self._wandb_run.finish()

    def on_step_end(self, trainer: Trainer) -> None:
        step = trainer.state.global_step

        if step % self.log_interval == 0:
            self._log_metrics(trainer)

    def _log_metrics(self, trainer: Trainer) -> None:
        step = trainer.state.global_step
        metrics = trainer.state.metrics

        # Calculate speed
        now = time.time()
        elapsed = now - self._last_log_time
        steps_delta = step - self._last_log_step
        speed = steps_delta / elapsed if elapsed > 0 else 0.0

        self._last_log_time = now
        self._last_log_step = step

        # Format log message
        total_steps = trainer.config.total_steps
        progress = step / total_steps * 100

        loss_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info(
            f"Step {step}/{total_steps} ({progress:.1f}%) | "
            f"{loss_str} | "
            f"{speed:.1f} steps/s"
        )

        # Log to wandb
        if self._wandb_run is not None:
            import wandb

            wandb.log({"step": step, "speed": speed, **metrics})


class CheckpointCallback(Callback):
    """
    Callback for saving model checkpoints.

    Args:
        save_interval: Steps between checkpoint saves.
        output_dir: Directory to save checkpoints.
        save_best: Whether to save the best model (lowest loss).
        max_checkpoints: Maximum number of checkpoints to keep.

    Example:
        >>> callback = CheckpointCallback(
        ...     save_interval=10000,
        ...     output_dir="./checkpoints",
        ...     save_best=True,
        ... )
    """

    def __init__(
        self,
        save_interval: int = 10000,
        output_dir: str = "./outputs",
        save_best: bool = True,
        max_checkpoints: int | None = 5,
    ):
        self.save_interval = save_interval
        self.output_dir = Path(output_dir)
        self.save_best = save_best
        self.max_checkpoints = max_checkpoints

        self._checkpoint_paths: list[Path] = []
        self._best_loss: float = float("inf")

    def on_train_begin(self, trainer: Trainer) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, trainer: Trainer) -> None:
        step = trainer.state.global_step
        metrics = trainer.state.metrics

        # Save at interval
        if step > 0 and step % self.save_interval == 0:
            self._save_checkpoint(trainer, f"checkpoint_{step}.pt")

        # Save best model
        if self.save_best and "loss" in metrics:
            loss = metrics["loss"]
            if loss < self._best_loss:
                self._best_loss = loss
                trainer.state.best_loss = loss
                self._save_checkpoint(trainer, "checkpoint_best.pt")
                logger.info(f"New best loss: {loss:.4f}")

    def _save_checkpoint(self, trainer: Trainer, filename: str) -> None:
        path = self.output_dir / filename
        trainer.save_checkpoint(str(path))

        # Track non-best checkpoints for cleanup
        if "best" not in filename:
            self._checkpoint_paths.append(path)
            self._cleanup_old_checkpoints()

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints if max_checkpoints is set."""
        if self.max_checkpoints is None:
            return

        while len(self._checkpoint_paths) > self.max_checkpoints:
            old_path = self._checkpoint_paths.pop(0)
            if old_path.exists():
                old_path.unlink()
                logger.debug(f"Removed old checkpoint: {old_path}")


class EarlyStoppingCallback(Callback):
    """
    Callback for early stopping based on loss plateau.

    Args:
        patience: Number of steps to wait before stopping.
        min_delta: Minimum improvement to reset patience.
        monitor: Metric to monitor (default: "loss").

    Example:
        >>> callback = EarlyStoppingCallback(patience=5000, min_delta=1e-4)
    """

    def __init__(
        self,
        patience: int = 5000,
        min_delta: float = 1e-4,
        monitor: str = "loss",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor

        self._best_value: float = float("inf")
        self._steps_without_improvement: int = 0

    def on_step_end(self, trainer: Trainer) -> None:
        metrics = trainer.state.metrics

        if self.monitor not in metrics:
            return

        current = metrics[self.monitor]

        if current < self._best_value - self.min_delta:
            self._best_value = current
            self._steps_without_improvement = 0
        else:
            self._steps_without_improvement += 1

        if self._steps_without_improvement >= self.patience:
            logger.info(
                f"Early stopping: no improvement for {self.patience} steps"
            )
            trainer.state.should_stop = True


class ProgressCallback(Callback):
    """
    Callback for displaying progress bar using tqdm.

    Args:
        desc: Description for progress bar.

    Example:
        >>> callback = ProgressCallback(desc="Training")
    """

    def __init__(self, desc: str = "Training"):
        self.desc = desc
        self._pbar: Any = None

    def on_train_begin(self, trainer: Trainer) -> None:
        try:
            from tqdm import tqdm

            total_steps = trainer.config.total_steps
            initial = trainer.state.global_step
            self._pbar = tqdm(
                total=total_steps,
                initial=initial,
                desc=self.desc,
                unit="step",
            )
        except ImportError:
            logger.debug("tqdm not installed, progress bar disabled")

    def on_train_end(self, trainer: Trainer) -> None:
        if self._pbar is not None:
            self._pbar.close()

    def on_step_end(self, trainer: Trainer) -> None:
        if self._pbar is not None:
            self._pbar.update(1)

            # Update postfix with metrics
            metrics = trainer.state.metrics
            if metrics:
                postfix = {k: f"{v:.4f}" for k, v in metrics.items()}
                self._pbar.set_postfix(postfix)
