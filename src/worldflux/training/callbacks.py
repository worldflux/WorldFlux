"""Callbacks for training hooks and logging."""

from __future__ import annotations

import logging
import math
import time
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from worldflux.telemetry.wasr import make_run_id, write_event

from .report import HealthSignal, LossCurveSummary, TrainingReport

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
        if log_interval <= 0:
            raise ValueError(f"log_interval must be positive, got {log_interval}")
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
                    project=self.wandb_project or "worldflux",
                    name=self.wandb_run_name,
                    config=trainer.config.to_dict(),
                )
            except ImportError:
                logger.warning("wandb not installed, disabling wandb logging")
                self.use_wandb = False

        logger.info("Training started")

    def on_train_end(self, trainer: Trainer) -> None:
        elapsed = time.time() - self._start_time
        logger.info(f"Training completed in {elapsed:.1f}s ({trainer.state.global_step} steps)")

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
            f"Step {step}/{total_steps} ({progress:.1f}%) | {loss_str} | {speed:.1f} steps/s"
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
        if save_interval <= 0:
            raise ValueError(f"save_interval must be positive, got {save_interval}")
        if max_checkpoints is not None and max_checkpoints <= 0:
            raise ValueError(f"max_checkpoints must be positive when set, got {max_checkpoints}")
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
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be non-negative, got {min_delta}")
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
            logger.info(f"Early stopping: no improvement for {self.patience} steps")
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


class HeartbeatCallback(Callback):
    """Periodic anonymous telemetry heartbeat for successful training progress."""

    def __init__(
        self,
        interval_steps: int = 100,
        scenario: str = "trainer",
        metrics_path: str | Path | None = None,
        run_id: str | None = None,
    ) -> None:
        if interval_steps <= 0:
            raise ValueError(f"interval_steps must be positive, got {interval_steps}")
        self.interval_steps = interval_steps
        self.scenario = scenario
        self.metrics_path = Path(metrics_path) if metrics_path is not None else None
        self.run_id = run_id or make_run_id()
        self._start_time: float | None = None

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return number

    def on_train_begin(self, trainer: Trainer) -> None:
        self._start_time = time.time()
        if trainer.state.train_start_time is not None:
            self._start_time = trainer.state.train_start_time

    def _current_duration(self) -> float:
        if self._start_time is None:
            return 0.0
        return max(0.0, time.time() - self._start_time)

    def on_step_end(self, trainer: Trainer) -> None:
        step = int(trainer.state.global_step)
        if step <= 0 or step % self.interval_steps != 0:
            return

        profile_fn = getattr(trainer, "runtime_profile", None)
        throughput: float | None = None
        if callable(profile_fn):
            profile = profile_fn()
            throughput = self._safe_float(profile.get("throughput_steps_per_sec"))

        metrics = trainer.state.metrics
        flops = self._safe_float(metrics.get("flops_estimate") or metrics.get("flops"))
        watts = self._safe_float(metrics.get("watts_estimate") or metrics.get("power_watts"))
        flops_per_watt: float | None = None
        if flops is not None and watts is not None and watts > 0:
            flops_per_watt = flops / watts

        write_event(
            event="heartbeat",
            scenario=self.scenario,
            success=True,
            duration_sec=self._current_duration(),
            ttfi_sec=float(trainer.state.ttfi_sec or 0.0),
            artifacts={},
            error="",
            run_id=self.run_id,
            path=self.metrics_path,
            epoch=int(trainer.state.epoch),
            step=step,
            throughput_steps_per_sec=throughput,
            flops_estimate=flops,
            watts_estimate=watts,
            flops_per_watt=flops_per_watt,
        )


class DiagnosisCallback(Callback):
    """Detect common training pathologies and emit remediation suggestions."""

    def __init__(
        self,
        check_interval: int = 100,
        gradient_min_norm: float = 1e-8,
        latent_std_min: float = 1e-6,
        scenario: str = "trainer",
        metrics_path: str | Path | None = None,
        run_id: str | None = None,
    ) -> None:
        if check_interval <= 0:
            raise ValueError(f"check_interval must be positive, got {check_interval}")
        if gradient_min_norm < 0:
            raise ValueError(f"gradient_min_norm must be non-negative, got {gradient_min_norm}")
        if latent_std_min < 0:
            raise ValueError(f"latent_std_min must be non-negative, got {latent_std_min}")

        self.check_interval = check_interval
        self.gradient_min_norm = gradient_min_norm
        self.latent_std_min = latent_std_min
        self.scenario = scenario
        self.metrics_path = Path(metrics_path) if metrics_path is not None else None
        self.run_id = run_id or make_run_id()
        self._start_time: float | None = None
        self.last_suggestions: list[str] = []

    def on_train_begin(self, trainer: Trainer) -> None:
        self._start_time = time.time()
        if trainer.state.train_start_time is not None:
            self._start_time = trainer.state.train_start_time

    def _current_duration(self) -> float:
        if self._start_time is None:
            return 0.0
        return max(0.0, time.time() - self._start_time)

    def _detect_gradient_issues(self, trainer: Trainer) -> list[str]:
        suggestions: list[str] = []
        grad_norms: list[float] = []
        has_nan_or_inf = False

        for param in trainer.model.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            if torch.isnan(grad).any() or torch.isinf(grad).any():
                has_nan_or_inf = True
                break
            grad_norms.append(float(torch.norm(grad).item()))

        if has_nan_or_inf:
            suggestions.append(
                "Detected NaN/Inf gradients. Try lower learning rate or stronger gradient clipping."
            )
            return suggestions

        if grad_norms and max(grad_norms) < self.gradient_min_norm:
            suggestions.append(
                "Detected vanishing gradients. Consider normalization changes or shorter horizons."
            )

        return suggestions

    def _detect_metric_issues(self, trainer: Trainer) -> list[str]:
        suggestions: list[str] = []
        metrics = trainer.state.metrics

        loss = metrics.get("loss")
        if loss is not None:
            loss_value = float(loss)
            if not math.isfinite(loss_value):
                suggestions.append(
                    "Loss is non-finite. Validate inputs and reduce optimizer aggressiveness."
                )

        latent_keys = ("latent_std", "z_std", "latent_variance", "z_var")
        for key in latent_keys:
            value = metrics.get(key)
            if value is None:
                continue
            value_f = float(value)
            if math.isfinite(value_f) and value_f < self.latent_std_min:
                suggestions.append(
                    "Latent collapse indicator detected. Increase latent regularization or capacity."
                )
                break

        return suggestions

    def on_step_end(self, trainer: Trainer) -> None:
        step = int(trainer.state.global_step)
        if step <= 0 or step % self.check_interval != 0:
            return

        suggestions = [
            *self._detect_gradient_issues(trainer),
            *self._detect_metric_issues(trainer),
        ]
        self.last_suggestions = suggestions
        if not suggestions:
            return

        joined = " | ".join(suggestions)
        logger.warning("Training diagnostics at step %s: %s", step, joined)
        write_event(
            event="diagnostic",
            scenario=self.scenario,
            success=False,
            duration_sec=self._current_duration(),
            ttfi_sec=float(trainer.state.ttfi_sec or 0.0),
            artifacts={},
            error=joined,
            run_id=self.run_id,
            path=self.metrics_path,
            epoch=int(trainer.state.epoch),
            step=step,
            suggestions=suggestions,
        )


class TrainingReportCallback(Callback):
    """Generates structured training report on completion."""

    def __init__(
        self,
        output_dir: str | Path = "./outputs",
        scenario: str = "trainer",
        metrics_path: Path | str | None = None,
        run_id: str | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.scenario = scenario
        self.metrics_path = Path(metrics_path) if metrics_path else None
        self.run_id = run_id or make_run_id()

        # Data collection
        self._loss_history: list[float] = []
        self._gradient_nan_count: int = 0
        self._non_finite_count: int = 0
        self._throughput_history: list[float] = []
        self._start_time: float = 0.0

    def on_train_begin(self, trainer: Trainer) -> None:
        self._start_time = time.time()

    def on_step_end(self, trainer: Trainer) -> None:
        # Collect loss
        loss = trainer.state.metrics.get("loss")
        if loss is not None:
            if math.isfinite(loss):
                self._loss_history.append(loss)
            else:
                self._non_finite_count += 1

        # Collect throughput periodically
        if trainer.state.global_step % 100 == 0:
            profile_fn = getattr(trainer, "runtime_profile", None)
            if callable(profile_fn):
                profile = profile_fn()
                throughput = profile.get("throughput_steps_per_sec")
                if throughput is not None and throughput > 0:
                    self._throughput_history.append(throughput)

    def on_train_end(self, trainer: Trainer) -> None:
        report = self._build_report(trainer)
        report_path = self.output_dir / "training_report.json"
        report.save(report_path)
        logger.info("Training report saved to %s", report_path)

        # Write WASR event
        if self.metrics_path:
            write_event(
                event="run.summary",
                scenario=self.scenario,
                success=True,
                duration_sec=report.wall_time_sec,
                ttfi_sec=report.ttfi_sec,
                path=self.metrics_path,
                run_id=self.run_id,
                step=report.total_steps,
            )

    def _build_report(self, trainer: Trainer) -> TrainingReport:
        wall_time = time.time() - self._start_time
        total_steps = trainer.state.global_step
        ttfi = float(trainer.state.ttfi_sec or 0.0)

        # Throughput
        if self._throughput_history:
            throughput = sum(self._throughput_history) / len(self._throughput_history)
        elif wall_time > 0:
            throughput = total_steps / wall_time
        else:
            throughput = 0.0

        # Loss curve summary
        loss_summary = self._compute_loss_summary()

        # Health signals
        signals = self._compute_health_signals(loss_summary)

        # Health score (weighted average)
        health_score = self._compute_health_score(signals)

        # Recommendations
        recommendations = self._generate_recommendations(signals, loss_summary)

        model_id = type(trainer.model).__name__

        return TrainingReport(
            model_id=model_id,
            total_steps=total_steps,
            wall_time_sec=wall_time,
            final_loss=loss_summary.final_loss,
            best_loss=loss_summary.best_loss,
            ttfi_sec=ttfi,
            throughput_steps_per_sec=throughput,
            health_score=health_score,
            health_signals=signals,
            loss_curve_summary=loss_summary,
            recommendations=recommendations,
        )

    def _compute_loss_summary(self) -> LossCurveSummary:
        if not self._loss_history:
            return LossCurveSummary(
                initial_loss=0.0,
                final_loss=0.0,
                best_loss=0.0,
                best_step=0,
                convergence_slope=0.0,
                plateau_detected=False,
            )

        initial = self._loss_history[0]
        final = self._loss_history[-1]
        best = min(self._loss_history)
        best_step = self._loss_history.index(best)

        # Convergence slope: linear regression over loss values
        n = len(self._loss_history)
        if n > 1:
            x_mean = (n - 1) / 2.0
            y_mean = sum(self._loss_history) / n
            numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(self._loss_history))
            denominator = sum((i - x_mean) ** 2 for i in range(n))
            slope = numerator / denominator if denominator > 0 else 0.0
        else:
            slope = 0.0

        # Plateau detection: compare last window vs previous window
        plateau = False
        window = min(1000, n // 4) if n >= 8 else 0
        if window > 0:
            recent = self._loss_history[-window:]
            previous = self._loss_history[-2 * window : -window]
            if previous:
                recent_avg = sum(recent) / len(recent)
                prev_avg = sum(previous) / len(previous)
                if prev_avg > 0:
                    relative_change = abs(recent_avg - prev_avg) / abs(prev_avg)
                    plateau = relative_change < 0.01

        return LossCurveSummary(
            initial_loss=initial,
            final_loss=final,
            best_loss=best,
            best_step=best_step,
            convergence_slope=slope,
            plateau_detected=plateau,
        )

    def _compute_health_signals(self, loss_summary: LossCurveSummary) -> dict[str, HealthSignal]:
        signals: dict[str, HealthSignal] = {}

        # Loss convergence
        if not self._loss_history:
            signals["loss_convergence"] = HealthSignal(
                name="loss_convergence",
                status="critical",
                value=0.0,
                message="No loss values recorded.",
            )
        elif loss_summary.convergence_slope >= 0:
            signals["loss_convergence"] = HealthSignal(
                name="loss_convergence",
                status="warning",
                value=loss_summary.convergence_slope,
                message="Loss is not decreasing.",
            )
        else:
            signals["loss_convergence"] = HealthSignal(
                name="loss_convergence",
                status="healthy",
                value=loss_summary.convergence_slope,
                message="Loss is decreasing.",
            )

        # Numerical stability
        total = len(self._loss_history) + self._non_finite_count
        if total > 0:
            non_finite_ratio = self._non_finite_count / total
        else:
            non_finite_ratio = 0.0

        if non_finite_ratio > 0.1:
            status = "critical"
            msg = f"{self._non_finite_count} non-finite loss values detected."
        elif non_finite_ratio > 0.0:
            status = "warning"
            msg = f"{self._non_finite_count} non-finite loss values detected."
        else:
            status = "healthy"
            msg = "All loss values are finite."
        signals["numerical_stability"] = HealthSignal(
            name="numerical_stability",
            status=status,
            value=1.0 - non_finite_ratio,
            message=msg,
        )

        # Throughput stability
        if len(self._throughput_history) >= 2:
            mean_tp = sum(self._throughput_history) / len(self._throughput_history)
            if mean_tp > 0:
                variance = sum((t - mean_tp) ** 2 for t in self._throughput_history) / len(
                    self._throughput_history
                )
                cv = (variance**0.5) / mean_tp
            else:
                cv = 0.0
            if cv > 0.5:
                tp_status = "warning"
                tp_msg = f"Throughput coefficient of variation is high ({cv:.2f})."
            else:
                tp_status = "healthy"
                tp_msg = f"Throughput is stable (CV={cv:.2f})."
            signals["throughput_stability"] = HealthSignal(
                name="throughput_stability",
                status=tp_status,
                value=1.0 - min(cv, 1.0),
                message=tp_msg,
            )
        else:
            signals["throughput_stability"] = HealthSignal(
                name="throughput_stability",
                status="healthy",
                value=1.0,
                message="Insufficient data for throughput analysis.",
            )

        return signals

    @staticmethod
    def _compute_health_score(signals: dict[str, HealthSignal]) -> float:
        if not signals:
            return 0.0
        weights = {
            "loss_convergence": 0.4,
            "numerical_stability": 0.3,
            "throughput_stability": 0.15,
            "gradient_health": 0.15,
        }
        total_weight = 0.0
        weighted_sum = 0.0
        for name, signal in signals.items():
            w = weights.get(name, 0.1)
            score = {"healthy": 1.0, "warning": 0.5, "critical": 0.0}.get(signal.status, 0.0)
            weighted_sum += w * score
            total_weight += w
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    @staticmethod
    def _generate_recommendations(
        signals: dict[str, HealthSignal],
        loss_summary: LossCurveSummary,
    ) -> list[str]:
        recs: list[str] = []
        for signal in signals.values():
            if signal.status == "critical":
                recs.append(f"[CRITICAL] {signal.name}: {signal.message}")
            elif signal.status == "warning":
                recs.append(f"[WARNING] {signal.name}: {signal.message}")
        if loss_summary.plateau_detected:
            recs.append(
                "Loss plateau detected. Consider adjusting learning rate or model capacity."
            )
        return recs


class EvalCallback(Callback):
    """Run model evaluation suite periodically during training.

    Args:
        eval_interval: Steps between evaluation runs.
        suite: Evaluation suite name (``quick``, ``standard``, ``comprehensive``).
        scenario: Telemetry scenario tag.
        metrics_path: Optional path for WASR telemetry events.
        run_id: Optional run identifier for telemetry.
    """

    def __init__(
        self,
        eval_interval: int = 5000,
        suite: str = "quick",
        scenario: str = "trainer",
        metrics_path: Path | str | None = None,
        run_id: str | None = None,
    ) -> None:
        if eval_interval <= 0:
            raise ValueError(f"eval_interval must be positive, got {eval_interval}")
        self.eval_interval = eval_interval
        self.suite = suite
        self.scenario = scenario
        self.metrics_path = Path(metrics_path) if metrics_path is not None else None
        self.run_id = run_id or make_run_id()
        self._start_time: float | None = None

    def on_train_begin(self, trainer: Trainer) -> None:
        self._start_time = time.time()
        if trainer.state.train_start_time is not None:
            self._start_time = trainer.state.train_start_time

    def _current_duration(self) -> float:
        if self._start_time is None:
            return 0.0
        return max(0.0, time.time() - self._start_time)

    def on_step_end(self, trainer: Trainer) -> None:
        step = int(trainer.state.global_step)
        if step <= 0 or step % self.eval_interval != 0:
            return

        from worldflux.evals.suite import run_eval_suite

        model_id = type(trainer.model).__name__
        try:
            report = run_eval_suite(
                trainer.model,  # type: ignore[arg-type]
                suite=self.suite,
                model_id=model_id,
            )
        except Exception:
            logger.warning("Eval suite failed at step %s", step, exc_info=True)
            return

        passed_count = sum(1 for r in report.results if r.passed is True)
        total_checked = sum(1 for r in report.results if r.passed is not None)
        logger.info(
            "Eval [%s] at step %s: %d/%d passed (%.2fs)",
            self.suite,
            step,
            passed_count,
            total_checked,
            report.wall_time_sec,
        )

        write_event(
            event="eval.quick",
            scenario=self.scenario,
            success=report.all_passed is True or report.all_passed is None,
            duration_sec=self._current_duration(),
            ttfi_sec=float(trainer.state.ttfi_sec or 0.0),
            artifacts={},
            error="",
            run_id=self.run_id,
            path=self.metrics_path,
            epoch=int(trainer.state.epoch),
            step=step,
        )
