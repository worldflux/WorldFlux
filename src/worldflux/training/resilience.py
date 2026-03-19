# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Error recovery framework for training resilience.

Provides circuit breaker patterns, NaN recovery, mixed precision fallback,
and batch loading retry with exponential backoff. All recovery actions fire
events on the EventBus when available.

Example::

    from worldflux.training.resilience import (
        TrainingCircuitBreaker,
        NaNRecoveryPolicy,
        MixedPrecisionFallback,
        BatchLoadingRetry,
    )

    breaker = TrainingCircuitBreaker(failure_threshold=5)
    action = breaker.record_failure(RuntimeError("NaN detected"))
    # action is RecoveryAction.RETRY, SKIP, or HALT
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class RecoveryAction(enum.Enum):
    """Action to take after a failure."""

    RETRY = "retry"
    SKIP = "skip"
    HALT = "halt"


@dataclass
class CircuitBreakerState:
    """Internal state for the circuit breaker."""

    failure_count: int = 0
    last_failure_time: float = 0.0
    consecutive_successes: int = 0
    is_open: bool = False
    total_failures: int = 0
    total_successes: int = 0


class TrainingCircuitBreaker:
    """Circuit breaker pattern for training failure management.

    Tracks consecutive failures and determines whether to retry, skip, or
    halt training. After ``failure_threshold`` consecutive failures, the
    circuit opens and returns HALT. During the recovery timeout window,
    failures return SKIP to allow the system to stabilize.

    Args:
        failure_threshold: Number of consecutive failures before halting.
            Default is 5.
        recovery_timeout: Seconds to wait before attempting recovery after
            the circuit opens. Default is 60.0.
        event_bus: Optional EventBus for publishing recovery events.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        event_bus: Any = None,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.event_bus = event_bus
        self.state = CircuitBreakerState()

    def record_success(self) -> None:
        """Record a successful operation and reset failure count."""
        self.state.failure_count = 0
        self.state.consecutive_successes += 1
        self.state.total_successes += 1
        if self.state.is_open:
            logger.info(
                "Circuit breaker closing after %d consecutive successes",
                self.state.consecutive_successes,
            )
            self.state.is_open = False

    def record_failure(self, error: Exception) -> RecoveryAction:
        """Record a failure and determine the recovery action.

        Args:
            error: The exception that occurred.

        Returns:
            RecoveryAction indicating what the caller should do.
        """
        self.state.failure_count += 1
        self.state.consecutive_successes = 0
        self.state.total_failures += 1
        self.state.last_failure_time = time.time()

        logger.warning(
            "Circuit breaker: failure %d/%d: %s",
            self.state.failure_count,
            self.failure_threshold,
            error,
        )

        self._publish_event(
            "recovery.circuit_breaker",
            failure_count=self.state.failure_count,
            threshold=self.failure_threshold,
            error=str(error),
        )

        if self.state.failure_count >= self.failure_threshold:
            self.state.is_open = True
            logger.error(
                "Circuit breaker OPEN after %d consecutive failures. Halting.",
                self.state.failure_count,
            )
            return RecoveryAction.HALT

        if self.state.is_open:
            elapsed = time.time() - self.state.last_failure_time
            if elapsed < self.recovery_timeout:
                return RecoveryAction.SKIP

        return RecoveryAction.RETRY

    def reset(self) -> None:
        """Reset the circuit breaker to its initial state."""
        self.state = CircuitBreakerState()

    def _publish_event(self, event_type: str, **data: Any) -> None:
        if self.event_bus is None:
            return
        from worldflux.core.events import Event

        self.event_bus.publish(Event(event_type=event_type, data=data, source="circuit_breaker"))


@dataclass
class NaNRecoveryResult:
    """Result of NaN recovery attempt."""

    recovered: bool
    action_taken: str
    new_learning_rate: float | None = None
    checkpoint_path: str | None = None
    consecutive_nan_count: int = 0


class NaNRecoveryPolicy:
    """Policy for recovering from NaN values during training.

    When NaN is detected:
    1. Roll back to the most recent checkpoint
    2. Decay learning rate by 50%
    3. Restart training from the checkpoint

    After ``max_consecutive_nans`` consecutive NaN detections, halt
    training and produce a diagnostic report.

    Args:
        max_consecutive_nans: Maximum consecutive NaN detections before
            halting. Default is 5.
        lr_decay_factor: Factor to multiply learning rate by on each
            NaN recovery. Default is 0.5.
        event_bus: Optional EventBus for publishing recovery events.
    """

    def __init__(
        self,
        max_consecutive_nans: int = 5,
        lr_decay_factor: float = 0.5,
        event_bus: Any = None,
    ) -> None:
        self.max_consecutive_nans = max_consecutive_nans
        self.lr_decay_factor = lr_decay_factor
        self.event_bus = event_bus
        self._consecutive_nan_count = 0

    def on_nan_detected(
        self,
        trainer: Any,
        checkpoint_path: str | None = None,
    ) -> NaNRecoveryResult:
        """Handle NaN detection during training.

        Args:
            trainer: The Trainer instance (used to adjust LR and load checkpoint).
            checkpoint_path: Path to checkpoint to roll back to.

        Returns:
            NaNRecoveryResult describing what action was taken.
        """
        self._consecutive_nan_count += 1

        logger.warning(
            "NaN detected (consecutive: %d/%d)",
            self._consecutive_nan_count,
            self.max_consecutive_nans,
        )

        self._publish_event(
            "nan.detected",
            consecutive_count=self._consecutive_nan_count,
            max_count=self.max_consecutive_nans,
        )

        if self._consecutive_nan_count >= self.max_consecutive_nans:
            logger.error(
                "NaN recovery exhausted after %d consecutive NaN detections. "
                "Halting training. Diagnostic: check input data, learning rate, "
                "and gradient clipping settings.",
                self._consecutive_nan_count,
            )
            return NaNRecoveryResult(
                recovered=False,
                action_taken="halt",
                consecutive_nan_count=self._consecutive_nan_count,
            )

        # Attempt checkpoint rollback
        if checkpoint_path is not None:
            try:
                load_fn = getattr(trainer, "load_checkpoint", None)
                if callable(load_fn):
                    load_fn(checkpoint_path)
                    logger.info("Rolled back to checkpoint: %s", checkpoint_path)
                    self._publish_event(
                        "recovery.nan_rollback",
                        checkpoint_path=checkpoint_path,
                    )
            except Exception as e:
                logger.warning("Checkpoint rollback failed: %s", e)

        # Decay learning rate
        new_lr = self._decay_learning_rate(trainer)
        if new_lr is not None:
            self._publish_event("recovery.lr_decay", new_lr=new_lr)

        return NaNRecoveryResult(
            recovered=True,
            action_taken="rollback_and_lr_decay",
            new_learning_rate=new_lr,
            checkpoint_path=checkpoint_path,
            consecutive_nan_count=self._consecutive_nan_count,
        )

    def record_success(self) -> None:
        """Reset the consecutive NaN counter after a successful step."""
        self._consecutive_nan_count = 0

    def _decay_learning_rate(self, trainer: Any) -> float | None:
        """Decay the optimizer learning rate by lr_decay_factor."""
        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is None:
            return None

        new_lr: float | None = None
        for param_group in optimizer.param_groups:
            old_lr = param_group["lr"]
            param_group["lr"] = old_lr * self.lr_decay_factor
            new_lr = param_group["lr"]

        if new_lr is not None:
            logger.info(
                "Learning rate decayed by %.0f%%: new lr=%.2e",
                (1 - self.lr_decay_factor) * 100,
                new_lr,
            )
        return new_lr

    def _publish_event(self, event_type: str, **data: Any) -> None:
        if self.event_bus is None:
            return
        from worldflux.core.events import Event

        self.event_bus.publish(Event(event_type=event_type, data=data, source="nan_recovery"))


class MixedPrecisionFallback:
    """Fallback from mixed precision (AMP) to full float32 on errors.

    When AMP-related errors occur (e.g. GradScaler scale reaches zero),
    this policy disables mixed precision and continues training in float32.

    Args:
        event_bus: Optional EventBus for publishing recovery events.
    """

    def __init__(self, event_bus: Any = None) -> None:
        self.event_bus = event_bus
        self._fallen_back = False

    @property
    def has_fallen_back(self) -> bool:
        """Whether fallback to float32 has occurred."""
        return self._fallen_back

    def on_amp_error(self, trainer: Any) -> bool:
        """Handle an AMP error by falling back to float32.

        Args:
            trainer: The Trainer instance to modify.

        Returns:
            True if fallback was applied, False if already fallen back.
        """
        if self._fallen_back:
            return False

        logger.warning("Mixed precision error detected. Falling back to float32.")

        # Disable mixed precision on the trainer
        config = getattr(trainer, "config", None)
        if config is not None:
            config.mixed_precision = False

        # Remove the GradScaler
        if hasattr(trainer, "scaler"):
            trainer.scaler = None

        self._fallen_back = True

        self._publish_event("recovery.amp_fallback")
        return True

    def _publish_event(self, event_type: str, **data: Any) -> None:
        if self.event_bus is None:
            return
        from worldflux.core.events import Event

        self.event_bus.publish(Event(event_type=event_type, data=data, source="amp_fallback"))


@dataclass
class BatchRetryResult:
    """Result of a batch loading retry."""

    success: bool
    batch: Any = None
    attempts: int = 0
    skipped: bool = False


class BatchLoadingRetry:
    """Retry logic for batch loading with exponential backoff.

    On data loading errors, retries up to ``max_retries`` times with
    exponential backoff. After exhausting retries, skips the batch and
    logs a warning.

    Args:
        max_retries: Maximum number of retry attempts. Default is 3.
        base_delay: Base delay in seconds for exponential backoff.
            Default is 0.1.
        event_bus: Optional EventBus for publishing recovery events.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.1,
        event_bus: Any = None,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.event_bus = event_bus

    def retry_load(
        self,
        load_fn: Any,
        *args: Any,
        **kwargs: Any,
    ) -> BatchRetryResult:
        """Attempt to load a batch with retries and exponential backoff.

        Args:
            load_fn: Callable that loads and returns a batch.
            *args: Positional arguments for load_fn.
            **kwargs: Keyword arguments for load_fn.

        Returns:
            BatchRetryResult with the loaded batch or skip indication.
        """
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                batch = load_fn(*args, **kwargs)
                return BatchRetryResult(success=True, batch=batch, attempts=attempt + 1)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self.base_delay * (2**attempt)
                    logger.warning(
                        "Batch loading failed (attempt %d/%d): %s. " "Retrying in %.2fs...",
                        attempt + 1,
                        self.max_retries + 1,
                        e,
                        delay,
                    )
                    self._publish_event(
                        "recovery.batch_retry",
                        attempt=attempt + 1,
                        max_retries=self.max_retries,
                        error=str(e),
                        delay=delay,
                    )
                    time.sleep(delay)

        logger.warning(
            "Batch loading failed after %d attempts. Skipping batch. " "Last error: %s",
            self.max_retries + 1,
            last_error,
        )
        return BatchRetryResult(
            success=False,
            attempts=self.max_retries + 1,
            skipped=True,
        )

    def _publish_event(self, event_type: str, **data: Any) -> None:
        if self.event_bus is None:
            return
        from worldflux.core.events import Event

        self.event_bus.publish(Event(event_type=event_type, data=data, source="batch_retry"))
