"""Event bus system for extensible event-driven architecture.

Provides a publish/subscribe event system that supplements the existing
Callback hooks with fine-grained, pattern-matchable events. The EventBus
is designed to have near-zero overhead when no subscribers are registered.

Example::

    from worldflux.core.events import EventBus, Event

    bus = EventBus()

    def on_loss(event: Event) -> None:
        print(f"Loss computed: {event.data}")

    bus.subscribe("loss.computed", on_loss)
    bus.publish(Event(event_type="loss.computed", data={"total": 0.42}))
"""

from __future__ import annotations

import fnmatch
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


class EventTypes:
    """Constants for standard event types.

    Model Lifecycle events:
        model.created, model.validated, component.swapped

    Training events:
        train.begin, train.end, epoch.begin, epoch.end,
        step.begin, step.end

    Compute events:
        loss.computed, backward.complete, optimizer.stepped,
        gradients.clipped

    Error events:
        nan.detected, training.error, checkpoint.failed

    Data events:
        batch.loaded, buffer.overflow

    Recovery events:
        recovery.nan_rollback, recovery.lr_decay, recovery.amp_fallback,
        recovery.batch_retry, recovery.circuit_breaker
    """

    # Model Lifecycle
    MODEL_CREATED = "model.created"
    MODEL_VALIDATED = "model.validated"
    COMPONENT_SWAPPED = "component.swapped"

    # Training
    TRAIN_BEGIN = "train.begin"
    TRAIN_END = "train.end"
    EPOCH_BEGIN = "epoch.begin"
    EPOCH_END = "epoch.end"
    STEP_BEGIN = "step.begin"
    STEP_END = "step.end"

    # Compute
    LOSS_COMPUTED = "loss.computed"
    BACKWARD_COMPLETE = "backward.complete"
    OPTIMIZER_STEPPED = "optimizer.stepped"
    GRADIENTS_CLIPPED = "gradients.clipped"

    # Error
    NAN_DETECTED = "nan.detected"
    TRAINING_ERROR = "training.error"
    CHECKPOINT_FAILED = "checkpoint.failed"

    # Data
    BATCH_LOADED = "batch.loaded"
    BUFFER_OVERFLOW = "buffer.overflow"

    # Recovery
    RECOVERY_NAN_ROLLBACK = "recovery.nan_rollback"
    RECOVERY_LR_DECAY = "recovery.lr_decay"
    RECOVERY_AMP_FALLBACK = "recovery.amp_fallback"
    RECOVERY_BATCH_RETRY = "recovery.batch_retry"
    RECOVERY_CIRCUIT_BREAKER = "recovery.circuit_breaker"


@dataclass(frozen=True)
class Event:
    """An event published through the EventBus.

    Attributes:
        event_type: Dot-separated event type string (e.g. "train.begin").
        data: Arbitrary payload for the event.
        source: Optional string identifying the event source.
        timestamp: Time of event creation (seconds since epoch).
    """

    event_type: str
    data: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class Subscription:
    """Handle for an active event subscription.

    Attributes:
        pattern: The event type pattern this subscription matches.
        handler: The callable invoked when a matching event is published.
        priority: Lower values are invoked first. Default is 50.
        subscription_id: Unique identifier for this subscription.
    """

    pattern: str
    handler: Callable[[Event], None]
    priority: int = 50
    subscription_id: int = 0


class EventBus:
    """Publish/subscribe event bus with pattern matching and priority ordering.

    Features:
        - Exact match subscriptions: ``bus.subscribe("train.begin", handler)``
        - Pattern subscriptions: ``bus.subscribe("train.*", handler)``
        - Priority ordering: lower priority values are called first
        - Near-zero overhead when disabled or no subscribers are registered
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._exact_handlers: dict[str, list[Subscription]] = defaultdict(list)
        self._pattern_handlers: list[Subscription] = []
        self._next_id = 0

    @property
    def enabled(self) -> bool:
        """Whether the event bus is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None],
        *,
        priority: int = 50,
    ) -> Subscription:
        """Subscribe a handler to events matching event_type.

        Args:
            event_type: Event type string or glob pattern (e.g. "train.*").
            handler: Callable accepting an Event argument.
            priority: Handler priority. Lower values are invoked first.
                Default is 50.

        Returns:
            A Subscription handle that can be passed to unsubscribe().
        """
        sub = Subscription(
            pattern=event_type,
            handler=handler,
            priority=priority,
            subscription_id=self._next_id,
        )
        self._next_id += 1

        if "*" in event_type or "?" in event_type or "[" in event_type:
            self._pattern_handlers.append(sub)
            self._pattern_handlers.sort(key=lambda s: s.priority)
        else:
            self._exact_handlers[event_type].append(sub)
            self._exact_handlers[event_type].sort(key=lambda s: s.priority)

        return sub

    def unsubscribe(self, subscription: Subscription) -> None:
        """Remove a subscription.

        Args:
            subscription: The Subscription handle returned by subscribe().
        """
        target_id = subscription.subscription_id

        if subscription.pattern in self._exact_handlers:
            self._exact_handlers[subscription.pattern] = [
                s
                for s in self._exact_handlers[subscription.pattern]
                if s.subscription_id != target_id
            ]
            if not self._exact_handlers[subscription.pattern]:
                del self._exact_handlers[subscription.pattern]

        self._pattern_handlers = [
            s for s in self._pattern_handlers if s.subscription_id != target_id
        ]

    def publish(self, event: Event) -> None:
        """Publish an event to all matching subscribers.

        If the bus is disabled, this is a no-op.

        Args:
            event: The event to publish.
        """
        if not self._enabled:
            return

        # Collect matching handlers (exact + pattern)
        handlers: list[Subscription] = []

        exact = self._exact_handlers.get(event.event_type)
        if exact:
            handlers.extend(exact)

        for sub in self._pattern_handlers:
            if fnmatch.fnmatch(event.event_type, sub.pattern):
                handlers.append(sub)

        # Sort by priority and invoke
        handlers.sort(key=lambda s: (s.priority, s.subscription_id))

        for sub in handlers:
            try:
                sub.handler(event)
            except Exception:
                logger.exception(
                    "Event handler %r failed for event %s",
                    sub.handler,
                    event.event_type,
                )

    def clear(self) -> None:
        """Remove all subscriptions."""
        self._exact_handlers.clear()
        self._pattern_handlers.clear()

    @property
    def subscriber_count(self) -> int:
        """Total number of active subscriptions."""
        exact_count = sum(len(subs) for subs in self._exact_handlers.values())
        return exact_count + len(self._pattern_handlers)
