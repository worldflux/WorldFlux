# ADR 0006: Callback vs Event Bus

## Status

Accepted

## Context

The original training system uses a Callback class with 6 fixed hooks:
`on_train_begin`, `on_train_end`, `on_epoch_begin`, `on_epoch_end`,
`on_step_begin`, `on_step_end`. This design has limitations:

- Adding new hook points requires modifying the Callback ABC and all
  CallbackList dispatch methods.
- Fine-grained events (loss computed, gradients clipped, NaN detected)
  cannot be observed without subclassing internal Trainer methods.
- No priority ordering - callbacks execute in registration order.
- No pattern-based filtering - callbacks receive all events for their
  hook type.

## Decision

Introduce an EventBus system (`core/events.py`) that supplements (not
replaces) the existing Callback system:

- **EventBus** provides publish/subscribe with pattern matching and
  priority ordering.
- **18 event types** cover model lifecycle, training flow, compute
  steps, errors, and recovery actions.
- **CallbackAdapter** bridges legacy Callback instances to the
  EventBus, providing full backward compatibility.
- The EventBus is opt-in on the Trainer (default `None`) with
  near-zero overhead when unused.

The existing 6 Callback hooks remain the primary stable API. The
EventBus is the extension mechanism for fine-grained observability.

## Consequences

- Existing Callback subclasses continue to work without modification.
- New event types can be added without changing any existing interface.
- `subscribe("train.*", handler)` pattern matching enables flexible
  event filtering.
- Priority ordering ensures deterministic handler execution order.
- The EventBus is optional - training works identically without it.
- Error recovery (ARCH-06) publishes events through the EventBus,
  enabling external monitoring without coupling to resilience internals.
