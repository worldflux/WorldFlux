# ADR 0005: Adapter Pattern for Component Wrapping

## Status

Accepted

## Context

WorldFlux components can be synchronous or asynchronous. When a user
provides a sync-only encoder but the execution path needs async
operation, the framework must bridge the gap transparently. Similarly,
the deprecated `RolloutEngine` interface needs to coexist with the
current `RolloutExecutor` interface.

## Decision

Use the Adapter pattern for interface bridging:

- `AsyncObservationEncoderAdapter` wraps a sync `ObservationEncoder`
  and exposes `encode_async()` via `asyncio.to_thread()`.
- `AsyncDynamicsModelAdapter` wraps a sync `DynamicsModel` and
  exposes `transition_async()`.
- `AsyncDecoderAdapter` wraps a sync `Decoder` and exposes
  `decode_async()`.
- `AsyncRolloutExecutorAdapter` wraps a sync `RolloutExecutor` and
  exposes `rollout_open_loop_async()`.

Helper functions (`ensure_async_observation_encoder`, etc.) provide
a clean API for obtaining an async-capable component from either
sync or async inputs.

The `WorldModel` base class uses `_run_component_async()` internally
to dispatch to either native async methods or thread-wrapped sync
methods, providing a uniform async interface.

## Consequences

- Users can provide sync-only components and still use async execution
  paths (e.g. `async_encode`, `async_transition`).
- The adapter overhead is minimal - sync methods are dispatched to
  `asyncio.to_thread()` only when called from async context.
- New async protocols (`AsyncObservationEncoder`, etc.) allow
  components to provide native async implementations for better
  performance.
- The deprecated `rollout_engine` slot is transparently mapped to
  `rollout_executor` through the adapter and registry alias system.
- Component protocol checks (`isinstance`) work correctly because
  adapters implement the target protocol.
