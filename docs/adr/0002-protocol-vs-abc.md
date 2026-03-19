# ADR 0002: Protocol vs ABC for Component Interfaces

## Status

Accepted

## Context

WorldFlux defines five pluggable component slots (observation_encoder,
action_conditioner, dynamics_model, decoder_module, rollout_executor).
These interfaces need to be checkable at runtime while remaining
flexible enough to accept third-party implementations that do not
inherit from WorldFlux base classes.

Two patterns were evaluated:

1. **ABC (Abstract Base Class)** - requires explicit inheritance,
   provides `@abstractmethod` enforcement at class definition time.
2. **Protocol (PEP 544)** - structural subtyping, allows any class
   with matching methods to satisfy the interface without inheritance.

## Decision

Use `typing.Protocol` with `@runtime_checkable` for all component
interfaces.

Rationale:

- **Third-party compatibility**: Plugin authors can implement
  WorldFlux interfaces without importing or inheriting from the
  framework. A plain class with `encode(observations) -> State` is
  a valid `ObservationEncoder`.
- **Duck typing alignment**: Protocols match Python's duck-typing
  philosophy. Components work if they have the right methods,
  regardless of inheritance chain.
- **Runtime validation**: `@runtime_checkable` enables `isinstance()`
  checks in the component validation framework (ARCH-01) without
  requiring registration.
- **Minimal coupling**: Protocols do not impose constructor signatures,
  lifecycle hooks, or state requirements on implementors.

## Consequences

- Component interfaces are defined as `Protocol` classes in
  `core/interfaces.py`.
- Runtime checks use `isinstance(component, ProtocolType)`.
- Method signature mismatches are only caught at call time (not at
  class definition time as with ABC). The validation framework
  mitigates this gap.
- `@runtime_checkable` checks only method existence, not full
  signatures. The `ComponentValidator` protocol adds deeper
  validation for components that opt in.
