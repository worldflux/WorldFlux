# ADR 0001: Core Boundary Policy

## Status

Accepted

## Context

WorldFlux exposes one public API across multiple model families, training paths,
and verification backends. As the repository grows, the largest architectural
risk is accidental coupling between:

- `worldflux.core`
- `worldflux.models`
- `worldflux.training`
- `worldflux.verify`
- `worldflux.execution`
- `worldflux.parity`

## Decision

- `worldflux.core` is the canonical contract layer.
- `worldflux.models` implements `core` contracts.
- `worldflux.training` consumes model contracts and must not redefine them.
- `worldflux.verify` may evaluate models and artifacts, but must treat `core`
  contracts as the source of truth.
- `worldflux.execution` and `worldflux.parity` may orchestrate external proof
  and backend-native flows, but should not leak backend-specific semantics back
  into `core`.

## Rules

- New public interfaces should land in `core` only if they are family-agnostic.
- Backend/proof-specific metadata belongs in `execution` or `parity`.
- Component override semantics are governed by `WorldModelRegistry` and model
  `composable_support`, not by ad hoc family code.
- Public top-level exports default to stable unless explicitly marked otherwise
  by API stability metadata.

## Consequences

- `core` remains small, typed, and reusable.
- New families can be added without changing training or proof semantics.
- Docs and tooling can derive support matrices from registry/model metadata.
