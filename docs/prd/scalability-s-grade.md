---
id: prd-scalability-s-grade
title: Scalability S-Grade Program
owner: platform
status: proposed
priority: p1
target_window: 90d
depends_on:
  - prd-production-maturity-s-grade
  - prd-architecture-s-grade
in_scope:
  - shard-aware data paths
  - replay scaling improvements
  - data-parallel foundations
out_of_scope:
  - unbounded distributed systems redesign
allowed_paths:
  - src/worldflux/training/**
  - src/worldflux/core/**
  - docs/reference/**
blocked_paths:
  - third_party/**
verification_commands:
  - uv run pytest -q tests/test_training/test_distributed_config.py
  - uv run pytest -q tests/test_training/test_replay_backends.py
acceptance_gates:
  - replay and distributed paths support larger datasets without API drift
  - scalability changes are benchmarkable and documented
---

# Scalability S-Grade Program

## 1. Problem

Current replay and distributed paths are good enough for smoke workloads but not
yet framed for larger-scale operation.

## 2. Why Now

Scale work must build on explicit contracts before larger artifacts and parity
bundles depend on it.

## 3. S-Level End State

WorldFlux supports larger data and training footprints through explicit,
benchmarkable scaling paths.

## 4. 90-Day Target

Add shard-aware data handling, improve replay backends, and prepare real
data-parallel foundations.

## 5. Requirements

- Scalability changes must preserve stable public contracts.
- Replay backend behavior must remain testable under larger workloads.
- Distributed configuration must be explicit and documented.

## 6. Non-Goals

- Solving every multi-node orchestration concern in one phase.
- Shipping scale claims without evidence.

## 7. Implementation Plan

Improve replay and distributed primitives in bounded, test-backed increments.

## 8. Verification

Run replay backend and distributed configuration tests plus any new benchmarks.

## 9. Rollout and Rollback

Keep scale paths additive where possible; if new backends regress existing runs,
fall back to the prior backend while preserving tests.

## 10. Open Questions

- Which replay bottleneck should be addressed first for largest user impact?
