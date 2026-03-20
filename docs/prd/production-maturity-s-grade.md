---
id: prd-production-maturity-s-grade
title: Production Maturity S-Grade Program
owner: platform
status: proposed
priority: p0
target_window: 90d
depends_on:
  - prd-code-quality-s-grade
in_scope:
  - deterministic training mode
  - structured logging
  - config reliability
  - artifact rigor
out_of_scope:
  - operating a hosted training service
allowed_paths:
  - src/worldflux/training/**
  - src/worldflux/telemetry/**
  - docs/guides/**
  - scripts/**
blocked_paths:
  - third_party/**
verification_commands:
  - uv run pytest -q tests/test_training/
  - uv run pytest -q tests/test_telemetry/
acceptance_gates:
  - deterministic mode is documented and test-covered
  - artifact and logging paths are reproducible
---

# Production Maturity S-Grade Program

## 1. Problem

Operational reliability is uneven across config merges, telemetry, and training
artifacts.

## 2. Why Now

Reproducibility gaps block trustworthy evidence generation and release rigor.

## 3. S-Level End State

Training runs expose deterministic, observable, and reproducible operational
behavior.

## 4. 90-Day Target

Add deterministic execution paths, reliable override merges, structured logging,
and stronger artifact discipline.

## 5. Requirements

- Deterministic mode must be opt-in, explicit, and tested.
- Logging backends must preserve structured fields.
- Artifact paths must be stable enough for release and parity workflows.

## 6. Non-Goals

- Building a full remote orchestration platform.
- Replacing all existing telemetry backends at once.

## 7. Implementation Plan

Address override loading, telemetry structure, and artifact handling in bounded
tasks that each update docs and tests.

## 8. Verification

Run training and telemetry test suites plus any new deterministic-mode checks.

## 9. Rollout and Rollback

Ship deterministic and logging changes behind explicit config switches where
needed; if compatibility breaks, roll back only the new path.

## 10. Open Questions

- Which artifact invariants should become release-blocking first?
