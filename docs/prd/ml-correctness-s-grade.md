---
id: prd-ml-correctness-s-grade
title: ML Correctness S-Grade Program
owner: research-platform
status: proposed
priority: p0
target_window: 90d
depends_on:
  - prd-architecture-s-grade
in_scope:
  - Dreamer objective alignment
  - TD-MPC2 terminal handling
  - target update correctness
out_of_scope:
  - new algorithm families
allowed_paths:
  - src/worldflux/models/**
  - src/worldflux/training/**
  - tests/test_models/**
  - tests/test_training/**
blocked_paths:
  - third_party/**
verification_commands:
  - uv run pytest -q tests/test_models/test_tdmpc2.py
  - uv run pytest -q tests/test_training/
acceptance_gates:
  - known correctness blockers are closed with regression tests
  - no silent target update side effects remain
---

# ML Correctness S-Grade Program

## 1. Problem

Reference-family training behavior still contains correctness gaps that weaken
parity claims.

## 2. Why Now

Correctness defects invalidate downstream benchmarking, docs, and evidence.

## 3. S-Level End State

Reference-family training behavior matches declared objectives and terminal
semantics with regression coverage.

## 4. 90-Day Target

Fix Dreamer objective alignment, TD-MPC2 terminal handling, and target update
side effects.

TD-MPC2 terminal handling and target-update side-effect coverage now have
explicit regression tests in-repo. Dreamer image-objective alignment is tracked
via a dedicated regression task and should be considered closed only when the
CNN reconstruction path is covered by regression tests and alignment docs are
updated in the same task.

## 5. Requirements

- Each correctness fix must ship with a failing-then-passing regression test.
- Public docs must not imply parity until evidence exists.
- Behavioral fixes must update any affected training docs.

## 6. Non-Goals

- Reproducing every paper result inside this phase.
- Broad optimizer redesign unrelated to correctness blockers.

## 7. Implementation Plan

Sequence the known blockers from highest user risk to lowest and land each with
tests plus doc notes.

## 8. Verification

Run model and training regression suites relevant to each corrected behavior.

## 9. Rollout and Rollback

Land fixes behind existing stable APIs; if a fix regresses unrelated families,
revert the specific change and keep the regression test.

## 10. Open Questions

- Which parity suites best expose terminal-handling regressions?
