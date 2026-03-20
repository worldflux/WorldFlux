---
id: prd-architecture-s-grade
title: Architecture S-Grade Program
owner: platform
status: proposed
priority: p0
target_window: 90d
depends_on:
  - prd-code-quality-s-grade
in_scope:
  - dependency boundary enforcement
  - backend abstraction cleanup
  - composable support metadata
out_of_scope:
  - full package tree rewrite
allowed_paths:
  - src/worldflux/core/**
  - src/worldflux/training/**
  - src/worldflux/execution/**
  - docs/architecture/**
  - docs/adr/**
blocked_paths:
  - third_party/**
verification_commands:
  - uv run pytest -q tests/test_public_contract_freeze.py
  - uv run pytest -q tests/test_factory.py
acceptance_gates:
  - no forbidden layer imports remain
  - component support is machine-readable
---

# Architecture S-Grade Program

## 1. Problem

The documented layer model and the real dependency graph have drifted.

## 2. Why Now

Boundary drift is still small enough to correct without a repository rewrite.

## 3. S-Level End State

Architecture boundaries are documented and mechanically enforced.

## 4. 90-Day Target

Enforce dependency direction, isolate delegated execution concerns, and publish
runtime-derived support metadata.

## 5. Requirements

- `training` must not directly depend on parity orchestration concerns.
- Boundary rules must be CI-enforced.
- Component support must be explicit and queryable.
- Architecture decisions must be recorded in ADRs.

## 6. Non-Goals

- Rewriting all packages into a new layout.
- Redesigning every internal abstraction in one pass.

## 7. Implementation Plan

Add a dependency-check path, narrow backend bridges, and generate support
manifests from code rather than hand-written tables.

## 8. Verification

Run the listed pytest commands and confirm docs match allowed import edges.

## 9. Rollout and Rollback

Land boundary checks in warn-first mode only if breakage blocks adoption; roll
back to the last passing rule set if import enforcement is mis-specified.

## 10. Open Questions

- Which import-lint tool best fits current CI time budgets?
