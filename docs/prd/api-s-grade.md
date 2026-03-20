---
id: prd-api-s-grade
title: API S-Grade Program
owner: platform
status: proposed
priority: p0
target_window: 90d
depends_on:
  - prd-architecture-s-grade
in_scope:
  - public CLI and docs consistency
  - stable surface labeling
  - contract freeze coverage
out_of_scope:
  - net-new end-user feature expansion
allowed_paths:
  - src/worldflux/cli/**
  - src/worldflux/api/**
  - docs/api/**
  - docs/reference/**
blocked_paths:
  - third_party/**
verification_commands:
  - uv run pytest -q tests/test_public_contract_freeze.py
  - uv run pytest -q tests/test_cli.py
acceptance_gates:
  - public docs and runtime flags agree
  - unstable surfaces are explicitly labeled
---

# API S-Grade Program

## 1. Problem

Public entry points risk over-claiming capability when docs and runtime drift.

## 2. Why Now

API mismatches erode trust faster than internal implementation debt.

## 3. S-Level End State

Every public contract is explicit, tested, and documented with conservative
language.

## 4. 90-Day Target

Close CLI and doc mismatches, label experimental surfaces, and harden public
contract freeze coverage.

The highest-priority user-facing mismatch is the scaffolded newcomer path:
`worldflux init -> worldflux train -> worldflux verify --mode quick`.
That path must stay aligned across generated files, CLI behavior, and docs.

## 5. Requirements

- CLI help, docs, and behavior must align.
- Stable and experimental surfaces must be machine-identifiable.
- Public contract regressions must fail tests before release.

## 6. Non-Goals

- Designing new major APIs unrelated to current model families.
- Backfilling every historical alias.

## 7. Implementation Plan

Audit public entry points, align reference docs, and extend contract-freeze
coverage where gaps exist.

## 8. Verification

Run public contract and CLI test paths, then inspect docs for matching language.

## 9. Rollout and Rollback

Additive labeling lands first; if a contract freeze catches a false positive,
roll back only the offending snapshot or label change.

## 10. Open Questions

- Which surfaces should remain stable in `v0.1.x` versus experimental?
