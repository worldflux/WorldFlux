---
id: prd-differentiation-s-grade
title: Technical Differentiation S-Grade Program
owner: founder-platform
status: proposed
priority: p1
target_window: 90d
depends_on:
  - prd-ml-correctness-s-grade
  - prd-production-maturity-s-grade
in_scope:
  - publishable evidence bundles
  - capability comparison artifacts
  - defensible technical messaging
out_of_scope:
  - unsupported benchmark marketing
allowed_paths:
  - docs/reference/**
  - docs/guides/**
  - reports/**
  - scripts/**
blocked_paths:
  - third_party/**
verification_commands:
  - uv run pytest -q tests/test_docs/
acceptance_gates:
  - differentiation claims are backed by evidence artifacts
  - comparison docs do not exceed verified capability
---

# Technical Differentiation S-Grade Program

## 1. Problem

Differentiation claims are only defensible when they are narrower than the
evidence supporting them.

## 2. Why Now

WorldFlux needs proof-oriented artifacts before making stronger public
positioning claims.

## 3. S-Level End State

Public comparison material is evidence-backed, precise, and technically useful.

## 4. 90-Day Target

Publish at least one defensible evidence bundle and a conservative capability
comparison against adjacent frameworks.

## 5. Requirements

- No public comparison may exceed code-backed capability.
- Evidence bundles must link to the exact commands and artifacts used.
- Messaging must distinguish compatibility, reference, and proof surfaces.

## 6. Non-Goals

- Competitive claims without reproducible artifacts.
- Broad marketing copy rewrites unrelated to evidence.

## 7. Implementation Plan

Generate one publishable bundle, then derive comparison documentation from that
artifact set.

## 8. Verification

Confirm docs tests pass and evidence links resolve to generated artifacts.

## 9. Rollout and Rollback

Publish evidence first, then comparison docs; if evidence becomes stale, remove
or demote the associated claim.

## 10. Open Questions

- Which evidence bundle is smallest but still clearly differentiating?
