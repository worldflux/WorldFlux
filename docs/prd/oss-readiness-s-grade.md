---
id: prd-oss-readiness-s-grade
title: OSS Readiness S-Grade Program
owner: maintainer
status: proposed
priority: p0
target_window: 90d
depends_on:
  - prd-api-s-grade
  - prd-code-quality-s-grade
in_scope:
  - governance and maintainer process clarity
  - contributor guidance accuracy
  - release and roadmap discoverability
out_of_scope:
  - community program expansion unrelated to engineering process
allowed_paths:
  - CONTRIBUTING.md
  - GOVERNANCE.md
  - MAINTAINERS.md
  - docs/**
blocked_paths:
  - third_party/**
verification_commands:
  - uv run pytest -q tests/test_docs/
acceptance_gates:
  - contributor docs point to current roadmap and release process
  - maintainer process is explicit and linked from governance docs
---

# OSS Readiness S-Grade Program

## 1. Problem

Open-source process artifacts need a tighter link to actual release and quality
program sources.

## 2. Why Now

As public adoption grows, ambiguous contributor guidance creates avoidable churn.

## 3. S-Level End State

Contributors and maintainers can discover the authoritative roadmap, release
process, and program tasks without guesswork.

## 4. 90-Day Target

Align governance, maintainer, and contributor documents with the structured
quality program and release process.

## 5. Requirements

- Contributor docs must point to the current roadmap and release runbook.
- Governance docs must reference the quality program source of truth.
- Maintainer onboarding must remain linked from role docs.

## 6. Non-Goals

- Building a broad community portal.
- Adding governance ceremony without operational value.

## 7. Implementation Plan

Tighten the documentation graph and add policy tests so stale references fail in
CI.

## 8. Verification

Run documentation policy tests covering roadmap, governance, and maintainer
references.

## 9. Rollout and Rollback

Ship docs and tests together; if a link target changes, update the test and
docs in the same task.

## 10. Open Questions

- Which public docs should surface the S-grade roadmap most prominently?
