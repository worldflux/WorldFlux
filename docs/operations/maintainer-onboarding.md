# Maintainer Onboarding

This document describes the minimum onboarding path for a new WorldFlux
maintainer.

## Required Access

- GitHub repository write access
- access to release workflows and artifact review
- access to issue triage and security response channels

## Core Responsibilities

- review API, training, and docs changes
- keep release and parity gates healthy
- follow `docs/operations/release-runbook.md` for release-critical changes

## First-Week Checklist

1. Read `GOVERNANCE.md`
2. Read `MAINTAINERS.md`
3. Read `docs/operations/release-runbook.md`
4. Review `docs/roadmap.md`
5. Run the local CI gate once:
   ```bash
   uv run python scripts/run_local_ci_gate.py
   ```
