# 2026 Q2 WorldFlux Quality Program

This roadmap operationalizes the 2026-03-19 S-grade master spec for the next
90 days. It is the canonical roadmap for the structured quality program.

## Phase 0: Program Setup (Days 1-7)

Objectives:

- install an agent-readable delivery structure
- add missing process gates
- stop false signals before deeper feature work continues

Mandatory outputs:

- root `AGENTS.md`
- PRD directory skeleton under `docs/prd/`
- task envelope guidance under `docs/tasks/`
- collect-only CI gate
- release metadata alignment

## Phase 1: Correctness and Contract Recovery (Days 8-35)

Objectives:

- fix ML correctness blockers
- fix public API and runtime mismatches
- remove broken or misleading surfaces

Mandatory outputs:

- Dreamer image objective alignment fix
- TD-MPC2 terminal-aware target fix with regression coverage
- TD-MPC2 target update side-effect removal with regression coverage
- CLI, docs, and API consistency fixes

Current focus inside this phase:

- keep the newcomer `init -> train -> verify --mode quick` path aligned across
  CLI, scaffolded helpers, and docs
- keep Dreamer reference-family correctness guarded by explicit regression tests
  before making stronger training claims

## Phase 2: Production and Quality Hardening (Days 36-60)

Objectives:

- reproducibility
- operational logging
- config reliability
- checkpoint and artifact rigor

Mandatory outputs:

- deterministic training mode
- real override merge path
- structured logging integration
- lockfile-aware container path

## Phase 3: Scale and Proof Differentiation (Days 61-90)

Objectives:

- real data-parallel foundations
- replay scaling improvements
- publishable technical proof

Mandatory outputs:

- shard-aware data path
- replay backend redesign for large datasets
- one publishable parity or evidence bundle
- capability comparison against competing frameworks

## Program References

- Master spec: `docs/superpowers/specs/2026-03-19-worldflux-s-grade-program-design.md`
- Program plan: `docs/superpowers/plans/2026-03-19-worldflux-s-grade-program.md`
- Task templates: `docs/tasks/README.md`
