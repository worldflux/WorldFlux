task_id: wf-phase1-tdmpc2-terminal-target
title: TD-MPC2 terminal-aware target and target-Q side-effect removal
priority: p0
depends_on:
  - prd-ml-correctness-s-grade
allowed_paths:
  - src/worldflux/models/tdmpc2/world_model.py
  - src/worldflux/core/model.py
  - src/worldflux/training/trainer.py
  - tests/test_training/test_tdmpc2_phase1_correctness.py
  - docs/reference/paper-alignment-tdmpc2.md
blocked_paths:
  - third_party/**
problem_statement: >
  TD-MPC2 currently bootstraps through terminal transitions and mutates target-Q
  state inside loss() evaluation, which makes the loss impure and weakens
  correctness guarantees.
implementation_constraints:
  - No unrelated refactors
  - Update tests and docs in same task
  - Keep the fix additive and internal-only
verification_commands:
  - uv run pytest tests/test_training/test_tdmpc2_phase1_correctness.py -v
  - uv run pytest tests/test_models/test_tdmpc2_reference_fidelity.py tests/test_training/test_trainer_safety_guards.py -v
done_when:
  - TD targets do not bootstrap when the resulting transition is terminal
  - loss() no longer mutates target-Q parameters
  - target-Q EMA runs after optimizer.step() instead

# Background

This task implements two explicit Phase 1 mandatory outputs from the S-grade
program spec:

1. TD-MPC2 terminal-aware target fix
2. TD-MPC2 target update side-effect removal

# Rollback

Revert this task if the trainer hook causes cross-family regressions, but keep
the regression tests and narrow the hook contract rather than restoring
loss-time mutation.
