task_id: wf-phase1-dreamer-image-objective
title: DreamerV3 image reconstruction objective alignment
priority: p0
depends_on:
  - prd-ml-correctness-s-grade
allowed_paths:
  - src/worldflux/models/dreamer/world_model.py
  - tests/test_models/test_dreamer_phase1_correctness.py
  - docs/reference/paper-alignment-dreamerv3.md
  - docs/prd/ml-correctness-s-grade.md
  - docs/roadmap/2026-q2-worldflux-quality-program.md
blocked_paths:
  - third_party/**
problem_statement: >
  DreamerV3 currently applies symlog-transformed reconstruction targets to CNN
  image observations, which misaligns the image reconstruction objective from
  the intended image path and weakens reference-family correctness claims.
implementation_constraints:
  - No unrelated refactors
  - Update tests and docs in same task
verification_commands:
  - uv run pytest tests/test_models/test_dreamer_phase1_correctness.py -v
  - uv run pytest tests/test_models/test_dreamer.py tests/test_models/test_dreamer_reference_alignment.py -v
done_when:
  - CNN/image reconstruction no longer symlogs observation targets
  - MLP/vector reconstruction behavior remains covered by regression tests
  - Dreamer alignment docs no longer describe the image objective as unresolved

# Background

This task closes the remaining Phase 1 Dreamer objective-alignment blocker
called out by the S-grade roadmap and ML correctness PRD.
