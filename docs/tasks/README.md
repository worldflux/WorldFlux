# Task Documents

Store execution-scoped task documents for the S-grade program in `docs/tasks/`.

Every task document should use the standard task envelope below before adding
task-specific prose.

```yaml
task_id: unique-id
title: short title
priority: p0|p1|p2
depends_on: []
allowed_paths: []
blocked_paths: []
problem_statement: >
  Exact defect or gap being addressed.
implementation_constraints:
  - No unrelated refactors
  - Update tests and docs in same task
verification_commands: []
done_when:
  - measurable condition
```

Recommended sections after the envelope:

1. Background
2. Acceptance Notes
3. Rollback
4. Evidence Links
