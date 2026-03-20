# WorldFlux Agent Rules

This repository uses agent-readable execution envelopes for program work derived
from the WorldFlux S-grade quality program.

## Non-Negotiable Rules

- **No AI agent traces in git history.** Commit messages, author fields, trailers
  (Co-Authored-By, Signed-off-by, etc.), branch names, PR descriptions, and any
  other git-visible metadata must NEVER contain references to AI assistants,
  LLMs, or automated agents (e.g. "Claude", "GPT", "Copilot", "AI-generated").
  This rule is absolute and has zero exceptions.
- No public claim may exceed what code, tests, docs, and evidence bundles support.
- No stable public surface may expose placeholder behavior unless it is marked
  experimental and documented as incomplete.
- Every major change must update code, tests, and docs in the same task.
- No execution task may rely on prose-only semantics when the behavior can be
  encoded as tests or machine-readable metadata.

## Required Task Envelope

Every implementation task derived from the program must include this shape:

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

## Review Expectations

- `allowed_paths` must be explicit before editing begins.
- `blocked_paths` must prevent unrelated refactors and third-party drift.
- `verification_commands` must be executable as-written from repository root.
- `done_when` must be measurable, not aspirational.
- Additive, breaking, and internal-only changes must be labeled in the task or
  PRD that authorizes the work.
