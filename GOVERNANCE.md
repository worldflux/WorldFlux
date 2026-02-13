# Governance

## Decision Making

- Default path: maintainer consensus on pull requests and design discussions.
- If consensus is blocked and a decision is needed, the Release Manager decides.
- Emergency path: the Release Manager can apply immediate mitigations to protect
  users, then documents follow-up actions in an issue.

## Change Management

- Public API-affecting changes require tests and documentation updates.
- Breaking changes require explicit changelog entries and migration notes.
- Release gates in CI must stay green before merge to `main`.

## Access and Permissions

- Write access is limited to maintainers listed in `MAINTAINERS.md`.
- Changes to workflows and release automation require maintainer review.
- Access grants and removals are tracked in repository settings and documented in
  maintainer discussions.
