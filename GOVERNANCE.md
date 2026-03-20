# Governance

## Decision Making

WorldFlux uses **Lazy Consensus** as the default decision-making process.

### Lazy Consensus

A proposal is considered accepted if:

1. The proposal is posted as a GitHub Issue or PR discussion.
2. At least 72 hours have passed since the proposal was posted.
3. No maintainer has raised a blocking objection.
4. At least one maintainer has expressed support (thumbs-up, approval, or
   explicit comment).

For non-trivial architectural decisions, a longer review period (7 days) is
recommended.

### Escalation

- If consensus is blocked and a decision is needed, the Release Manager decides.
- Emergency path: the Release Manager can apply immediate mitigations to protect
  users, then documents follow-up actions in an issue.
- Disputes between maintainers are resolved through private discussion first,
  then escalated to a majority vote among all active maintainers if needed.

## Change Management

- Public API-affecting changes require tests and documentation updates.
- Breaking changes require explicit changelog entries and migration notes.
- Release gates in CI must stay green before merge to `main`.
- Release operations follow `docs/operations/release-runbook.md`.

## Access and Permissions

- Write access is limited to maintainers listed in `MAINTAINERS.md`.
- Changes to workflows and release automation require maintainer review.
- Access grants and removals are tracked in repository settings and documented in
  maintainer discussions.
- Current technical priorities are tracked in `ROADMAP.md`.
- The structured S-grade quality program is tracked in
  `docs/roadmap/2026-q2-worldflux-quality-program.md`.

## Committer Promotion Criteria

See `MAINTAINERS.md` for detailed promotion criteria and process. Summary:

1. 10+ merged PRs over 3+ months of sustained contribution.
2. 5+ constructive code reviews demonstrating domain understanding.
3. Unanimous maintainer approval (or Lazy Consensus after 7 days).
4. Adherence to Code of Conduct.

## Roles

| Role | Access | Responsibilities |
| --- | --- | --- |
| Contributor | Fork + PR | Submit patches, report issues, participate in discussions |
| Committer | Write access | Merge PRs, triage issues, review code |
| Maintainer | Write + admin | Release management, governance decisions, access control |
| Release Manager | Full admin | Final release approval, emergency response, tie-breaking |

## Emergency Procedures

### Security Incidents

1. The reporter follows `SECURITY.md` to disclose the vulnerability privately.
2. The Release Manager acknowledges within 48 hours.
3. A fix is developed in a private fork or branch.
4. A security advisory is published alongside the patched release.
5. Post-incident review is conducted and documented within 14 days.

### Service Disruption (PyPI, Docs, CI)

1. Any maintainer can apply temporary mitigations immediately.
2. The Release Manager is notified within 4 hours.
3. Root cause analysis is documented in a GitHub Issue within 7 days.

### Maintainer Unavailability

If the sole maintainer (Release Manager) is unavailable for 14+ days:

1. Emeritus maintainers or nominated contributors may be granted temporary
   write access by GitHub organization owners.
2. Temporary access is limited to critical bug fixes and security patches.
3. Full access decisions are deferred until the Release Manager returns.

## Amendments

Changes to this governance document require:

1. A pull request with the proposed changes.
2. Lazy Consensus among all active maintainers (7-day minimum review).
3. No blocking objections from any active maintainer.
