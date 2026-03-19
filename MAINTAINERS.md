# Maintainers

## Current Maintainers

| GitHub | Role | Scope |
| --- | --- | --- |
| `@yoshihyoda` | Lead Maintainer / Release Manager | Core library, CI/CD, releases, governance |

## Ownership and Responsibilities

- **Core code (`src/worldflux/`)**: API stability, architecture, release safety.
- **Workflows (`.github/workflows/`)**: quality gates, supply chain controls, publishing.
- **Documentation (`docs/`, `README.md`)**: user-facing accuracy and policy alignment.

## Review and Response SLAs

- Pull request first response: within 3 business days.
- Security report first response: within 48 hours.
- Release-critical fixes: same day triage when possible.

## Release Responsibility

- The Release Manager approves version bumps, tag creation, and final publish jobs.
- New maintainers should complete `docs/operations/maintainer-onboarding.md`.

## Committer Promotion Criteria

Contributors can be promoted to Committer (write access) when they meet the
following criteria, evaluated by existing maintainers:

1. **Sustained contribution**: 10+ merged pull requests over 3+ months.
2. **Review quality**: Demonstrated constructive, thorough code reviews on 5+ PRs.
3. **Domain expertise**: Deep understanding of at least one subsystem
   (e.g., training, parity, CLI, models).
4. **Community conduct**: Adherence to the Code of Conduct with no unresolved
   violations.
5. **Reliability**: Responsive to review requests and issue discussions.

### Promotion Process

1. Any existing maintainer may nominate a contributor by opening a private
   maintainer discussion.
2. All current maintainers vote. Approval requires unanimous consent (or Lazy
   Consensus after 7 days with no objection).
3. On approval, the Lead Maintainer grants repository write access and updates
   this file.
4. The new committer completes the onboarding checklist in
   `docs/operations/maintainer-onboarding.md`.

### Emeritus Status

Maintainers who are inactive for 6+ months may be moved to Emeritus status by
mutual agreement. Emeritus maintainers retain recognition but lose write access.
They can be reinstated through the standard promotion process.

## Onboarding Plan for New Maintainers

New maintainers should complete the following within their first 30 days:

1. Review `GOVERNANCE.md`, `CONTRIBUTING.md`, and `SECURITY.md`.
2. Complete `docs/operations/maintainer-onboarding.md`.
3. Set up local development environment and run `make ci` successfully.
4. Review 3+ open pull requests to build context.
5. Shadow one release cycle (see `docs/operations/release-runbook.md`).
6. Join the maintainer communication channel (Discord #maintainers).

## Contributor Growth Program

### Good First Issues

Maintainers should keep at least 10 open issues labeled `good first issue` at
all times, covering areas such as:

- Documentation improvements
- Test coverage gaps
- Small refactoring tasks
- Example scripts and tutorials

### Mentoring

- Each new contributor's first PR receives a thorough, encouraging review.
- Maintainers participate in monthly community office hours (Discord).
- Complex contributions receive a design review before implementation begins.
