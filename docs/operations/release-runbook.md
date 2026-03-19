# Release Runbook

This runbook describes the minimum operator flow for preparing and publishing a
WorldFlux release.

## Preconditions

- CI is green on `main`
- release checklist items in `docs/reference/release-checklist.md` are complete
- parity evidence artifacts and verification report are available

## Release Flow

1. Confirm target tag/version and changelog scope.
2. Run the release dry-run helper:
   ```bash
   uv run python scripts/run_release_dry_run.py --tag vX.Y.Z --profile verify
   ```
3. Review generated verification outputs under `reports/release/`.
4. Confirm parity evidence artifacts under `reports/parity/`.
5. Publish the GitHub release and allow `.github/workflows/release.yml` to run.

## Ownership

- Release Manager: final go/no-go decision
- Maintainers: checklist completion, artifact review, workflow triage

## Related Docs

- `docs/reference/release-checklist.md`
- `docs/operations/maintainer-onboarding.md`
- `docs/roadmap.md`
