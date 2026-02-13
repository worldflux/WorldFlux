# Publishing (PyPI Trusted Publisher)

WorldFlux uses PyPI Trusted Publishing (OIDC) from GitHub Actions.

## Workflow Model

- Build artifacts in a dedicated build job.
- Validate release metadata (`tag/version/changelog`) before build.
- Validate fixed parity artifacts (`DreamerV3` + `TD-MPC2`) before build/publish.
- Generate verification report artifacts (`verification-report.json` / `.md`) and attach to release workflow artifacts.
- Publish in separate jobs with `id-token: write`.
- No PyPI API token/password is required in GitHub secrets.

See workflow: `.github/workflows/release.yml`

## PyPI-side Setup (one-time)

1. Open your project on PyPI.
2. Go to **Publishing** -> **Add a new pending publisher**.
3. Set:
   - Owner: `worldflux`
   - Repository: `WorldFlux`
   - Workflow: `release.yml`
   - Environment (optional but recommended): `pypi`
4. Save publisher and run a test release.

For TestPyPI, repeat the same setup on TestPyPI and map the `testpypi` environment.

## Release Procedure

1. Ensure CI is green on `main`.
2. Update `CHANGELOG.md` and version metadata.
3. Validate release metadata locally:
   - `uv run python scripts/check_release_metadata.py --tag vX.Y.Z`
4. (Optional) Run parity pipeline report generation:
   - `worldflux parity run ...`
   - `worldflux parity aggregate ...`
   - `worldflux parity report ...`
5. Validate release parity gate against fixed artifacts:
   - `uv run python scripts/validate_parity_artifacts.py --run reports/parity/runs/dreamer_atari100k.json --run reports/parity/runs/tdmpc2_dmcontrol39.json --aggregate reports/parity/aggregate.json --lock reports/parity/upstream_lock.json --required-suite dreamer_atari100k --required-suite tdmpc2_dmcontrol39 --max-missing-pairs 0`
6. Create a GitHub release tag.
7. Publish workflow runs automatically on release publication.

## Troubleshooting

- OIDC permission error: verify `id-token: write` on publish jobs.
- Publisher not recognized: verify owner/repo/workflow/environment names exactly.
- Artifact issues: ensure build job uploaded `dist/` and publish jobs download it.
- Parity gate fail:
  - Check `reports/parity/aggregate.json` for `ci_upper_ratio > margin_ratio` or missing pairs.
  - Check `reports/parity/upstream_lock.json` commit pins match run artifacts.
