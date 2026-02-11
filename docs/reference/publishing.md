# Publishing (PyPI Trusted Publisher)

WorldFlux uses PyPI Trusted Publishing (OIDC) from GitHub Actions.

## Workflow Model

- Build artifacts in a dedicated build job.
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
3. Create a GitHub release tag.
4. Publish workflow runs automatically on release publication.

## Troubleshooting

- OIDC permission error: verify `id-token: write` on publish jobs.
- Publisher not recognized: verify owner/repo/workflow/environment names exactly.
- Artifact issues: ensure build job uploaded `dist/` and publish jobs download it.
