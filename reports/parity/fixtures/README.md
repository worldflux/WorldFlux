# Release Parity Fixtures

These files are deterministic release-gate fixtures.

- The checked-in source of truth is the fixture specification in `scripts/generate_release_parity_fixtures.py`.
- Running that script regenerates local ignored JSON under `reports/parity/runs/`, `reports/parity/fixtures/*.json`, and `reports/parity/aggregate.json`.
- They are not proof-grade parity evidence.
- They must not be cited as public equivalence claims or paper-reproduction evidence.

Regenerate them with:

```bash
uv run python scripts/generate_release_parity_fixtures.py
```
