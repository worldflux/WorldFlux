# Release Parity Fixtures

These files are deterministic release-gate fixtures.

- They exist so `scripts/run_release_dry_run.py` can regenerate `reports/parity/runs/*.json` and `reports/parity/aggregate.json` on any checkout.
- They are not proof-grade parity evidence.
- They must not be cited as public equivalence claims or paper-reproduction evidence.

Regenerate them with:

```bash
uv run python scripts/generate_release_parity_fixtures.py
```
