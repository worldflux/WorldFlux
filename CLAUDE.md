# WorldFlux

## Commands

```bash
uv sync --extra dev          # Install dev deps
uv run pytest tests/ -v      # Run tests
uv run pytest tests/test_factory.py -x  # Run single file
uv run ruff check src/ tests/ examples/  # Lint
uv run ruff check --fix src/  # Auto-fix
uv run mypy src/worldflux/   # Type check
uv run mkdocs build --strict # Build docs (catches broken refs)
make ci                      # Full local CI gate
```

## Workflow

- Run `make ci` after changes to verify before marking done.
- New `.py` files must include `from __future__ import annotations`.
- `uv.lock` is committed and hash-verified in CI. Run `uv lock` after dependency changes.

## Key Decisions

- v3 API is current. v0.2 is deprecated bridge mode.
- DreamerV3 and TD-MPC2 are the only production models (parity proof targets). All others are experimental/skeleton.

## Testing

- Run specific test files, not the full suite, for speed.
- CI presets (`dreamer:ci`, `tdmpc2:ci`) are tiny models for fast testing.

## Do NOT

- Modify parity proof artifacts in `reports/parity/`.
- Change `tests/fixtures/public_contract_snapshot.json` without intent (it freezes the public API).
- Use `torch.load(weights_only=False)` outside Trainer checkpoint loading. Model weights always use `weights_only=True`.

## Gotchas

- `ReplayBuffer` is NOT thread-safe. Single writer thread only.
- mkdocs build requires `--strict` — broken cross-refs fail the build.
