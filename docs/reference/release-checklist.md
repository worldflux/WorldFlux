# OSS Release Checklist

Minimum criteria before tagging a public release.

## CI and Quality Gates

- [ ] Lint passes: `uv run ruff check src/ tests/ examples/`
- [ ] Format check passes: `uv run ruff format --check src/ tests/ examples/`
- [ ] Typecheck passes: `uv run mypy src/worldflux/`
- [ ] Test suite passes: `uv run pytest tests/`
- [ ] Example smoke tests pass:
  - `uv run python examples/train_dreamer.py --test`
  - `uv run python examples/train_tdmpc2.py --test`
  - `uv run python examples/train_jepa.py --steps 5 --batch-size 4 --obs-dim 8`
  - `uv run python examples/train_token_model.py --steps 5 --batch-size 4 --seq-len 8 --vocab-size 32`
  - `uv run python examples/train_diffusion_model.py --steps 5 --batch-size 4 --obs-dim 4 --action-dim 2`
  - `uv run python examples/plan_cem.py --horizon 3 --action-dim 2`
- [ ] Docs build passes in strict mode: `uv run mkdocs build --strict`
- [ ] Security checks pass:
  - `uv run --with bandit bandit -r src/worldflux/ -ll`
  - `uv run --with pip-audit pip-audit`
- [ ] Planner boundary tests pass:
  - `uv run pytest -q tests/test_planners/test_cem.py tests/test_integration/test_planner_dynamics_decoupling.py`

## Packaging and Metadata

- [ ] Build succeeds: `uv run --with build python -m build`
- [ ] `CHANGELOG.md` includes release-ready notes for the tag
- [ ] Version and tag are aligned (`pyproject.toml` version matches release tag)

## Alpha Note

For Alpha releases, include links or artifacts for benchmark/repro gate runs
(for example, nightly/release `scripts/measure_quality_gates.py` outputs) in
the release notes to document current confidence.
