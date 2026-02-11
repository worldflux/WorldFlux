# OSS Release Checklist

Minimum criteria before tagging a public release.

## CI and Quality Gates

- [ ] Lint passes: `uvx ruff check src/ tests/ examples/ benchmarks/ scripts/`
- [ ] Format check passes: `uvx ruff format --check src/ tests/ examples/ benchmarks/ scripts/`
- [ ] Typecheck passes: `uv run mypy src/worldflux/`
- [ ] Test suite passes: `uv run pytest tests/`
- [ ] Example smoke tests pass:
  - `uv run python examples/quickstart_cpu_success.py --quick`
  - `uv run python examples/compare_unified_training.py --quick`
  - `uv run python examples/train_dreamer.py --test`
  - `uv run python examples/train_tdmpc2.py --test`
  - `uv run python examples/train_jepa.py --steps 5 --batch-size 4 --obs-dim 8`
  - `uv run python examples/train_token_model.py --steps 5 --batch-size 4 --seq-len 8 --vocab-size 32`
  - `uv run python examples/train_diffusion_model.py --steps 5 --batch-size 4 --obs-dim 4 --action-dim 2`
  - `uv run python examples/plan_cem.py --horizon 3 --action-dim 2`
  - `uv pip install -e examples/plugins/minimal_plugin && uv run python examples/plugins/smoke_minimal_plugin.py`
- [ ] Benchmark quick checks pass:
  - `uv run python benchmarks/benchmark_dreamerv3_atari.py --quick --seed 42`
  - `uv run python benchmarks/benchmark_tdmpc2_mujoco.py --quick --seed 42`
  - `uv run python benchmarks/benchmark_diffusion_imagination.py --quick --seed 42`
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
- [ ] Trusted Publishing setup is validated against [Publishing Guide](publishing.md)
