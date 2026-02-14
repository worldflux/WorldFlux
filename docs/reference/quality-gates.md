# OSS Quality Gates

This document defines baseline quality checks for WorldFlux.

## CI Gates (Required on PR)

All PRs must pass:

- **Local all-in-one gate**: `uv run python scripts/run_local_ci_gate.py`
- **Lint**: `uvx ruff check src/ tests/ examples/ benchmarks/ scripts/`
- **Format**: `uvx ruff format --check src/ tests/ examples/ benchmarks/ scripts/`
- **Typecheck**: `uv run mypy src/worldflux/`
- **Unit tests**: `uv run pytest tests/`
- **Public contract freeze**: `uv run pytest -q tests/test_public_contract_freeze.py`
- **Public contract snapshot update (additive-only)**:
  - `uv run python scripts/update_public_contract_snapshot.py --snapshot tests/fixtures/public_contract_snapshot.json`
  - `breaking` classification blocks PR; only additive changes are auto-updated.
- **Example smoke tests**:
  - `uv run python examples/quickstart_cpu_success.py --quick`
  - `uv run python examples/compare_unified_training.py --quick`
  - `uv run python examples/train_dreamer.py --test`
  - `uv run python examples/train_tdmpc2.py --test`
  - `uv run python examples/train_jepa.py --steps 5 --batch-size 4 --obs-dim 8`
  - `uv run python examples/train_token_model.py --steps 5 --batch-size 4 --seq-len 8 --vocab-size 32`
  - `uv run python examples/train_diffusion_model.py --steps 5 --batch-size 4 --obs-dim 4 --action-dim 2`
  - `uv run python examples/plan_cem.py --horizon 3 --action-dim 2`
- **Benchmark quick checks**:
  - `uv run python benchmarks/benchmark_dreamerv3_atari.py --quick --seed 42`
  - `uv run python benchmarks/benchmark_tdmpc2_mujoco.py --quick --seed 42`
  - `uv run python benchmarks/benchmark_diffusion_imagination.py --quick --seed 42`
- **Docs build**: `uv run mkdocs build --strict`
- **Docs domain TLS gate**:
  - `uv run python scripts/check_docs_domain_tls.py --host worldflux.ai --url https://worldflux.ai/ --expected-san worldflux.ai`
- **Critical coverage threshold**: `uv run python scripts/check_critical_coverage.py --report coverage.xml`
- **Planner boundary tests**: verify planner/dynamics decoupling invariants
- **Parity harness smoke**: `uv run pytest -q tests/test_parity/`
- **Parity suite governance**:
  - `uv run python scripts/check_parity_suite_coverage.py --policy reports/parity/suite_policy.json --lock reports/parity/upstream_lock.json`

## Release-only Parity Gate (Required on Release)

- Fixed parity artifacts must validate before publish:
  - `uv run python scripts/validate_parity_artifacts.py --run reports/parity/runs/dreamer_atari100k.json --run reports/parity/runs/tdmpc2_dmcontrol39.json --aggregate reports/parity/aggregate.json --lock reports/parity/upstream_lock.json --required-suite dreamer_atari100k --required-suite tdmpc2_dmcontrol39 --max-missing-pairs 0`
- Required-family suite policy must validate:
  - `uv run python scripts/check_parity_suite_coverage.py --policy reports/parity/suite_policy.json --lock reports/parity/upstream_lock.json --aggregate reports/parity/aggregate.json --enforce-pass`
- Release stops when either DreamerV3 or TD-MPC2 fails non-inferiority.

## Operational Reliability Checks

- **Save/Load parity**: outputs after `save_pretrained` and reload should remain within tolerance.
- **Numerical stability**: no NaN/Inf during defined smoke-check budgets.
- **Contract compliance**: required payload and planner metadata fields are validated.

## Family Validation Scope

- **DreamerV3 / TD-MPC2**: finite losses and stable train/eval smoke paths.
- **Token / Diffusion / JEPA / V-JEPA2**: finite outputs and smoke-train viability.
- **Skeleton families** (DiT/SSM/Renderer3D/Physics/GAN): contract validation and trainer smoke pass.

## Planner Boundary Checks

- Planner returns `ActionPayload` with `extras["wf.planner.horizon"]`.
- Dynamics family swaps do not require planner code changes.

## Recommended Commands

```bash
uv run python scripts/run_local_ci_gate.py
uvx ruff check src/ tests/ examples/ benchmarks/ scripts/
uvx ruff format --check src/ tests/ examples/ benchmarks/ scripts/
uv run mypy src/worldflux/
uv run pytest tests/
uv run pytest -q tests/test_public_contract_freeze.py
uv run pytest -q tests/test_parity/
uv run python scripts/check_parity_suite_coverage.py --policy reports/parity/suite_policy.json --lock reports/parity/upstream_lock.json
uv run python scripts/update_public_contract_snapshot.py --snapshot tests/fixtures/public_contract_snapshot.json
uv run python scripts/check_docs_domain_tls.py --host worldflux.ai --url https://worldflux.ai/ --expected-san worldflux.ai
uv run python scripts/check_critical_coverage.py --report coverage.xml
uv run mkdocs build --strict
```
