# OSS Quality Gates

This document defines baseline quality checks for WorldFlux.

## CI Gates (Required on PR)

All PRs must pass:

- **Lint**: `uvx ruff check src/ tests/ examples/`
- **Format**: `uvx ruff format --check src/ tests/ examples/`
- **Typecheck**: `uv run mypy src/worldflux/`
- **Unit tests**: `uv run pytest tests/`
- **Example smoke tests**:
  - `uv run python examples/train_dreamer.py --test`
  - `uv run python examples/train_tdmpc2.py --test`
  - `uv run python examples/train_jepa.py --steps 5 --batch-size 4 --obs-dim 8`
  - `uv run python examples/train_token_model.py --steps 5 --batch-size 4 --seq-len 8 --vocab-size 32`
  - `uv run python examples/train_diffusion_model.py --steps 5 --batch-size 4 --obs-dim 4 --action-dim 2`
  - `uv run python examples/plan_cem.py --horizon 3 --action-dim 2`
- **Docs build**: `uv run mkdocs build --strict`
- **Planner boundary tests**: verify planner/dynamics decoupling invariants

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
uvx ruff check src/ tests/ examples/
uvx ruff format --check src/ tests/ examples/
uv run mypy src/worldflux/
uv run pytest tests/
uv run mkdocs build --strict
```
