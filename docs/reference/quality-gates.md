# OSS Quality Gates

This document defines the **release readiness gates** for WorldFlux.

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

## Reproducibility Gates (Nightly / Release)

- **Seed stability**: same seed produces key losses/metrics within ±5–10%.
- **Save/Load parity**: outputs after `save_pretrained` and reload differ only within tolerance.
- **Seed success rate**: at least **80%** of seeds pass family-specific success criteria.
- **Confidence interval reporting**: report bootstrap CI for success rate on every run.

## Benchmark Gates (Release Only)

- **Loss trend**: training loss decreases by a minimum threshold within N steps.
- **Numerical stability**: no NaN/Inf during training in defined step budget.

## Family-Specific Gates

- **Reference families** (DreamerV3, TD-MPC2):
  - loss drop threshold
  - finite loss components
  - seed success rate >= 80%
- **Token family**:
  - token cross-entropy/perplexity trend
  - finite logits and loss
- **Diffusion family**:
  - denoise error trend
  - finite scheduler states and losses
- **JEPA family**:
  - representation prediction loss trend
  - finite context/target projections
- **V-JEPA2 family**:
  - representation prediction loss trend
  - finite context/target/mask pipelines
- **Skeleton families** (DiT/SSM/Renderer3D/Physics/GAN):
  - contract validation pass
  - trainer 1-step smoke pass
  - no core modifications required when adding additional family variants

- **Planner boundary gates**:
  - planner returns `ActionPayload` with `extras["wf.planner.horizon"]`
  - dynamics family swap does not require planner code changes

## Recommended Threshold Defaults

These can be tightened over time:

- Loss decrease: **≥10%** within **2,000 steps** on bundled datasets.
- Seed variance: **≤10%** on main loss components.
- Extensibility gate: new category integrations should be additive to `src/worldflux/models/` only.

## Output Schema (Summary)

`scripts/measure_quality_gates.py` writes summary fields including:

- `success_rate`
- `gates.common.ci_low`
- `gates.common.ci_high`
- `gates.family_pass`

Interpretation:

- `gates.common.pass` checks finite metrics + success rate threshold (default 80%).
- `gates.family_pass` is true only if both common gates and family-specific gates pass.

## Measurement Commands

The following scripts are committed for reproducible measurement runs.
Defaults use **bundled datasets** and **small CI-sized models**.

### Run Measurements (Balanced)

```bash
uv run python scripts/measure_quality_gates.py \
  --models dreamer,tdmpc2 \
  --seeds 0,1,2,3,4 \
  --steps 5000 \
  --device auto
```

Output:

- `reports/quality-gates/seed_runs.json`

### Estimate Runtime from Measured Results

```bash
uv run python scripts/estimate_quality_gate_runtime.py \
  --input reports/quality-gates/seed_runs.json \
  --scenarios 5x5000,7x10000,10x10000,7x20000,10x20000
```
