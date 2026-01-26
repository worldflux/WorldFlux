# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WorldLoom is a unified Python interface for latent world models used in reinforcement learning (DreamerV3, TD-MPC2, with V-JEPA planned). It provides a common API across different architectures for encoding observations, predicting dynamics, and imagination rollouts.

## Commands

```bash
# Install dependencies (dev mode)
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_models/test_dreamer.py

# Run a specific test
pytest tests/test_models/test_dreamer.py::TestDreamerV3WorldModel::test_encode

# Type checking
mypy src/

# Linting
ruff check src/

# Fix lint issues
ruff check --fix src/
```

## Architecture

### Core Protocol (`src/worldloom/core/protocol.py`)

The `WorldModel` Protocol defines the unified interface all models must implement:
- `encode(obs)` → `LatentState` - Observation to latent state
- `predict(state, action)` → `LatentState` - Prior/imagination step (no observation)
- `observe(state, action, obs)` → `LatentState` - Posterior step (with observation)
- `decode(state)` → predictions dict - Latent to observation/reward/continue
- `imagine(initial_state, actions)` → `Trajectory` - Multi-step rollout
- `compute_loss(batch)` → losses dict - Training losses

### State Representation (`src/worldloom/core/state.py`)

`LatentState` is a universal dataclass supporting multiple architectures:
- DreamerV3: `deterministic` (h) + `stochastic` (z) + `posterior_logits`/`prior_logits`
- TD-MPC2: `deterministic` only (SimNorm embedding)
- VQ-VAE: `codebook_indices`

The `.features` property concatenates components for downstream heads.

### Latent Spaces (`src/worldloom/core/latent_space.py`)

Three implementations extending `LatentSpace` ABC:
- `GaussianLatentSpace` - Continuous with reparameterization
- `CategoricalLatentSpace` - Discrete with Gumbel-Softmax (DreamerV3)
- `SimNormLatentSpace` - Simplicial normalization (TD-MPC2)

### Model Registry (`src/worldloom/core/registry.py`)

HuggingFace-style auto-loading with decorators:
```python
# Register with decorator
@WorldModelRegistry.register("dreamer", DreamerV3Config)
class DreamerV3WorldModel(nn.Module): ...

# Load by preset
model = AutoWorldModel.from_pretrained("dreamerv3:size12m")
model = AutoWorldModel.from_pretrained("tdmpc2:5m")
```

### Model Implementations

**DreamerV3** (`src/worldloom/models/dreamer/`):
- RSSM dynamics with categorical stochastic state
- CNN/MLP encoder-decoder
- Reward and continue prediction heads
- Size presets: size12m, size25m, size50m, size100m, size200m

**TD-MPC2** (`src/worldloom/models/tdmpc2/`):
- Implicit model (no decoder)
- SimNorm latent space
- Q-function ensemble for planning
- Size presets: 5m, 19m, 48m, 317m

### Configuration (`src/worldloom/core/config.py`)

Dataclass configs with `from_size()` factory methods for standard presets. Configs serialize to JSON for `save_pretrained`/`from_pretrained`.

## Key Patterns

- All models register via `@WorldModelRegistry.register(type, ConfigClass)`
- Use `state.features` to get combined latent for prediction heads
- Batch format for `compute_loss`: `{obs, actions, rewards, continues}` with shape `[batch, seq_len, ...]`
- Tests use smaller configs (e.g., `cnn_depth=16`, `hidden_dim=128`) for speed
