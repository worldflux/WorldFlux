# Model Developer Guide

## 5-Layer Component Architecture

Every world model is built from five pluggable component slots, defined in `WorldModelRegistry._component_slots` (`src/worldflux/core/registry.py`):

| Slot | Attribute | Interface | Purpose |
|------|-----------|-----------|---------|
| `observation_encoder` | `observation_encoder` | `ObservationEncoder` | Raw observations to latent State |
| `action_conditioner` | `action_conditioner` | `ActionConditioner` | Fuse action + condition into dynamics input |
| `dynamics_model` | `dynamics_model` | `DynamicsModel` | Predict next latent state |
| `decoder` | `decoder_module` | `Decoder` | Latent state to predictions (obs, reward, continue) |
| `rollout_executor` | `rollout_executor` | `RolloutExecutor` | Multi-step open-loop rollouts |

The base `WorldModel` class (`src/worldflux/core/model.py`) delegates `encode()`, `transition()`, `decode()`, and `rollout()` to these components. Override only when default delegation is insufficient.

## Registration Pattern

Register a new model using the `@WorldModelRegistry.register()` decorator:

```python
from worldflux.core.registry import WorldModelRegistry
from worldflux.core.model import WorldModel
from worldflux.core.config import WorldModelConfig

@WorldModelRegistry.register("mymodel", config_class=MyModelConfig)
class MyWorldModel(WorldModel):
    def __init__(self, config: MyModelConfig):
        super().__init__()
        self.config = config
        # Initialize components...
        self.capabilities = {Capability.LATENT_DYNAMICS, Capability.OBS_DECODER}
        self.composable_support = {"observation_encoder", "decoder"}

    def loss(self, batch: Batch) -> LossOutput:
        # Required: training objective
        ...

    def io_contract(self) -> ModelIOContract:
        # Recommended: declares I/O shapes for runtime validation
        ...
```

Duplicate `model_type` registration raises `ConfigurationError`. Use `WorldModelRegistry.unregister()` if needed.

## ModelMaturity Levels

Defined in `src/worldflux/core/spec.py`:

| Level | Meaning | Examples |
|-------|---------|---------|
| `REFERENCE` | Production-grade, parity-proven | DreamerV3, TD-MPC2 |
| `EXPERIMENTAL` | Functional but not parity-proven | JEPA, V-JEPA2, Token, Diffusion |
| `SKELETON` | Interface validation only, minimal logic | DiT, SSM, Renderer3D, Physics, GAN |

Only REFERENCE models are parity proof targets. Set maturity in the `MODEL_CATALOG` entry.

## Required Methods from WorldModel ABC

At minimum, implement:

- **`loss(batch: Batch) -> LossOutput`** (abstract) -- Training objective. Must return a `LossOutput` with `.loss` tensor and `.components` dict.

Recommended overrides:

- **`io_contract() -> ModelIOContract`** -- Declares observation, action, state, and prediction specs. Trainer validates batches against this contract at runtime.
- **`encode(obs, ...) -> State`** -- Override if default `observation_encoder` delegation is insufficient.
- **`transition(state, action, ...) -> State`** -- Override if default `dynamics_model` delegation is insufficient.
- **`decode(state, ...) -> ModelOutput`** -- Override if default `decoder_module` delegation is insufficient.
- **`rollout(state, actions, ...) -> Trajectory`** -- Override if default `rollout_executor` delegation is insufficient.

## Test Placement

- Unit tests: `tests/test_<model_family>.py` (e.g., `tests/test_dreamer.py`)
- Factory integration tests: `tests/test_factory.py`
- CI presets: Use tiny configs (`dreamer:ci`, `tdmpc2:ci`) for fast validation.
- Run specific test files, not the full suite, for speed.

## Factory Catalog Entry

Add entries to `MODEL_CATALOG` in `src/worldflux/factory.py`:

```python
MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "mymodel:base": {
        "description": "MyModel base - description here",
        "params": "~5M",
        "type": "mymodel",
        "default_obs": "vector",
        "maturity": ModelMaturity.EXPERIMENTAL.value,
    },
}
```

Also add user-friendly aliases to `MODEL_ALIASES`:

```python
MODEL_ALIASES: dict[str, str] = {
    "mymodel": "mymodel:base",
}
```

Aliases and catalog entries are auto-registered on import via `WorldModelRegistry.register_alias()` and `WorldModelRegistry.register_catalog_entry()`.
