# WorldFlux Extensibility Guide

This document describes how to extend WorldFlux safely while keeping API contracts stable.

WorldFlux follows a **contract-first** approach: each model family must define runtime I/O contracts before wider integration.

## Table of Contents

1. [Implementation Verification](#implementation-verification)
2. [Extensibility Assessment](#extensibility-assessment)
3. [World Model Classification](#world-model-classification)
4. [Adding New Models](#adding-new-models)
5. [Future Architecture Roadmap](#future-architecture-roadmap)
6. [Change Type Analysis](#change-type-analysis)

---

## Implementation Verification

### Unified API Verification

WorldFlux provides a unified API through the `WorldModel` base class so all model families share a consistent interface:

```python
class WorldModel(ABC):
    config: WorldModelConfig

    def encode(self, obs: Tensor | dict[str, Tensor] | WorldModelInput, deterministic: bool = False) -> State: ...
    def transition(self, state: State, action: ActionPayload | Tensor | None, conditions: ConditionPayload | None = None, deterministic: bool = False) -> State: ...
    def update(self, state: State, action: ActionPayload | Tensor | None, obs: Tensor | dict[str, Tensor] | WorldModelInput, conditions: ConditionPayload | None = None) -> State: ...
    def decode(self, state: State, conditions: ConditionPayload | None = None) -> ModelOutput: ...
    def rollout(self, initial_state: State, action_sequence: ActionSequence | Tensor | None, conditions: ConditionPayload | None = None, deterministic: bool = False, mode: str = "autoregressive") -> Trajectory: ...
    def initial_state(self, batch_size: int, device: ...) -> State: ...
    def loss(self, batch: Batch) -> LossOutput: ...
```

### Design Patterns

1. **Registry Pattern** (`src/worldflux/core/registry.py`)

```python
@WorldModelRegistry.register("dreamer", DreamerV3Config)
class DreamerV3WorldModel(nn.Module): ...
```

2. **Universal State Representation** (`src/worldflux/core/state.py`)
- `State` supports model-specific tensor keys via `tensors` + `meta`.

3. **Polymorphic Latent Spaces** (`src/worldflux/core/latent_space.py`)
- `GaussianLatentSpace`
- `CategoricalLatentSpace`
- `SimNormLatentSpace`

4. **Unified Trainer** (`src/worldflux/training/trainer.py`)
- Common training flow for model families implementing the `WorldModel` contract.

---

## Extensibility Assessment

### Architectural Characteristics

| Category | Notes |
|----------|-------|
| Model addition | Registry-based registration and config classes make family additions straightforward |
| Latent space extension | New latent-space implementations can be added through existing abstractions |
| Training integration | Trainer works with any model that implements `loss(batch)` and contracted outputs |
| State representation | `State.tensors` and `State.meta` allow additive model-specific fields |
| Decoder patterns | Optional decoder path is explicit in the model contract |

### Strengths

- Additive model-family integration via registry.
- Shared runtime contracts and payload types.
- Unified serialization path (`save_pretrained` / `from_pretrained`).
- External plugin discovery hooks.

### Areas for Enhancement

- Additional support patterns for long-horizon video/sequence use cases.
- Broader reusable abstractions for iterative decoders/samplers.
- More examples for advanced external plugin packaging.

---

## World Model Classification

### Taxonomy of Architecture Families

| Category | Examples | Support Status |
|----------|----------|----------------|
| Latent Dynamics (RSSM) | DreamerV3, PlaNet | Supported |
| Implicit Dynamics | TD-MPC2, MBPO-style | Supported |
| Transformer Sequence | IRIS-like families | Partial |
| Diffusion-based | Diffusion world models | Partial |
| Video Prediction | V-JEPA style families | Planned extension |
| Foundation-style | Large multimodal families | Planned extension |

### Five-Layer Pluggable Core

WorldFlux standardizes model composition around replaceable components:

1. `ObservationEncoder`
2. `DynamicsModel`
3. `ActionConditioner`
4. `Decoder` (optional)
5. `RolloutExecutor` (open-loop execution)

Factory-level overrides are supported:

```python
create_world_model(..., component_overrides={"action_conditioner": "my.plugin.component"})
```

### Planner Metadata Contract

Planner outputs must include:

- `extras["wf.planner.horizon"]`

### External Plugin Hooks

Third-party packages can register through entry-point groups:

- `worldflux.models`
- `worldflux.components`

Plugin manifests should provide compatibility metadata (`plugin_api_version`, `worldflux_version_range`, `capabilities`).

### Required Components by Family

| Family | Required Components |
|--------|---------------------|
| Latent Dynamics | Encoder, Dynamics, Decoder, Reward/Continue heads |
| Implicit Models | Encoder, Dynamics, Value/Policy heads |
| Token Models | Tokenizer, Dynamics, Sampler |
| Diffusion Models | Encoder, Sampler, Decoder |
| JEPA-style | Encoder, Predictor, Objective |

---

## Adding New Models

### Step 1: Create Config Class

```python
# src/worldflux/core/config.py
@dataclass
class MyModelConfig(WorldModelConfig):
    model_type: str = "mymodel"
    custom_param: int = 256
```

### Step 2: Implement World Model

```python
# src/worldflux/models/mymodel/world_model.py
@WorldModelRegistry.register("mymodel", MyModelConfig)
class MyWorldModel(nn.Module):
    def encode(self, obs: Tensor, deterministic: bool = False) -> State: ...
    def transition(self, state: State, action: Tensor, ...) -> State: ...
    def update(self, state: State, action: Tensor, obs: Tensor) -> State: ...
    def decode(self, state: State) -> ModelOutput: ...
    def rollout(self, initial_state: State, actions: Tensor, ...) -> Trajectory: ...
    def loss(self, batch: Batch) -> LossOutput: ...
```

### Step 3: Export and Register

- Export symbols in `src/worldflux/models/mymodel/__init__.py`
- Ensure registry and factory aliases are configured where needed.

### Step 4: Add Tests and Docs

- Add family tests under `tests/test_models/`.
- Document required batch/state keys and contract expectations.

---

## Future Architecture Roadmap

The following areas are tracked as technical extension directions:

- Tokenization/VQ pathways for sequence-oriented models.
- Additional iterative sampler/decoder abstractions.
- Enhanced support for video-shaped tensors and temporal objectives.
- Integration patterns for large-scale external pretrained families.

These are implementation directions, not release commitments.

---

## Change Type Analysis

### Additive Changes (Backward Compatible)

| Change | Expected Impact |
|--------|------------------|
| New model registration | No impact on existing families |
| New optional `State.tensors` key | Additive usage only |
| New latent-space subclass | Isolated to new family |
| New callback/loss utilities | Opt-in |

### Potentially Breaking Changes

| Change | Risk | Mitigation |
|--------|------|------------|
| `WorldModel` signature changes | High | Versioned migration path |
| Required state-key changes | Medium | Deprecation period and validation errors |
| Batch convention changes | High | Feature flags and adapters |

### Compatibility Guidelines

- Add new config fields with safe defaults.
- Keep optional extensions additive where possible.
- Provide explicit migration notes for contract-affecting changes.

### Extension Points

| Extension | Mechanism |
|-----------|-----------|
| Custom model | `@WorldModelRegistry.register()` |
| Custom config | Inherit `WorldModelConfig` |
| Custom latent space | Inherit `LatentSpace` |
| Custom callback | Implement callback interface |
| Custom data source | Implement `BatchProvider` / `BatchProviderV2` |
