# WorldFlux Extensibility Guide

This document provides a comprehensive analysis of WorldFlux's architecture, extensibility characteristics, and guidelines for extending the framework with new world model architectures.

WorldFlux uses a **contract-first** policy: every model family defines and validates
its runtime I/O contract before it is treated as release-ready.

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

WorldFlux implements a **unified API** through the `WorldModel` base class, ensuring all world models share a consistent interface:

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

| Operation | Purpose | DreamerV3 | TD-MPC2 |
|-----------|---------|-----------|---------|
| `encode` | obs → latent | CNN/MLP encoder | MLP encoder |
| `transition` | prior transition | RSSM prior | Dynamics network |
| `update` | posterior update | RSSM posterior | N/A (implicit) |
| `decode` | latent → prediction | CNN/MLP decoder | None (implicit) |
| `rollout` | multi-step rollout | Prior sequence | Dynamics sequence |
| `loss` | training losses | ELBO components | TD + auxiliary |

### Design Patterns Verified

**1. Registry Pattern** (`src/worldflux/core/registry.py`)
```python
@WorldModelRegistry.register("dreamer", DreamerV3Config)
class DreamerV3WorldModel(nn.Module): ...

# Usage: AutoWorldModel.from_pretrained("dreamerv3:size12m")
```

**2. Universal State Representation** (`src/worldflux/core/state.py`)
- `State` supports all architectures via `tensors` + `meta`
- Deterministic (h), stochastic (z), VQ-VAE, and Gaussian components stored by key
- Downstream heads consume model-specific tensor keys

**3. Polymorphic Latent Spaces** (`src/worldflux/core/latent_space.py`)
- `GaussianLatentSpace` - VAE-style models
- `CategoricalLatentSpace` - DreamerV3 (discrete)
- `SimNormLatentSpace` - TD-MPC2 (simplicial normalization)

**4. Unified Trainer** (`src/worldflux/training/trainer.py`)
- Works with any `WorldModel` via duck-typing
- Requires only `loss(batch)` method
- Handles checkpointing, logging, scheduling

---

## Extensibility Assessment

### Overall Score: **7.5/10**

| Category | Score | Description |
|----------|-------|-------------|
| **Model Addition** | 9/10 | Registry pattern + base class = easy integration |
| **Latent Space** | 8/10 | ABC extensible; may need new `State.tensors` keys |
| **Training** | 8/10 | Trainer abstracted; custom losses supported |
| **State Representation** | 7/10 | Universal container; VQ/flow may need new tensor keys |
| **Decoder Patterns** | 6/10 | Implicit models supported; diffusion decoders need work |
| **Temporal Modeling** | 6/10 | Autoregressive/video models need new patterns |

### Strengths

1. **Zero Existing Code Changes**: New models register without modifying existing implementations
2. **Type Safety**: Base class + shared types (`Batch`, `State`, `ModelOutput`)
3. **HuggingFace Compatibility**: `from_pretrained`/`save_pretrained` patterns
4. **Flexible State**: `State.meta` for architecture-specific data
5. **Consistent Training**: Single `Trainer` class works across all models

### Areas for Enhancement

1. **Video/Sequence Models**: May need batch dimension handling
2. **Diffusion Decoders**: Iterative decoding not yet abstracted
3. **Hierarchical States**: Multi-scale representations
4. **Flow-based Models**: Invertible transformations

---

## Maturity Tiers (Public Policy)

WorldFlux now classifies model families by maturity tier in the model catalog:

- **reference**: Production-grade baseline families with stronger compatibility expectations.
- **experimental**: Functional but evolving APIs/metrics (not yet release-grade parity).
- **skeleton**: Interface stubs intended for design exploration only.

Current default policy:

- **reference**: DreamerV3, TD-MPC2
- **experimental**: JEPA, V-JEPA2, Token, Diffusion
- **experimental (skeleton families)**: DiT, SSM, Renderer3D, Physics, GAN

Promotion rule (experimental -> reference):

- pass common quality gates (finite metrics, save/load parity, seed success >= 80%)
- pass family-specific gates
- keep API/runtime contract stable across releases

---

## World Model Classification

### Taxonomy of World Model Architectures

| Category | Examples | WorldFlux Support |
|----------|----------|-------------------|
| **Latent Dynamics (RSSM)** | DreamerV3, PlaNet | ✅ Fully supported |
| **Implicit Models** | TD-MPC2, MBPO | ✅ Fully supported |
| **Transformer Sequence** | IRIS, Gato | 🔶 Partial (needs VQ-VAE extension) |
| **Diffusion-based** | Diffusion World Models | 🔶 Partial (decoder abstraction needed) |
| **Video Prediction** | V-JEPA, VideoGPT | ⚪ Planned |
| **Foundation Models** | Cosmos, Genie 3 | ⚪ Future consideration |

### Five-Layer Pluggable Core (v0.2)

WorldFlux now standardizes model composition around five replaceable component types:

1. `ObservationEncoder`
2. `DynamicsModel`
3. `ActionConditioner`
4. `Decoder` (optional)
5. `RolloutExecutor` (open-loop execution only)

Planning strategies (`CEM`, future `MPPI`/tree-search) are defined separately through
`worldflux.planners.interfaces.Planner` and return `ActionPayload`.

New families should be assembled from these components instead of hard-coding monolithic model classes.

### Planner Metadata Contract

Planner outputs must set:

- `extras["wf.planner.horizon"]` (required in `v0.3`)

Compatibility in `v0.2`:

- if horizon is missing, it is inferred from tensor shape with `DeprecationWarning`
- `extras["wf.planner.sequence"]` is still accepted with `DeprecationWarning`

Condition extras must be namespaced:

- format: `wf.<domain>.<name>`

### Required Components by Family

| Family | Required Components |
|--------|---------------------|
| **Latent Dynamics (RSSM)** | Encoder, Dynamics, Decoder, Reward/Continue heads |
| **Implicit Models** | Encoder, Dynamics, Value/Policy heads |
| **Token Models** | Tokenizer, Dynamics (Transformer), Sampler |
| **Diffusion Models** | Encoder, Sampler (denoising), Decoder |
| **JEPA** | Encoder, Predictor, Objective (masked prediction) |
| **Foundation Models** | Tokenizer/Encoder, Large generator, Sampler, Optional planner |

### Contract Compatibility Matrix

Each model family should publish and satisfy:

- **Required capabilities** (`Capability` flags)
- **Required batch keys** (for training/eval)
- **Required state keys** (for planner/sampler hooks)
- **Sequence layout** (explicit `B`/`T` axis mapping by field)

### Category Details

#### 1. Latent Dynamics Models (RSSM-based)
- **Architecture**: Encoder → RSSM (deterministic + stochastic) → Decoder
- **State**: `State.tensors["deter"]`, `State.tensors["stoch"]`, logits in `State.tensors`
- **Examples**: DreamerV3, PlaNet, Dreamer
- **Status**: ✅ Reference implementation complete

#### 2. Implicit World Models
- **Architecture**: Encoder → Dynamics → Q-functions (no decoder)
- **State**: `State.tensors["latent"]` (SimNorm embedding)
- **Examples**: TD-MPC2, TD-MPC, MBPO
- **Status**: ✅ Reference implementation complete

#### 3. Transformer Sequence Models
- **Architecture**: VQ-VAE tokenizer → Transformer → Token prediction
-- **State**: tokens stored in `State.tensors` (e.g. `tokens`, `codebook_indices`)
- **Examples**: IRIS, GAIA-1, Gato
- **Required Changes**:
  - Store codebook embeddings/tokens in `State.tensors`
  - Add `VQVAELatentSpace` implementation
  - Support variable-length sequences

#### 4. Diffusion World Models
- **Architecture**: Encoder → Latent diffusion → Iterative decoder
-- **State**: diffusion latents + timestep in `State.meta`
- **Examples**: Diffusion World Models, DIAMOND
- **Required Changes**:
  - Add `DiffusionLatentSpace` with score function
  - Extend `decode()` for iterative denoising
  - Support noise schedule in training

#### 5. Video Prediction Models
- **Architecture**: Spatiotemporal encoder → Frame prediction
- **State**: Multi-frame representation
- **Examples**: V-JEPA, VideoGPT, MCVD
- **Required Changes**:
  - Support 5D tensors (batch, time, channels, height, width)
  - Add frame-level prediction heads
  - Temporal consistency losses

#### 6. Foundation World Models
- **Architecture**: Large-scale pretrained models
- **Examples**: Cosmos (NVIDIA), Genie 3 (DeepMind)
- **Considerations**:
  - Multi-modal inputs (video, text, actions)
  - Massive parameter counts (billions)
  - Inference optimization critical

---

## Adding New Models

### Step-by-Step Guide

#### Step 1: Create Config Class

```python
# src/worldflux/core/config.py
@dataclass
class MyModelConfig(WorldModelConfig):
    model_type: str = "mymodel"
    custom_param: int = 256

    @classmethod
    def from_size(cls, size: str, **kwargs) -> "MyModelConfig":
        presets = {"small": {"custom_param": 128}, "large": {"custom_param": 512}}
        return cls(**presets.get(size, {}), **kwargs)
```

#### Step 2: Implement World Model

```python
# src/worldflux/models/mymodel/world_model.py
from worldflux.core.registry import WorldModelRegistry
from worldflux.core.state import State
from worldflux.core.output import LossOutput, ModelOutput

@WorldModelRegistry.register("mymodel", MyModelConfig)
class MyWorldModel(nn.Module):
    def __init__(self, config: MyModelConfig):
        super().__init__()
        self.config = config
        # Initialize components...

    def encode(self, obs: Tensor, deterministic: bool = False) -> State:
        # obs → latent
        embedding = self.encoder(obs)
        return State(tensors={"latent": embedding}, meta={"latent_type": "deterministic"})

    def transition(self, state: State, action: Tensor, ...) -> State:
        # state, action → next_state (prior)
        next_embed = self.dynamics(state.tensors["latent"], action)
        return State(tensors={"latent": next_embed}, meta={"latent_type": "deterministic"})

    def update(self, state: State, action: Tensor, obs: Tensor) -> State:
        # For implicit models, often same as transition or not used
        return self.transition(state, action)

    def decode(self, state: State) -> ModelOutput:
        reward = self.reward_head(state.tensors["latent"])
        return ModelOutput(preds={"reward": reward})

    def rollout(self, initial_state: State, actions: Tensor, ...) -> Trajectory:
        states, rewards = [initial_state], []
        state = initial_state
        for t in range(actions.shape[1]):
            state = self.transition(state, actions[:, t])
            preds = self.decode(state)
            states.append(state)
            rewards.append(preds["reward"])
        return Trajectory(states=states, rewards=torch.stack(rewards, dim=1))

    def loss(self, batch: Batch) -> LossOutput:
        # Implement training losses
        return LossOutput(loss=total_loss, components={"reward_loss": r_loss})
```

#### Step 3: Export in Package

```python
# src/worldflux/models/mymodel/__init__.py
from .world_model import MyWorldModel
__all__ = ["MyWorldModel"]

# src/worldflux/models/__init__.py
from .mymodel import MyWorldModel
```

#### Step 4: Use the Model

```python
from worldflux import create_world_model

model = create_world_model("mymodel:small")
# or
model = AutoWorldModel.from_pretrained("mymodel:large")
```

### Extending State (If Needed)

For models requiring new state components:

```python
# Add new tensors by key
state = State(
    tensors={"latent": embed, "noise": noise},
    meta={"diffusion_timestep": t, "noise_level": sigma},
)
```

---

## Future Architecture Roadmap

### Phase 1: VQ-VAE Foundation (Q2 2026)

**Goal**: Support transformer-based models like IRIS

| Component | Change Type | Description |
|-----------|-------------|-------------|
| `VQVAELatentSpace` | Additive | New latent space class |
| `State.tensors["codebook_embeddings"]` | Additive | Optional tensor key |
| `TransformerDynamics` | Additive | Autoregressive dynamics |

**Implementation Priority**: IRIS → Gato-style models

### Phase 2: Diffusion Integration (Q3 2026)

**Goal**: Support diffusion-based world models

| Component | Change Type | Description |
|-----------|-------------|-------------|
| `DiffusionLatentSpace` | Additive | Score-based sampling |
| `IterativeDecoder` | Additive | Multi-step decode interface |
| `NoiseScheduler` | Additive | Training utility |

**Implementation Priority**: Diffusion World Models → DIAMOND

### Phase 3: Video Prediction (Q4 2026)

**Goal**: Support V-JEPA and video generation models

| Component | Change Type | Description |
|-----------|-------------|-------------|
| Video batch handling | Modification | 5D tensor support |
| `TemporalConsistencyLoss` | Additive | Training loss |
| `FramePredictionHead` | Additive | Multi-frame output |

**Implementation Priority**: V-JEPA → VideoGPT

### Phase 4: Foundation Model Compatibility (2027+)

**Goal**: Integration with large-scale pretrained models

**Considerations for Cosmos/Genie 3**:
- **Scale**: Models with billions of parameters
- **Multi-modal**: Text + video + action inputs
- **Inference**: Need efficient serving (ONNX, TensorRT)
- **Fine-tuning**: LoRA/adapter patterns
- **API**: Possibly external API integration

| Challenge | Approach |
|-----------|----------|
| Memory constraints | Gradient checkpointing, model sharding |
| Multi-modal inputs | Unified tokenization interface |
| Inference speed | Quantization, ONNX export |
| Adaptation | Parameter-efficient fine-tuning |

---

## Change Type Analysis

### Additive Changes (Backward Compatible)

These changes extend functionality without breaking existing code:

| Change | Impact | Existing Code Affected |
|--------|--------|------------------------|
| New model registration | None | ❌ No changes needed |
| New `State` tensor key | None | ❌ Use new key in `State.tensors` |
| New `LatentSpace` subclass | None | ❌ ABC inheritance |
| New loss function | None | ❌ Opt-in via config |
| New decoder type | None | ❌ Factory pattern |

### Potentially Breaking Changes

These changes require careful migration:

| Change | Risk Level | Mitigation |
|--------|------------|------------|
| `WorldModel` signature change | High | Version the API |
| Required `State` tensor key | Medium | Deprecation period |
| `Trainer` interface change | Medium | Adapter pattern |
| Batch dimension convention | High | Feature flag |

### Backward Compatibility Guidelines

1. **Optional Fields**: Always add with `default=None`
2. **New Methods**: Add to base class with default implementation
3. **Config Changes**: New fields must have defaults
4. **Deprecation**: Minimum 2 minor versions warning

### Extension Points (No Core Changes)

| Extension | Mechanism |
|-----------|-----------|
| Custom model | `@WorldModelRegistry.register()` |
| Custom config | Inherit `WorldModelConfig` |
| Custom latent space | Inherit `LatentSpace` ABC |
| Custom trainer callback | Implement `Callback` interface |
| Custom data loader | Implement `ReplayBuffer` interface |

---

## Summary

WorldFlux provides a **highly extensible** framework for world models with:

- **Base-class API** ensuring consistent interfaces
- **Registry pattern** enabling zero-modification model addition
- **Universal state representation** accommodating diverse architectures
- **Unified training** working across all model types

**Extensibility Score: 7.5/10**

The framework is well-positioned for current latent-space models (DreamerV3, TD-MPC2) and has clear paths for extending to:
- Transformer sequence models (IRIS)
- Diffusion world models
- Video prediction models (V-JEPA)
- Foundation models (Cosmos, Genie 3)

Most future architectures can be added with **purely additive changes**, maintaining backward compatibility with existing implementations.
