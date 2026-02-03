# WorldFlux Extensibility Guide

This document provides a comprehensive analysis of WorldFlux's architecture, extensibility characteristics, and guidelines for extending the framework with new world model architectures.

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

WorldFlux implements a **unified API** through Python's `Protocol` class, ensuring all world models share a consistent interface:

```python
@runtime_checkable
class WorldModel(Protocol):
    config: WorldModelConfig

    def encode(self, obs: Tensor, deterministic: bool = False) -> LatentState: ...
    def predict(self, state: LatentState, action: Tensor, ...) -> LatentState: ...
    def observe(self, state: LatentState, action: Tensor, obs: Tensor) -> LatentState: ...
    def decode(self, state: LatentState) -> dict[str, Tensor]: ...
    def imagine(self, initial_state: LatentState, actions: Tensor, ...) -> Trajectory: ...
    def initial_state(self, batch_size: int, device: ...) -> LatentState: ...
    def compute_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]: ...
```

| Operation | Purpose | DreamerV3 | TD-MPC2 |
|-----------|---------|-----------|---------|
| `encode` | obs → latent | CNN/MLP encoder | MLP encoder |
| `predict` | prior transition | RSSM prior | Dynamics network |
| `observe` | posterior update | RSSM posterior | N/A (implicit) |
| `decode` | latent → prediction | CNN/MLP decoder | None (implicit) |
| `imagine` | multi-step rollout | Prior sequence | Dynamics sequence |
| `compute_loss` | training losses | ELBO components | TD + auxiliary |

### Design Patterns Verified

**1. Registry Pattern** (`src/worldflux/core/registry.py`)
```python
@WorldModelRegistry.register("dreamer", DreamerV3Config)
class DreamerV3WorldModel(nn.Module): ...

# Usage: AutoWorldModel.from_pretrained("dreamerv3:size12m")
```

**2. Universal State Representation** (`src/worldflux/core/state.py`)
- `LatentState` dataclass supports all architectures
- Deterministic (h), stochastic (z), VQ-VAE, and Gaussian components
- `.features` property unifies downstream head inputs

**3. Polymorphic Latent Spaces** (`src/worldflux/core/latent_space.py`)
- `GaussianLatentSpace` - VAE-style models
- `CategoricalLatentSpace` - DreamerV3 (discrete)
- `SimNormLatentSpace` - TD-MPC2 (simplicial normalization)

**4. Unified Trainer** (`src/worldflux/training/trainer.py`)
- Works with any `WorldModel` via duck-typing
- Requires only `compute_loss(batch)` method
- Handles checkpointing, logging, scheduling

---

## Extensibility Assessment

### Overall Score: **7.5/10**

| Category | Score | Description |
|----------|-------|-------------|
| **Model Addition** | 9/10 | Registry pattern + Protocol = easy integration |
| **Latent Space** | 8/10 | ABC extensible; may need new `LatentState` fields |
| **Training** | 8/10 | Trainer abstracted; custom losses supported |
| **State Representation** | 7/10 | Universal dataclass; VQ/flow may need extension |
| **Decoder Patterns** | 6/10 | Implicit models supported; diffusion decoders need work |
| **Temporal Modeling** | 6/10 | Autoregressive/video models need new patterns |

### Strengths

1. **Zero Existing Code Changes**: New models register without modifying existing implementations
2. **Type Safety**: Protocol-based interface with `@runtime_checkable`
3. **HuggingFace Compatibility**: `from_pretrained`/`save_pretrained` patterns
4. **Flexible State**: `LatentState.metadata` for architecture-specific data
5. **Consistent Training**: Single `Trainer` class works across all models

### Areas for Enhancement

1. **Video/Sequence Models**: May need batch dimension handling
2. **Diffusion Decoders**: Iterative decoding not yet abstracted
3. **Hierarchical States**: Multi-scale representations
4. **Flow-based Models**: Invertible transformations

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

### Category Details

#### 1. Latent Dynamics Models (RSSM-based)
- **Architecture**: Encoder → RSSM (deterministic + stochastic) → Decoder
- **State**: `deterministic` + `stochastic` + `prior/posterior_logits`
- **Examples**: DreamerV3, PlaNet, Dreamer
- **Status**: ✅ Reference implementation complete

#### 2. Implicit World Models
- **Architecture**: Encoder → Dynamics → Q-functions (no decoder)
- **State**: `deterministic` only (SimNorm embedding)
- **Examples**: TD-MPC2, TD-MPC, MBPO
- **Status**: ✅ Reference implementation complete

#### 3. Transformer Sequence Models
- **Architecture**: VQ-VAE tokenizer → Transformer → Token prediction
- **State**: `codebook_indices` (VQ-VAE tokens)
- **Examples**: IRIS, GAIA-1, Gato
- **Required Changes**:
  - Extend `LatentState` with `codebook_embeddings`
  - Add `VQVAELatentSpace` implementation
  - Support variable-length sequences

#### 4. Diffusion World Models
- **Architecture**: Encoder → Latent diffusion → Iterative decoder
- **State**: `deterministic` + diffusion timestep
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
from worldflux.core.state import LatentState

@WorldModelRegistry.register("mymodel", MyModelConfig)
class MyWorldModel(nn.Module):
    def __init__(self, config: MyModelConfig):
        super().__init__()
        self.config = config
        # Initialize components...

    def encode(self, obs: Tensor, deterministic: bool = False) -> LatentState:
        # obs → latent
        embedding = self.encoder(obs)
        return LatentState(deterministic=embedding, latent_type="deterministic")

    def predict(self, state: LatentState, action: Tensor, ...) -> LatentState:
        # state, action → next_state (prior)
        next_embed = self.dynamics(state.features, action)
        return LatentState(deterministic=next_embed, latent_type="deterministic")

    def observe(self, state: LatentState, action: Tensor, obs: Tensor) -> LatentState:
        # For implicit models, often same as predict or not used
        return self.predict(state, action)

    def decode(self, state: LatentState) -> dict[str, Tensor]:
        return {"reward": self.reward_head(state.features)}

    def imagine(self, initial_state: LatentState, actions: Tensor, ...) -> Trajectory:
        states, rewards = [initial_state], []
        state = initial_state
        for t in range(actions.shape[1]):
            state = self.predict(state, actions[:, t])
            preds = self.decode(state)
            states.append(state)
            rewards.append(preds["reward"])
        return Trajectory(states=states, rewards=torch.stack(rewards, dim=1))

    def compute_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # Implement training losses
        return {"loss": total_loss, "reward_loss": r_loss}
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

### Extending LatentState (If Needed)

For models requiring new state components:

```python
# Option 1: Use metadata (recommended for model-specific data)
state = LatentState(
    deterministic=embed,
    metadata={"diffusion_timestep": t, "noise_level": sigma}
)

# Option 2: Propose new field (for widely-used components)
# Submit PR to add field to LatentState dataclass
```

---

## Future Architecture Roadmap

### Phase 1: VQ-VAE Foundation (Q2 2026)

**Goal**: Support transformer-based models like IRIS

| Component | Change Type | Description |
|-----------|-------------|-------------|
| `VQVAELatentSpace` | Additive | New latent space class |
| `LatentState.codebook_embeddings` | Additive | Optional field |
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
| New `LatentState` optional field | None | ❌ Uses `field(default=None)` |
| New `LatentSpace` subclass | None | ❌ ABC inheritance |
| New loss function | None | ❌ Opt-in via config |
| New decoder type | None | ❌ Factory pattern |

### Potentially Breaking Changes

These changes require careful migration:

| Change | Risk Level | Mitigation |
|--------|------------|------------|
| `Protocol` method signature change | High | Version the Protocol |
| Required `LatentState` field | Medium | Deprecation period |
| `Trainer` interface change | Medium | Adapter pattern |
| Batch dimension convention | High | Feature flag |

### Backward Compatibility Guidelines

1. **Optional Fields**: Always add with `default=None`
2. **New Methods**: Add to Protocol with default implementation
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

- **Protocol-based API** ensuring consistent interfaces
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
