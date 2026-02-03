# LatentState

The core representation used by all world models.

## Overview

`LatentState` is a dataclass that holds the latent representation of an observation. It provides a unified interface across different model architectures.

```python
from worldflux.core.state import LatentState
```

## Creating a LatentState

Typically created by calling `model.encode()`:

```python
state = model.encode(obs)
```

## Attributes

### deterministic

The deterministic component of the latent state.

```python
state.deterministic  # Tensor[B, deter_dim]
```

- **DreamerV3**: GRU hidden state (history/context)
- **TD-MPC2**: SimNorm embedding (full representation)

### stochastic

The stochastic component (uncertainty representation).

```python
state.stochastic  # Tensor[B, stoch_dim] or None
```

- **DreamerV3**: Categorical samples (captures uncertainty)
- **TD-MPC2**: `None` (implicit model, no stochastic component)

### features

Combined representation for downstream heads.

```python
state.features  # Tensor[B, total_dim]
```

Concatenates `deterministic` and `stochastic` (if present) for use in decoders and prediction heads.

### posterior_logits / prior_logits

Logits for the stochastic distribution (DreamerV3 only).

```python
state.posterior_logits  # Tensor or None
state.prior_logits      # Tensor or None
```

## Model-Specific Shapes

### DreamerV3

| Preset | deterministic | stochastic | features |
|--------|--------------|------------|----------|
| size12m | [B, 2048] | [B, 1024] | [B, 3072] |
| size25m | [B, 2048] | [B, 1024] | [B, 3072] |
| size50m | [B, 4096] | [B, 1024] | [B, 5120] |
| size100m | [B, 4096] | [B, 2048] | [B, 6144] |
| size200m | [B, 8192] | [B, 2048] | [B, 10240] |

### TD-MPC2

| Preset | deterministic | stochastic | features |
|--------|--------------|------------|----------|
| 5m | [B, 512] | None | [B, 512] |
| 19m | [B, 512] | None | [B, 512] |
| 48m | [B, 1024] | None | [B, 1024] |
| 317m | [B, 2048] | None | [B, 2048] |

## Usage Examples

### Basic Usage

```python
# Encode observation
state = model.encode(obs)

# Access components
print(f"Deterministic: {state.deterministic.shape}")
print(f"Stochastic: {state.stochastic}")  # May be None
print(f"Features: {state.features.shape}")
```

### Checking Model Type

```python
if state.stochastic is not None:
    print("This is a DreamerV3-style model with uncertainty")
else:
    print("This is a TD-MPC2-style implicit model")
```

### Using Features

```python
# Features are used for downstream predictions
features = state.features

# Custom head (example)
my_output = my_head(features)
```

### Batch Operations

```python
# States support batching
batch_obs = torch.randn(32, 3, 64, 64)
batch_state = model.encode(batch_obs)

print(batch_state.deterministic.shape)  # [32, deter_dim]
```

## Implementation Details

```python
@dataclass
class LatentState:
    """Universal latent state representation."""

    deterministic: torch.Tensor
    stochastic: torch.Tensor | None = None
    posterior_logits: torch.Tensor | None = None
    prior_logits: torch.Tensor | None = None
    codebook_indices: torch.Tensor | None = None

    @property
    def features(self) -> torch.Tensor:
        """Combined features for downstream heads."""
        if self.stochastic is not None:
            return torch.cat([self.deterministic, self.stochastic], dim=-1)
        return self.deterministic
```
