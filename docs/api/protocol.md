# WorldModel Protocol

All world models in WorldLoom implement this unified interface.

## Protocol Definition

```python
from typing import Protocol
import torch

class WorldModel(Protocol):
    """Unified interface for latent world models."""

    def encode(self, obs: torch.Tensor) -> LatentState:
        """Encode observation to latent state."""
        ...

    def predict(self, state: LatentState, action: torch.Tensor) -> LatentState:
        """Predict next state (imagination, no observation)."""
        ...

    def observe(self, state: LatentState, action: torch.Tensor, obs: torch.Tensor) -> LatentState:
        """Update state with observation (posterior)."""
        ...

    def decode(self, state: LatentState) -> dict[str, torch.Tensor]:
        """Decode latent state to predictions."""
        ...

    def imagine(self, initial_state: LatentState, actions: torch.Tensor) -> Trajectory:
        """Multi-step imagination rollout."""
        ...

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute training losses."""
        ...
```

## Methods

### encode

Encode an observation to a latent state.

```python
state = model.encode(obs)
```

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `obs` | `Tensor` | `[B, *obs_shape]` | Batch of observations |

**Returns**: [`LatentState`](state.md)

---

### predict

Predict next latent state without observation (prior/imagination).

```python
next_state = model.predict(state, action)
```

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `state` | `LatentState` | - | Current latent state |
| `action` | `Tensor` | `[B, action_dim]` | Actions to take |

**Returns**: [`LatentState`](state.md) - Predicted next state

---

### observe

Update latent state with actual observation (posterior).

```python
next_state = model.observe(state, action, obs)
```

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `state` | `LatentState` | - | Current latent state |
| `action` | `Tensor` | `[B, action_dim]` | Actions taken |
| `obs` | `Tensor` | `[B, *obs_shape]` | Next observation |

**Returns**: [`LatentState`](state.md) - Updated state with observation

---

### decode

Decode latent state to predictions.

```python
predictions = model.decode(state)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `state` | `LatentState` | Latent state to decode |

**Returns**: `dict[str, Tensor]`

| Key | Shape | Description |
|-----|-------|-------------|
| `"obs"` | `[B, *obs_shape]` | Reconstructed observation |
| `"reward"` | `[B, 1]` | Predicted reward |
| `"continue"` | `[B, 1]` | Episode continuation probability |

!!! note
    TD-MPC2 is an implicit model and does not reconstruct observations.
    The `"obs"` key may be absent or contain a placeholder.

---

### imagine

Multi-step imagination rollout.

```python
trajectory = model.imagine(initial_state, actions)
```

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `initial_state` | `LatentState` | - | Starting state |
| `actions` | `Tensor` | `[T, B, action_dim]` | Action sequence |

**Returns**: [`Trajectory`](#trajectory)

---

### compute_loss

Compute training losses from a batch.

```python
losses = model.compute_loss(batch)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `batch` | `dict` | Training batch (see below) |

**Batch Format**:

| Key | Shape | Description |
|-----|-------|-------------|
| `"obs"` | `[B, T, *obs_shape]` | Observation sequences |
| `"actions"` | `[B, T, action_dim]` | Action sequences |
| `"rewards"` | `[B, T]` | Reward sequences |
| `"continues"` | `[B, T]` | Episode continuation flags |

**Returns**: `dict[str, Tensor]`

| Key | Description |
|-----|-------------|
| `"loss"` | Total training loss |
| `"kl_loss"` | KL divergence (DreamerV3) |
| `"reconstruction_loss"` | Observation reconstruction |
| `"reward_loss"` | Reward prediction |
| `"continue_loss"` | Continuation prediction |

---

## Trajectory

Result of imagination rollout.

```python
trajectory = model.imagine(state, actions)

trajectory.states     # List[LatentState] - All states
trajectory.rewards    # Tensor[T, B, 1] - Predicted rewards
trajectory.continues  # Tensor[T, B, 1] - Continue probabilities
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `states` | `list[LatentState]` | Latent states at each step |
| `rewards` | `Tensor[T, B, 1]` | Predicted rewards |
| `continues` | `Tensor[T, B, 1]` or `None` | Episode continuation |

!!! note
    TD-MPC2 does not predict continuation, so `continues` may be `None`.

---

## save_pretrained / from_pretrained

Save and load models.

```python
# Save
model.save_pretrained("./my_model")

# Load
from worldloom import create_world_model
model = create_world_model("./my_model")
```

**Saved files**:

- `config.json` - Model configuration
- `model.pt` - Model weights
