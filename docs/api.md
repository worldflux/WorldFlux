# API Reference

## Factory

### create_world_model

```python
from worldloom import create_world_model

model = create_world_model(
    model="dreamerv3:size12m",  # Preset, alias, or path
    obs_shape=(3, 64, 64),      # Observation shape
    action_dim=4,               # Action dimension
    device="cuda",              # Device
)
```

**Aliases**: `dreamer`, `dreamer-small/medium/large`, `tdmpc`, `tdmpc-small/medium/large`

### list_models

```python
from worldloom import list_models

list_models()  # ['dreamerv3:size12m', ...]
list_models(verbose=True)  # With descriptions
```

## WorldModel Protocol

All models implement:

```python
# Encode observation to latent
state = model.encode(obs)  # obs: [B, *obs_shape]

# Predict next state (no observation)
next_state = model.predict(state, action)  # action: [B, action_dim]

# Update with observation
next_state = model.observe(state, action, obs)

# Decode to predictions
preds = model.decode(state)  # {"obs", "reward", "continue"}

# Multi-step imagination
trajectory = model.imagine(initial_state, actions)  # actions: [T, B, action_dim]

# Training loss
losses = model.compute_loss(batch)  # {"loss", "kl_loss", "reconstruction_loss", ...}
```

## LatentState

```python
state = model.encode(obs)

state.deterministic  # [B, deter_dim] - History/context
state.stochastic     # [B, stoch_dim] - Uncertainty (DreamerV3 only)
state.features       # [B, total_dim] - Combined for decoders
```

## Trajectory

```python
trajectory = model.imagine(state, actions)

trajectory.states     # List of LatentState
trajectory.rewards    # [T, B, 1]
trajectory.continues  # [T, B, 1]
```

## Training

### TrainingConfig

```python
from worldloom.training import TrainingConfig

config = TrainingConfig(
    total_steps=50000,
    batch_size=16,
    sequence_length=50,
    learning_rate=3e-4,
    grad_clip=100.0,
)
```

### ReplayBuffer

```python
from worldloom.training import ReplayBuffer

buffer = ReplayBuffer(capacity=100000, obs_shape=(4,), action_dim=2)
buffer.add_episode(obs=..., actions=..., rewards=..., dones=...)
batch = buffer.sample(batch_size=16, seq_len=50, device="cuda")
```
