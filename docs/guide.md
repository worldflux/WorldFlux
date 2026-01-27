# Guide

## Core Concepts

A **world model** learns to predict how an environment evolves:

```
Observation → Encoder → LatentState → Dynamics → Predictions
```

Key operations:

- `encode(obs)` - Observation to latent state
- `predict(state, action)` - Predict next state (imagination)
- `observe(state, action, obs)` - Update with real observation
- `decode(state)` - Latent to predictions (obs, reward, continue)
- `imagine(state, actions)` - Multi-step rollout

## Training

### Simple

```python
from worldloom import create_world_model
from worldloom.training import train, ReplayBuffer

model = create_world_model("dreamerv3:size12m", obs_shape=(4,), action_dim=2)
buffer = ReplayBuffer.load("data.npz")

trained = train(model, buffer, total_steps=50000)
trained.save_pretrained("./my_model")
```

### Full Control

```python
from worldloom.training import Trainer, TrainingConfig

config = TrainingConfig(
    total_steps=100000,
    batch_size=16,
    sequence_length=50,
    learning_rate=3e-4,
)

trainer = Trainer(model, config)
trainer.train(buffer)
```

## Data Collection

```python
from worldloom.training import ReplayBuffer
import numpy as np

buffer = ReplayBuffer(capacity=100000, obs_shape=(4,), action_dim=2)

# Add episodes
buffer.add_episode(
    obs=np.array([...]),      # [T, *obs_shape]
    actions=np.array([...]),  # [T, action_dim]
    rewards=np.array([...]),  # [T]
    dones=np.array([...]),    # [T]
)

# Save/load
buffer.save("buffer.npz")
buffer = ReplayBuffer.load("buffer.npz")
```

## Save & Load

```python
# Save
model.save_pretrained("./my_model")

# Load
model = create_world_model("./my_model")
```
