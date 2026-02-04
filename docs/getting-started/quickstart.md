# Quick Start

Get started with WorldFlux in 5 minutes.

## Create a World Model

```python
from worldflux import create_world_model

# DreamerV3 for image observations (Atari, visual tasks)
model = create_world_model(
    "dreamerv3:size12m",
    obs_shape=(3, 64, 64),  # RGB images
    action_dim=4,
)

# TD-MPC2 for state observations (MuJoCo, robotics)
model = create_world_model(
    "tdmpc2:5m",
    obs_shape=(39,),  # State vector
    action_dim=6,
)
```

## Encode Observations

Convert observations to latent states:

```python
import torch

obs = torch.randn(1, 3, 64, 64)  # Single observation
state = model.encode(obs)

print(state.tensors.keys())  # Latent tensor keys
```

## Imagination Rollout

Predict future states without environment interaction:

```python
# Define action sequence
actions = torch.randn(15, 1, 4)  # [horizon, batch, action_dim]

# Run imagination
trajectory = model.rollout(state, actions)

print(trajectory.rewards.shape)     # [15, 1] - Predicted rewards
print(trajectory.continues.shape)   # [15, 1] - Continue probabilities
```

## Decode Predictions

Get observation/reward predictions from latent states:

```python
output = model.decode(state)
preds = output.preds

print(preds["obs"].shape)      # Reconstructed observation
print(preds["reward"].shape)   # Predicted reward
print(preds.get("continue"))   # Episode continuation probability
```

## Train a Model

```python
from worldflux import set_seed
from worldflux.training import train, ReplayBuffer

# Load your data
buffer = ReplayBuffer.load("trajectories.npz")

# Reproducibility
set_seed(42)

# Train (one-liner)
trained_model = train(model, buffer, total_steps=50_000)

# Save
trained_model.save_pretrained("./my_model")
```

## Load a Saved Model

```python
model = create_world_model("./my_model")
```

## Next Steps

- [Core Concepts](concepts.md) - Understand the architecture
- [Train Your First Model](../tutorials/train-first-model.md) - Full training tutorial
- [API Reference](../api/factory.md) - Detailed API documentation
