# Quick Start

Minimal execution path for WorldFlux.

## 1) Create a Model

```python
from worldflux import create_world_model

model = create_world_model(
    "dreamerv3:size12m",
    obs_shape=(3, 64, 64),
    action_dim=4,
)
```

## 2) Run Imagination

```python
import torch

obs = torch.randn(1, 3, 64, 64)
state = model.encode(obs)

actions = torch.randn(15, 1, 4)  # [horizon, batch, action_dim]
trajectory = model.rollout(state, actions)

print(trajectory.rewards.shape)    # [15, 1]
print(trajectory.continues.shape)  # [15, 1]
```

## 3) Save and Reload

```python
model.save_pretrained("./my_model")
reloaded = create_world_model("./my_model")
```

## 4) Next References

- [Factory API Guide](../api/factory.md)
- [Training API Guide](../api/training.md)
- [Protocol API Guide](../api/protocol.md)
