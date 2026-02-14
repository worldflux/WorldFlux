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

## 4) Choosing a Model in `worldflux init`

`worldflux init` now recommends a model and asks you to choose explicitly between:

- `dreamer:ci`
- `tdmpc2:ci`

How recommendation works:

- If your environment is image/spatial (`3,64,64` style), it recommends `dreamer:ci`
- If your environment is vector/state (`39` style), it recommends `tdmpc2:ci`

How to choose:

- Start with the recommended model unless you have a clear reason to switch
- Switch to `tdmpc2:ci` for compact vector tasks when you want a lighter baseline
- Switch to `dreamer:ci` for image-heavy tasks when latent imagination quality is the priority

If you are unsure about input/output dimensions, read:
[Observation Shape and Action Dim](../reference/observation-action.md).

## 5) Next References

- [CPU Success Path](cpu-success.md)
- [Observation Shape and Action Dim](../reference/observation-action.md)
- [Factory API Guide](../api/factory.md)
- [Training API Guide](../api/training.md)
- [Protocol API Guide](../api/protocol.md)
