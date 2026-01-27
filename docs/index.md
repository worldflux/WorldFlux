# WorldLoom

Unified Python interface for latent world models (DreamerV3, TD-MPC2).

## Installation

```bash
git clone https://github.com/yoshihyoda/worldloom.git
cd worldloom
pip install -e ".[training]"
```

## Quick Start

```python
from worldloom import create_world_model
import torch

# Create model
model = create_world_model("dreamerv3:size12m", obs_shape=(3, 64, 64), action_dim=4)

# Encode observation
obs = torch.randn(1, 3, 64, 64)
state = model.encode(obs)

# Imagine future (15 steps)
actions = torch.randn(15, 1, 4)
trajectory = model.imagine(state, actions)

print(trajectory.rewards.shape)  # [15, 1, 1]
```

## Available Models

| Model | Preset | Best For |
|-------|--------|----------|
| DreamerV3 | `dreamerv3:size12m/25m/50m/100m/200m` | Images, Atari |
| TD-MPC2 | `tdmpc2:5m/19m/48m/317m` | State vectors, MuJoCo |

## Next

- [Guide](guide.md) - Core concepts and training
- [API](api.md) - Complete API reference
- [Models](models.md) - Model architecture details
