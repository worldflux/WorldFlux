# DreamerV3 vs TD-MPC2

Choosing the right world model architecture for your task.

## Quick Comparison

| Feature | DreamerV3 | TD-MPC2 |
|---------|-----------|---------|
| **Input Type** | Images or states | State vectors only |
| **Latent Space** | Categorical (discrete) | SimNorm (continuous) |
| **Architecture** | RSSM | Implicit MLP |
| **Decoder** | Yes | No |
| **Best For** | Atari, visual tasks | MuJoCo, robotics |
| **Planning** | Policy rollouts | MPC with Q-ensemble |

## When to Use DreamerV3

### Visual Observations (Images)

DreamerV3's CNN encoder handles image inputs:

```python
model = create_world_model(
    "dreamerv3:size50m",
    obs_shape=(3, 64, 64),  # RGB images
    action_dim=18,          # Atari action space
)
```

### Need Observation Reconstruction

DreamerV3 can decode latent states back to observations:

```python
state = model.encode(obs)
decoded = model.decode(state)
reconstructed_obs = decoded.preds["obs"]  # Same shape as input
```

### Discrete/Mixed Action Spaces

Works well with both discrete and continuous actions.

### Atari Games

```python
# Recommended preset for Atari
model = create_world_model(
    "dreamerv3:size50m",
    obs_shape=(3, 64, 64),
    action_dim=18,  # Atari action space
)
```

## When to Use TD-MPC2

### State Vector Observations

TD-MPC2 is optimized for low-dimensional state inputs:

```python
model = create_world_model(
    "tdmpc2:19m",
    obs_shape=(39,),  # HalfCheetah state
    action_dim=6,     # Joint torques
)
```

### Continuous Control Tasks

MuJoCo, robotics, and similar domains:

```python
# MuJoCo HalfCheetah
model = create_world_model(
    "tdmpc2:19m",
    obs_shape=(39,),
    action_dim=6,
)

# DMControl Walker
model = create_world_model(
    "tdmpc2:5m",
    obs_shape=(24,),
    action_dim=6,
)
```

### MPC-Style Planning

TD-MPC2 includes Q-function ensemble for planning:

```python
state = model.encode(obs)
action = torch.randn(1, 6)

# Get Q-values for planning
q_values = model.predict_q(state, action)
```

## Latent State Differences

### DreamerV3: Categorical Latent

```python
state = dreamer.encode(obs)
print(state.tensors["deter"].shape)  # [B, 2048] - GRU hidden
print(state.tensors["stoch"].shape)  # [B, 1024] - Categorical samples
```

The stochastic component captures uncertainty and enables exploration.

### TD-MPC2: SimNorm Latent

```python
state = tdmpc.encode(obs)
print(state.tensors["latent"].shape)  # [B, 512] - SimNorm embedding
```

SimNorm provides stable, continuous embeddings without explicit uncertainty.

## Model Size Guidelines

### DreamerV3 Presets

| Preset | Params | Use Case |
|--------|--------|----------|
| `size12m` | 12M | Quick experiments, testing |
| `size25m` | 25M | Simple environments |
| `size50m` | 50M | Standard Atari |
| `size100m` | 100M | Complex visual tasks |
| `size200m` | 200M | Maximum capacity |

### TD-MPC2 Presets

| Preset | Params | Use Case |
|--------|--------|----------|
| `5m` | 5M | Simple tasks, fast iteration |
| `19m` | 19M | Standard MuJoCo |
| `48m` | 48M | Complex control |
| `317m` | 317M | Large-scale multitask |

## Code Examples

### DreamerV3 for Atari

```python
from worldflux import create_world_model
from worldflux.training import train, ReplayBuffer

# Create model
model = create_world_model(
    "dreamerv3:size50m",
    obs_shape=(3, 64, 64),
    action_dim=18,
    device="cuda",
)

# Load Atari trajectories
buffer = ReplayBuffer.load("atari_breakout.npz")

# Train
trained = train(model, buffer, total_steps=200_000)

# Imagine
state = model.encode(obs)
trajectory = model.rollout(state, actions)
reconstructed = model.decode(trajectory.states[-1])
```

### TD-MPC2 for MuJoCo

```python
from worldflux import create_world_model
from worldflux.training import train, ReplayBuffer

# Create model
model = create_world_model(
    "tdmpc2:19m",
    obs_shape=(39,),
    action_dim=6,
    device="cuda",
)

# Load MuJoCo trajectories
buffer = ReplayBuffer.load("halfcheetah.npz")

# Train
trained = train(model, buffer, total_steps=1_000_000)

# Use for planning
state = model.encode(obs)
q_values = model.q(state, action)
```

## Migration Between Models

Both models share the same API:

```python
# Works with both DreamerV3 and TD-MPC2
def run_imagination(model, obs, actions):
    state = model.encode(obs)
    trajectory = model.rollout(state, actions)
    return trajectory.rewards

# Use with either model
rewards_dreamer = run_imagination(dreamer, obs, actions)
rewards_tdmpc = run_imagination(tdmpc, obs, actions)
```

## Summary

- **DreamerV3**: Visual tasks, Atari, when you need observation reconstruction
- **TD-MPC2**: State-based control, MuJoCo, robotics, MPC planning
