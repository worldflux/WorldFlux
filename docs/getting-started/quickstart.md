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

## 5) Comparing DreamerV3 and TD-MPC2

The two reference model families are designed for different observation spaces.
Here is how to create both and compare rollout outputs on the same task:

```python
import torch
from worldflux import create_world_model

# --- Image-based environment (e.g. Atari, DMControl vision) ---
obs_shape_img = (3, 64, 64)
action_dim = 4

dreamer = create_world_model(
    "dreamerv3:size12m",
    obs_shape=obs_shape_img,
    action_dim=action_dim,
    device="cpu",
)

# --- Vector-based environment (e.g. MuJoCo state, robotics) ---
obs_shape_vec = (39,)
action_dim_vec = 6

tdmpc = create_world_model(
    "tdmpc2:5m",
    obs_shape=obs_shape_vec,
    action_dim=action_dim_vec,
    device="cpu",
)

# Rollout with DreamerV3
obs_img = torch.randn(1, *obs_shape_img)
state_d = dreamer.encode(obs_img)
actions_d = torch.randn(10, 1, action_dim)
traj_d = dreamer.rollout(state_d, actions_d)

# Rollout with TD-MPC2
obs_vec = torch.randn(1, *obs_shape_vec)
state_t = tdmpc.encode(obs_vec)
actions_t = torch.randn(10, 1, action_dim_vec)
traj_t = tdmpc.rollout(state_t, actions_t)

print(f"DreamerV3  rewards shape: {traj_d.rewards.shape}")   # [10, 1]
print(f"TD-MPC2    rewards shape: {traj_t.rewards.shape}")   # [10, 1]
```

!!! tip "Which model should I pick?"
    Use **DreamerV3** for image/pixel observations with its RSSM latent dynamics.
    Use **TD-MPC2** for low-dimensional vector states where fast MLP planning shines.
    See [Unified Comparison](../reference/unified-comparison.md) for detailed benchmarks.

## 6) Configuration Customization

`create_world_model` accepts keyword arguments that override the preset defaults.
You can also inspect or modify a config object before model creation with `get_config`.

### Override via `create_world_model`

```python
from worldflux import create_world_model

# Override DreamerV3 architecture parameters
model = create_world_model(
    "dreamerv3:size12m",
    obs_shape=(3, 64, 64),
    action_dim=4,
    deter_dim=1024,          # smaller deterministic state
    hidden_dim=128,          # narrower MLPs
    kl_free=0.5,             # tighter KL constraint
    learning_rate=1e-4,      # lower learning rate
    device="cpu",
)

# Override TD-MPC2 planning parameters
model = create_world_model(
    "tdmpc2:5m",
    obs_shape=(39,),
    action_dim=6,
    horizon=10,              # longer planning horizon
    num_samples=1024,        # more CEM samples
    num_elites=128,          # more elite trajectories
    num_q_networks=3,        # smaller Q ensemble
    device="cpu",
)
```

### Inspect config before creation with `get_config`

```python
from worldflux import get_config

config = get_config("dreamerv3:size12m")
print(config.deter_dim)      # 2048
print(config.stoch_discrete)  # 16
print(config.hidden_dim)      # 256

config = get_config("tdmpc2:5m", obs_shape=(39,), action_dim=6)
print(config.latent_dim)   # 256
print(config.horizon)      # 5
print(config.num_samples)  # 512
```

### List all available models

```python
from worldflux import list_models

# Simple list of preset identifiers
print(list_models())
# ['dreamer:ci', 'dreamerv3:size12m', 'dreamerv3:size25m', ...]

# Detailed catalog with descriptions and parameter counts
catalog = list_models(verbose=True)
for name, info in catalog.items():
    print(f"{name:25s} {info['params']:>8s}  {info['description']}")

# Filter by maturity level
experimental = list_models(maturity="experimental")
print(experimental)
# ['jepa:base', 'vjepa2:ci', 'vjepa2:tiny', 'vjepa2:base', ...]
```

## 7) Batch Processing

WorldFlux models natively support batched observations.
Pass a batch dimension as the first axis of your tensors:

```python
import torch
from worldflux import create_world_model

model = create_world_model(
    "dreamerv3:size12m",
    obs_shape=(3, 64, 64),
    action_dim=4,
    device="cpu",
)

batch_size = 8

# Encode a batch of observations
obs_batch = torch.randn(batch_size, 3, 64, 64)
states = model.encode(obs_batch)
print(f"Batch size in state: {states.batch_size}")  # 8

# Rollout with batched actions: [horizon, batch, action_dim]
horizon = 15
actions = torch.randn(horizon, batch_size, 4)
trajectory = model.rollout(states, actions)

print(f"Rewards shape: {trajectory.rewards.shape}")    # [15, 8]
print(f"Continues shape: {trajectory.continues.shape}")  # [15, 8]
print(f"Actions shape: {trajectory.actions.shape}")     # [15, 8, 4]

# Access individual time-step states
for t in range(trajectory.horizon):
    step_state = trajectory.states[t + 1]  # states[0] is initial
    print(f"  Step {t}: batch_size={step_state.batch_size}")
```

!!! note "Tensor layout convention"
    Observations use **`[batch, ...]`** layout.
    Action sequences for rollout use **`[horizon, batch, action_dim]`** layout.
    Rewards and continues are returned as **`[horizon, batch]`**.

## 8) Error Handling

Common errors and how to fix them.

### Shape mismatch on encode

```python
import torch
from worldflux import create_world_model

model = create_world_model(
    "dreamerv3:size12m",
    obs_shape=(3, 64, 64),
    action_dim=4,
)

# Wrong: observation shape does not match obs_shape
try:
    bad_obs = torch.randn(1, 84, 84, 3)  # HWC instead of CHW
    model.encode(bad_obs)
except Exception as e:
    print(f"Error: {e}")
    # Fix: transpose to [batch, C, H, W]
    good_obs = torch.randn(1, 3, 64, 64)
    state = model.encode(good_obs)
```

### Action dimension mismatch in rollout

```python
# Wrong: action_dim=4 but passing 6-dim actions
try:
    state = model.encode(torch.randn(1, 3, 64, 64))
    bad_actions = torch.randn(10, 1, 6)  # 6 != expected 4
    model.rollout(state, bad_actions)
except Exception as e:
    print(f"Error: {e}")
    # Fix: match action_dim to model configuration
    good_actions = torch.randn(10, 1, 4)
    trajectory = model.rollout(state, good_actions)
```

### CNN encoder requires 3D obs_shape

```python
from worldflux import create_world_model

# Wrong: vector obs with CNN encoder (DreamerV3 defaults to CNN)
try:
    model = create_world_model(
        "dreamerv3:size12m",
        obs_shape=(39,),  # vector, but encoder_type defaults to "cnn"
        action_dim=6,
    )
except Exception as e:
    print(f"Error: {e}")
    # Fix: use encoder_type="mlp" for vector observations
    model = create_world_model(
        "dreamerv3:size12m",
        obs_shape=(39,),
        action_dim=6,
        encoder_type="mlp",
        decoder_type="mlp",
    )
```

!!! warning "DreamerV3 with vector observations"
    DreamerV3 defaults to CNN encoder/decoder, which requires `obs_shape` with 3
    dimensions (C, H, W). For vector observations, explicitly set
    `encoder_type="mlp"` and `decoder_type="mlp"`.

### Unknown model identifier

```python
from worldflux import create_world_model

try:
    model = create_world_model("nonexistent:model")
except Exception as e:
    print(f"Error: {e}")
    # Fix: check available models
    from worldflux import list_models
    print("Available:", list_models())
```

## 9) Next References

- [CPU Success Path](cpu-success.md)
- [Observation Shape and Action Dim](../reference/observation-action.md)
- [Factory API Guide](../api/factory.md)
- [Training API Guide](../api/training.md)
- [Protocol API Guide](../api/protocol.md)
