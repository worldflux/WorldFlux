# Factory Functions

The main entry points for creating and managing world models.

## create_world_model

Create a world model from a preset, alias, or saved path.

```python
from worldflux import create_world_model

model = create_world_model(
    model="dreamerv3:size12m",
    obs_shape=(3, 64, 64),
    action_dim=4,
    device="cuda",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Model preset, alias, or path to saved model |
| `obs_shape` | `tuple[int, ...]` | `None` | Observation shape (required for new models) |
| `action_dim` | `int` | `None` | Action dimension (required for new models) |
| `device` | `str` | `"cuda"` | Device to place model on |
| `**kwargs` | | | Additional config overrides |

### Model Specifiers

#### Presets (type:size)

```python
model = create_world_model("dreamerv3:size12m", ...)
model = create_world_model("tdmpc2:5m", ...)
```

#### Aliases

```python
# DreamerV3 aliases
model = create_world_model("dreamer", ...)       # size12m
model = create_world_model("dreamer-small", ...)  # size12m
model = create_world_model("dreamer-medium", ...) # size50m
model = create_world_model("dreamer-large", ...)  # size200m

# TD-MPC2 aliases
model = create_world_model("tdmpc", ...)         # 5m
model = create_world_model("tdmpc-small", ...)   # 5m
model = create_world_model("tdmpc-medium", ...)  # 48m
model = create_world_model("tdmpc-large", ...)   # 317m
```

#### Load from Path

```python
model = create_world_model("./my_saved_model")
model = create_world_model("/path/to/checkpoint")
```

### Config Overrides

```python
# Override default configuration
model = create_world_model(
    "dreamerv3:size12m",
    obs_shape=(3, 64, 64),
    action_dim=4,
    hidden_dim=256,  # Override default
    num_layers=3,    # Override default
)
```

### Returns

A world model instance implementing the [`WorldModel` protocol](protocol.md).

---

## list_models

List all available model presets.

```python
from worldflux import list_models

# Simple list
models = list_models()
# ['dreamerv3:size12m', 'dreamerv3:size25m', ..., 'tdmpc2:5m', ...]

# With descriptions
models = list_models(verbose=True)
# {
#     'dreamerv3:size12m': {
#         'params': '12M',
#         'type': 'dreamerv3',
#         'description': 'Small DreamerV3 model'
#     },
#     ...
# }
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `bool` | `False` | Return detailed info as dict |

### Returns

- `verbose=False`: `list[str]` of model names
- `verbose=True`: `dict[str, dict]` with model info

---

## Model Presets

### DreamerV3

| Preset | Parameters | deter_dim | stoch_dim |
|--------|------------|-----------|-----------|
| `dreamerv3:size12m` | ~12M | 2048 | 1024 |
| `dreamerv3:size25m` | ~25M | 2048 | 1024 |
| `dreamerv3:size50m` | ~50M | 4096 | 1024 |
| `dreamerv3:size100m` | ~100M | 4096 | 2048 |
| `dreamerv3:size200m` | ~200M | 8192 | 2048 |

### TD-MPC2

| Preset | Parameters | latent_dim | num_q |
|--------|------------|------------|-------|
| `tdmpc2:5m` | ~5M | 512 | 5 |
| `tdmpc2:19m` | ~19M | 512 | 5 |
| `tdmpc2:48m` | ~48M | 1024 | 5 |
| `tdmpc2:317m` | ~317M | 2048 | 10 |
