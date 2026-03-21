---
sidebar_label: Factory API
---

# Factory API Reference

The factory module provides a LangChain/HuggingFace-style interface for creating
and switching between different world model implementations.

```python
from worldflux import create_world_model, list_models, get_model_info, get_config
```

---

## create_world_model

```python
def create_world_model(
    model: str,
    *,
    obs_shape: tuple[int, ...] | None = None,
    action_dim: int | None = None,
    observation_modalities: dict[str, dict[str, Any]] | None = None,
    action_spec: dict[str, Any] | None = None,
    component_overrides: dict[str, object] | None = None,
    device: str = "cpu",
    api_version: str = "v3",
    **kwargs: Any,
) -> WorldModel
```

Create a world model with a simple, unified interface. This is the recommended
way to create world models.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Model identifier. Can be a full preset (`"dreamerv3:size12m"`, `"tdmpc2:5m"`), an alias (`"dreamer"`, `"tdmpc"`, `"dreamer-large"`), or a local path (`"./my_trained_model"`). |
| `obs_shape` | `tuple[int, ...] \| None` | `None` | Optional observation-shape override for the selected preset/config. |
| `action_dim` | `int \| None` | `None` | Optional action-dimension override (defaults to config value, typically `6`). |
| `observation_modalities` | `dict[str, dict[str, Any]] \| None` | `None` | Optional dict describing multi-modal observation inputs. Keys are modality names and values are dicts with `"kind"` and `"shape"` entries, e.g. `{"image": {"kind": "image", "shape": (3, 64, 64)}}`. |
| `action_spec` | `dict[str, Any] \| None` | `None` | Optional dict overriding the default action specification. Recognized keys include `"kind"` (`"continuous"`, `"discrete"`, etc.), `"dim"`, and `"num_actions"`. |
| `component_overrides` | `dict[str, object] \| None` | `None` | Optional component-slot overrides. Values may be a registered component id (`str`), a component class, or a pre-built component instance. |
| `device` | `str` | `"cpu"` | Device to place model on. |
| `api_version` | `str` | `"v3"` | API version. `"v3"` is current; `"v0.2"` enables deprecated legacy compatibility adapters. |
| `**kwargs` | `Any` | | Additional model-specific configuration parameters. |

### Returns

- `backend="native_torch"`: configured local `WorldModel` instance, placed on the specified device
- `backend!="native_torch"`: `OfficialBackendHandle` for delegated execution

The public signature remains `-> WorldModel` for compatibility, but delegated
backend requests intentionally return a handle object instead of constructing a
local model instance.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | If `api_version` is not `"v0.2"` or `"v3"`, or if invalid action kind is used with v3. |

### Examples

```python
# Basic usage
model = create_world_model("dreamerv3:size12m")

# With custom observation space
model = create_world_model(
    "tdmpc2:5m",
    obs_shape=(39,),
    action_dim=4,
)

# Using aliases
model = create_world_model("dreamer-large")  # dreamerv3:size200m

# Load trained model
model = create_world_model("./checkpoints/my_model")

# Multi-modal observations
model = create_world_model(
    "dreamerv3:size12m",
    observation_modalities={
        "image": {"kind": "image", "shape": (3, 64, 64)},
    },
)

# Delegated backend handle for proof/runtime workflows
handle = create_world_model(
    "dreamerv3:official_xl",
    backend="official_dreamerv3_jax_subprocess",
    device="cuda",
)
```

### Model Aliases

The following aliases are available for convenience:

| Alias | Resolves To |
|-------|-------------|
| `"dreamer"` | `"dreamerv3:size12m"` |
| `"dreamer-ci"` | `"dreamer:ci"` |
| `"dreamerv3"` | `"dreamerv3:size12m"` |
| `"dreamer-small"` | `"dreamerv3:size12m"` |
| `"dreamer-medium"` | `"dreamerv3:size50m"` |
| `"dreamer-large"` | `"dreamerv3:size200m"` |
| `"dreamerv3:official"` | `"dreamerv3:official_xl"` |
| `"tdmpc"` | `"tdmpc2:5m"` |
| `"tdmpc2-ci"` | `"tdmpc2:ci"` |
| `"tdmpc2"` | `"tdmpc2:5m"` |
| `"tdmpc-small"` | `"tdmpc2:5m"` |
| `"tdmpc-proof"` | `"tdmpc2:proof_5m"` |
| `"tdmpc2:proof"` | `"tdmpc2:proof_5m"` |
| `"tdmpc-legacy"` | `"tdmpc2:5m_legacy"` |
| `"tdmpc-medium"` | `"tdmpc2:48m"` |
| `"tdmpc-large"` | `"tdmpc2:317m"` |
| `"jepa"` | `"jepa:base"` |
| `"vjepa2"` | `"vjepa2:base"` |
| `"v-jepa2"` | `"vjepa2:base"` |
| `"token"` | `"token:base"` |
| `"diffusion"` | `"diffusion:base"` |
| `"dit"` | `"dit:base"` |
| `"ssm"` | `"ssm:base"` |
| `"renderer3d"` | `"renderer3d:base"` |
| `"physics"` | `"physics:base"` |
| `"gan"` | `"gan:base"` |

In the current implementation, `dreamer` family aliases resolve to DreamerV3 presets.

---

## list_models

```python
def list_models(
    verbose: bool = False,
    maturity: str | None = None,
    surface: str = "supported",
) -> list[str] | dict[str, dict[str, Any]]
```

List all available world model presets.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `bool` | `False` | If `True`, return detailed model information as a dict instead of a plain list. |
| `maturity` | `str \| None` | `None` | Optional maturity filter. One of `"reference"`, `"experimental"`, or `"skeleton"`. |
| `surface` | `str` | `"supported"` | Optional support-surface filter. One of `"supported"`, `"public"`, or `"all"`. |

### Returns

- `list[str]` -- List of model preset names when `verbose=False`.
- `dict[str, dict[str, Any]]` -- Dict mapping model names to detailed info when `verbose=True`.

### Examples

```python
# Simple list
list_models()
# ['dreamer:ci', 'dreamerv3:size12m', ..., 'tdmpc2:317m']

# With details
list_models(verbose=True)
# {
#     'dreamerv3:size12m': {
#         'description': 'DreamerV3 12M params - Good for simple environments',
#         'params': '~12M',
#         'type': 'dreamer',
#         'default_obs': 'image',
#         'maturity': 'reference',
#     },
#     ...
# }

# Include advanced proof-oriented presets
list_models(surface="public")

# Filter to reference models only
list_models(maturity="reference", surface="all")
```

---

## get_model_info

```python
def get_model_info(model: str) -> dict[str, Any]
```

Get detailed information about a specific model.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Model identifier or alias. |

### Returns

`dict[str, Any]` -- Dictionary containing model information including `description`, `params`, `type`, `default_obs`, `maturity`, `model_id`, and optionally `alias`.

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | If the model is not found in the catalog. |

### Examples

```python
info = get_model_info("dreamerv3:size12m")
# {
#     'description': 'DreamerV3 12M params - Good for simple environments',
#     'params': '~12M',
#     'type': 'dreamer',
#     'default_obs': 'image',
#     'maturity': 'reference',
#     'model_id': 'dreamerv3:size12m',
# }

# Using an alias
info = get_model_info("dreamer")
# includes 'alias': 'dreamer'
```

---

## get_config

```python
def get_config(
    model: str,
    *,
    obs_shape: tuple[int, ...] | None = None,
    action_dim: int | None = None,
    **kwargs: Any,
) -> WorldModelConfig
```

Get a configuration object without creating the model. Useful for inspecting
or modifying configuration before model creation.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Model identifier or alias. |
| `obs_shape` | `tuple[int, ...] \| None` | `None` | Override observation shape. |
| `action_dim` | `int \| None` | `None` | Override action dimension. |
| `**kwargs` | `Any` | | Additional configuration overrides. |

### Returns

`WorldModelConfig` -- Configuration object. The concrete type depends on the model family (e.g. `DreamerV3Config`, `TDMPC2Config`).

### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | If the model format is invalid (expected `"type:size"` format). |

### Examples

```python
# Get config and inspect
config = get_config("dreamerv3:size12m")
print(config.deter_dim)  # 2048

# Modify and create
config = get_config("tdmpc2:5m", obs_shape=(100,))
config.num_q_networks = 10  # Custom Q ensemble size
```
