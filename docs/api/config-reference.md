---
sidebar_label: Config API
---

# Configuration Reference

Configuration classes for world models. All configs are Python dataclasses with
built-in validation, JSON serialization, and size-preset factories.

```python
from worldflux.core.config import WorldModelConfig, DreamerV3Config, TDMPC2Config
```

:::note
`TrainingConfig` is documented separately in the
[Training API Reference](./training-reference.md#trainingconfig).
:::

---

## WorldModelConfig

```python
@dataclass
class WorldModelConfig
```

Base configuration for all world models. Subclasses (`DreamerV3Config`,
`TDMPC2Config`) extend this with model-specific parameters.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model_type` | `str` | `"base"` | Identifier for the model type (`"dreamer"`, `"tdmpc2"`, etc.). |
| `model_name` | `str` | `"unnamed"` | Human-readable name or size preset name. |
| `obs_shape` | `tuple[int, ...]` | `(3, 64, 64)` | Shape of observations (e.g. `(3, 64, 64)` for images, `(39,)` for vectors). |
| `action_dim` | `int` | `6` | Dimension of the action space. |
| `action_type` | `str` | `"continuous"` | Type of actions: `"continuous"`, `"discrete"`, `"token"`, `"latent"`, `"text"`, or `"none"`. |
| `observation_modalities` | `dict[str, dict[str, Any]]` | `{}` | Multi-modal observation spec. Auto-populated from `obs_shape` if empty. |
| `action_spec` | `dict[str, Any]` | `{}` | Normalized action specification. Auto-populated from `action_type` and `action_dim`. |
| `latent_type` | `LatentType` | `DETERMINISTIC` | Type of latent space: `DETERMINISTIC`, `GAUSSIAN`, `CATEGORICAL`, `VQ`, or `SIMNORM`. |
| `latent_dim` | `int` | `256` | Dimension of the primary latent space. |
| `deter_dim` | `int` | `256` | Dimension of deterministic state (RSSM models). |
| `stoch_dim` | `int` | `32` | Dimension of stochastic state (RSSM models). |
| `dynamics_type` | `DynamicsType` | `MLP` | Dynamics model architecture: `RSSM`, `MLP`, `TRANSFORMER`, or `SSM`. |
| `hidden_dim` | `int` | `512` | Hidden dimension for MLPs and other layers. |
| `learning_rate` | `float` | `3e-4` | Default learning rate for training. |
| `grad_clip` | `float` | `100.0` | Gradient clipping threshold. |
| `device` | `str` | `"cuda"` | Target device (`"cuda"`, `"cpu"`, `"auto"`). |
| `dtype` | `str` | `"float32"` | Data type: `"float32"`, `"float16"`, or `"bfloat16"`. |

### Enums

#### LatentType

| Value | Description |
|-------|-------------|
| `DETERMINISTIC` | Simple deterministic latent space. |
| `GAUSSIAN` | Gaussian latent space with mean and variance. |
| `CATEGORICAL` | Categorical latent space (used by DreamerV3). |
| `VQ` | Vector-quantized latent space. |
| `SIMNORM` | SimNorm latent space (used by TD-MPC2). |

#### DynamicsType

| Value | Description |
|-------|-------------|
| `RSSM` | Recurrent State-Space Model (DreamerV3). |
| `MLP` | Simple MLP dynamics (TD-MPC2). |
| `TRANSFORMER` | Transformer-based dynamics. |
| `SSM` | State-Space Model dynamics. |

### Methods

#### to_dict / from_dict

```python
def to_dict(self) -> dict[str, Any]

@classmethod
def from_dict(cls, d: dict[str, Any]) -> WorldModelConfig
```

Convert to/from dictionary for serialization.

#### save / load

```python
def save(self, path: str | Path) -> None

@classmethod
def load(cls, path: str | Path) -> WorldModelConfig
```

Save/load configuration to/from JSON file.

### Example

```python
config = WorldModelConfig(obs_shape=(84, 84, 3), action_dim=4)
config.save("config.json")
loaded = WorldModelConfig.load("config.json")
```

---

## DreamerV3Config

```python
@dataclass
class DreamerV3Config(WorldModelConfig)
```

DreamerV3 world model configuration. Uses an RSSM (Recurrent State-Space Model)
with categorical latent variables.

### Size Presets

| Preset | Params | deter_dim | stoch | hidden_dim | cnn_depth |
|--------|--------|-----------|-------|------------|-----------|
| `ci` | ~0.1M | 64 | 4x4 | 32 | 8 |
| `size12m` | ~12M | 2048 | 16x16 | 256 | 48 |
| `size25m` | ~25M | 4096 | 32x16 | 512 | 48 |
| `size50m` | ~50M | 4096 | 32x32 | 640 | 48 |
| `size100m` | ~100M | 8192 | 32x32 | 768 | 48 |
| `size200m` | ~200M | 8192 | 32x32 | 1024 | 48 |
| `official_xl` | ~200-300M | 8192 | 32x64 | 1024 | 64 |

`ci` is for quick validation / scaffold workflows and is not the canonical proof-grade parity preset. Dreamer parity proof uses `official_xl`.

### DreamerV3-Specific Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `stoch_discrete` | `int` | `32` | Number of categorical distributions. |
| `stoch_classes` | `int` | `32` | Number of classes per categorical distribution. |
| `encoder_type` | `str` | `"cnn"` | Type of encoder: `"cnn"` or `"mlp"`. |
| `decoder_type` | `str` | `"cnn"` | Type of decoder: `"cnn"` or `"mlp"`. |
| `cnn_depth` | `int` | `48` | Base depth multiplier for CNN encoder/decoder. |
| `cnn_kernels` | `tuple[int, ...]` | `(4, 4, 4, 4)` | Kernel sizes for CNN layers. |
| `learning_rate` | `float` | `1e-4` | DreamerV3 paper uses 1e-4 for world model. |
| `grad_clip` | `float` | `1000.0` | DreamerV3 paper uses 1000 for grad clip. |
| `kl_free` | `float` | `1.0` | Free nats for KL divergence (prevents posterior collapse). |
| `loss_scales` | `dict[str, float]` | see below | Weights for each loss component. |
| `use_symlog` | `bool` | `True` | Whether to use symlog transformation for predictions. |
| `use_twohot` | `bool` | `True` | Whether to use twohot categorical reward prediction. |
| `reward_num_bins` | `int` | `255` | Number of bins for twohot reward prediction. |
| `reward_bin_min` | `float` | `-20.0` | Minimum value for reward bins (symlog space). |
| `reward_bin_max` | `float` | `20.0` | Maximum value for reward bins (symlog space). |

**Actor-Critic Fields** (gated by `actor_critic=True`):

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `actor_critic` | `bool` | `False` | Enable actor-critic training. |
| `imagination_horizon` | `int` | `15` | Imagination rollout horizon. |
| `actor_lr` | `float` | `3e-5` | Actor learning rate. |
| `critic_lr` | `float` | `3e-5` | Critic learning rate. |
| `gamma` | `float` | `0.997` | Discount factor. |
| `lambda_` | `float` | `0.95` | GAE lambda. |
| `slow_critic_fraction` | `float` | `0.02` | EMA fraction for slow critic target. |
| `actor_entropy_coef` | `float` | `3e-4` | Entropy regularization coefficient. |
| `return_normalization` | `bool` | `True` | Whether to normalize returns. |

### Default loss_scales

```python
{
    "reconstruction": 1.0,
    "kl_dynamics": 0.5,      # beta_dyn
    "kl_representation": 0.1, # beta_rep
    "reward": 1.0,
    "continue": 1.0,
}
```

### Methods

#### from_size

```python
@classmethod
def from_size(cls, size: str, **kwargs: Any) -> DreamerV3Config
```

Create configuration from a size preset.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | `str` | required | Size preset name: `"ci"`, `"size12m"`, `"size25m"`, `"size50m"`, `"size100m"`, `"size200m"`, `"official_xl"`. |
| `**kwargs` | `Any` | | Override any preset parameters. |

**Returns:** `DreamerV3Config` with the specified size preset.

**Raises:** `ValueError` if the size preset is not recognized.

### Examples

```python
# Create from size preset
config = DreamerV3Config.from_size("size12m")

# Create with custom parameters
config = DreamerV3Config(
    obs_shape=(3, 64, 64),
    action_dim=4,
    deter_dim=1024,
    stoch_discrete=16,
    stoch_classes=16,
)

# Override preset values
config = DreamerV3Config.from_size("size50m", action_dim=18, cnn_depth=64)
```

---

## TDMPC2Config

```python
@dataclass
class TDMPC2Config(WorldModelConfig)
```

TD-MPC2 world model configuration. TD-MPC2 is an implicit world model that uses
SimNorm latent space and learns value functions for planning. It is particularly
effective for continuous control tasks.

### Size Presets

| Preset | Params | latent_dim | hidden_dim |
|--------|--------|------------|------------|
| `ci` | ~0.1M | 32 | 32 |
| `5m` | ~5M | 256 | 256 |
| `proof_5m` | ~5M | 256 | 256 |
| `5m_legacy` | ~5M | 256 | 256 |
| `19m` | ~19M | 512 | 512 |
| `48m` | ~48M | 512 | 1024 |
| `317m` | ~317M | 1024 | 2048 |

`ci` is for quick validation / scaffold workflows and is not the canonical proof-grade parity preset. TD-MPC2 parity proof uses `proof_5m`. `5m` is the compatibility preset, and `5m_legacy` is the legacy compatibility preset.

### TD-MPC2-Specific Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `simnorm_dim` | `int` | `8` | Dimension for SimNorm grouping. `latent_dim` must be divisible by this value. |
| `num_hidden_layers` | `int` | `2` | Number of hidden layers in MLPs. |
| `task_dim` | `int` | `96` | Dimension of task embedding for multi-task learning. |
| `num_tasks` | `int` | `1` | Number of tasks for multi-task learning. |
| `num_q_networks` | `int` | `5` | Number of Q-networks in the ensemble. |
| `horizon` | `int` | `5` | Planning horizon for MPC. |
| `num_samples` | `int` | `512` | Number of action samples for CEM planning. |
| `num_elites` | `int` | `64` | Number of elite samples for CEM planning. Must not exceed `num_samples`. |
| `temperature` | `float` | `0.5` | Temperature for action sampling. Must be positive. |
| `momentum` | `float` | `0.1` | Momentum for CEM mean update. Must be in `[0, 1]`. |
| `gamma` | `float` | `0.99` | Discount factor. Must be in `(0, 1]`. |
| `use_decoder` | `bool` | `False` | Whether to use a decoder. TD-MPC2 is typically implicit (no decoder). |

### Methods

#### from_size

```python
@classmethod
def from_size(cls, size: str, **kwargs: Any) -> TDMPC2Config
```

Create configuration from a size preset.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | `str` | required | Size preset name: `"ci"`, `"5m"`, `"proof_5m"`, `"5m_legacy"`, `"19m"`, `"48m"`, `"317m"`. |
| `**kwargs` | `Any` | | Override any preset parameters. |

**Returns:** `TDMPC2Config` with the specified size preset.

**Raises:** `ValueError` if the size preset is not recognized.

### Examples

```python
# Create from size preset
config = TDMPC2Config.from_size("5m", obs_shape=(39,), action_dim=6)

# Create with custom parameters
config = TDMPC2Config(
    obs_shape=(39,),
    action_dim=6,
    latent_dim=256,
    hidden_dim=256,
)

# Large model with custom Q ensemble
config = TDMPC2Config.from_size("317m", num_q_networks=10, horizon=8)
```
