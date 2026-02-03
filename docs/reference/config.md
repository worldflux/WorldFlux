# Configuration Reference

Complete configuration options for WorldFlux.

## Model Configuration

### DreamerV3Config

```python
from worldflux.models.dreamer import DreamerV3Config

config = DreamerV3Config(
    # Required
    obs_shape=(3, 64, 64),
    action_dim=18,

    # Latent dimensions
    deter_dim=2048,
    stoch_dim=1024,
    num_categories=32,

    # Network architecture
    cnn_depth=48,
    mlp_layers=5,
    hidden_dim=640,

    # Training
    kl_balance=0.8,
    kl_free=1.0,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `obs_shape` | `tuple` | required | Observation shape |
| `action_dim` | `int` | required | Action dimension |
| `deter_dim` | `int` | varies | Deterministic state dimension |
| `stoch_dim` | `int` | varies | Stochastic state dimension |
| `num_categories` | `int` | 32 | Categories per latent dim |
| `cnn_depth` | `int` | 48 | CNN depth multiplier |
| `mlp_layers` | `int` | 5 | MLP hidden layers |
| `hidden_dim` | `int` | varies | Hidden layer dimension |
| `kl_balance` | `float` | 0.8 | KL balancing coefficient |
| `kl_free` | `float` | 1.0 | Free nats threshold |

### TDMPC2Config

```python
from worldflux.models.tdmpc2 import TDMPC2Config

config = TDMPC2Config(
    # Required
    obs_shape=(39,),
    action_dim=6,

    # Latent dimensions
    latent_dim=512,

    # Network architecture
    num_q=5,
    mlp_layers=2,
    hidden_dim=512,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `obs_shape` | `tuple` | required | Observation shape |
| `action_dim` | `int` | required | Action dimension |
| `latent_dim` | `int` | varies | Latent state dimension |
| `num_q` | `int` | 5 | Number of Q-networks |
| `mlp_layers` | `int` | 2 | MLP hidden layers |
| `hidden_dim` | `int` | varies | Hidden layer dimension |

### Using from_size

```python
# Standard presets
config = DreamerV3Config.from_size(
    "size50m",
    obs_shape=(3, 64, 64),
    action_dim=18,
)

config = TDMPC2Config.from_size(
    "19m",
    obs_shape=(39,),
    action_dim=6,
)
```

---

## Training Configuration

### TrainingConfig

```python
from worldflux.training import TrainingConfig

config = TrainingConfig(
    # Duration
    total_steps=100_000,

    # Batch settings
    batch_size=16,
    sequence_length=50,

    # Optimizer
    learning_rate=3e-4,
    weight_decay=0.0,
    grad_clip=100.0,
    warmup_steps=0,

    # Logging
    log_interval=1000,
    eval_interval=5000,
    save_interval=10000,

    # Device
    device="cuda",
    seed=42,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `total_steps` | `int` | 50000 | Total training steps |
| `batch_size` | `int` | 16 | Batch size |
| `sequence_length` | `int` | 50 | BPTT sequence length |
| `learning_rate` | `float` | 3e-4 | Adam learning rate |
| `weight_decay` | `float` | 0.0 | L2 regularization |
| `grad_clip` | `float` | 100.0 | Max gradient norm |
| `warmup_steps` | `int` | 0 | LR warmup steps |
| `log_interval` | `int` | 1000 | Logging frequency |
| `eval_interval` | `int` | 5000 | Evaluation frequency |
| `save_interval` | `int` | 10000 | Checkpoint frequency |
| `device` | `str` | "cuda" | Training device |
| `seed` | `int` | 42 | Random seed |

---

## Callback Configuration

### CheckpointCallback

```python
from worldflux.training.callbacks import CheckpointCallback

callback = CheckpointCallback(
    save_dir="./checkpoints",
    save_every=10000,
    keep_last=3,
    save_best=True,
    monitor="loss",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_dir` | `str` | required | Directory for checkpoints |
| `save_every` | `int` | 10000 | Steps between saves |
| `keep_last` | `int` | 3 | Number to keep |
| `save_best` | `bool` | True | Save best model |
| `monitor` | `str` | "loss" | Metric to monitor |

### LoggingCallback

```python
from worldflux.training.callbacks import LoggingCallback

callback = LoggingCallback(
    log_dir="./logs",
    log_to_console=True,
    log_to_tensorboard=True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_dir` | `str` | None | TensorBoard log directory |
| `log_to_console` | `bool` | True | Print to console |
| `log_to_tensorboard` | `bool` | False | Enable TensorBoard |

### EarlyStoppingCallback

```python
from worldflux.training.callbacks import EarlyStoppingCallback

callback = EarlyStoppingCallback(
    monitor="loss",
    patience=5000,
    min_delta=1e-4,
    mode="min",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `monitor` | `str` | "loss" | Metric to monitor |
| `patience` | `int` | 5000 | Steps without improvement |
| `min_delta` | `float` | 1e-4 | Minimum change |
| `mode` | `str` | "min" | "min" or "max" |

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `WORLDLOOM_CACHE_DIR` | Model cache directory |
| `CUDA_VISIBLE_DEVICES` | GPU selection |

---

## Configuration Files

### Saved Model (config.json)

```json
{
    "model_type": "dreamerv3",
    "obs_shape": [3, 64, 64],
    "action_dim": 18,
    "deter_dim": 4096,
    "stoch_dim": 1024,
    "num_categories": 32,
    "cnn_depth": 48,
    "mlp_layers": 5
}
```

### Loading Custom Config

```python
from worldflux.models.dreamer import DreamerV3Config, DreamerV3WorldModel

# From dict
config = DreamerV3Config(**config_dict)

# From JSON file
import json
with open("config.json") as f:
    config = DreamerV3Config(**json.load(f))

model = DreamerV3WorldModel(config)
```
