---
sidebar_label: Training API
---

# Training API Reference

Training utilities for WorldFlux world models, including the `Trainer` class,
one-liner `train` function, replay buffer, and training callbacks.

`Trainer` has two modes:

- local mode for native Torch models
- delegated mode for `OfficialBackendHandle` jobs backed by external runtimes

Stable distributed entry points are intentionally narrow in v0.1.x:

- `DDPTrainer` is the supported distributed trainer surface
- planned FSDP support is not exported from `worldflux.training`

```python
from worldflux.training import Trainer, TrainingConfig, ReplayBuffer, train
from worldflux.training.callbacks import (
    Callback, LoggingCallback, CheckpointCallback, EarlyStoppingCallback,
)
```

---

## train

```python
def train(
    model: nn.Module,
    data: BatchProvider | Any,
    total_steps: int | None = None,
    batch_size: int = 16,
    sequence_length: int = 50,
    learning_rate: float = 3e-4,
    grad_clip: float = 100.0,
    output_dir: str = "./outputs",
    device: str = "auto",
    **kwargs: Any,
) -> nn.Module
```

One-liner training function for quick experimentation.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | required | World model to train (must implement `loss(batch)`). |
| `data` | `BatchProvider \| Any` | required | `BatchProvider` or iterable yielding `Batch`/`dict`. |
| `total_steps` | `int \| None` | `None` | Number of training steps. Defaults to `100_000` if `None`. |
| `batch_size` | `int` | `16` | Batch size. |
| `sequence_length` | `int` | `50` | Sequence length for trajectory sampling. |
| `learning_rate` | `float` | `3e-4` | Learning rate. |
| `grad_clip` | `float` | `100.0` | Gradient clipping value. |
| `output_dir` | `str` | `"./outputs"` | Directory for outputs. |
| `device` | `str` | `"auto"` | Device to train on (`"cuda"`, `"cpu"`, or `"auto"`). |
| `**kwargs` | `Any` | | Additional `TrainingConfig` options. |

### Returns

`nn.Module` -- The trained model.

### Example

```python
from worldflux import create_world_model
from worldflux.training import train, ReplayBuffer

model = create_world_model("dreamerv3:size12m")
buffer = ReplayBuffer.load("data.npz")
trained_model = train(model, buffer, total_steps=50_000)
```

---

## Trainer

```python
class Trainer
```

HuggingFace-style trainer for WorldFlux. Provides automatic device placement,
gradient clipping, checkpointing, logging (console and optional wandb), learning
rate scheduling, gradient accumulation, and mixed-precision training.

### Constructor

```python
def __init__(
    self,
    model: WorldModel | nn.Module | OfficialBackendHandle,
    config: TrainingConfig | None = None,
    callbacks: list[Callback] | None = None,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `WorldModel \| nn.Module \| OfficialBackendHandle` | required | Native model for local training, or delegated backend handle for external execution. |
| `config` | `TrainingConfig \| None` | `None` | Training configuration. Uses defaults if `None`. |
| `callbacks` | `list[Callback] \| None` | `None` | Additional callbacks for logging/checkpointing. `LoggingCallback` and `CheckpointCallback` are always included by default. |
| `optimizer` | `Optimizer \| None` | `None` | Custom optimizer. Created from config if `None`. |
| `scheduler` | `LRScheduler \| None` | `None` | Custom learning rate scheduler. Created from config if `None`. |

When `model` is an `OfficialBackendHandle`, `Trainer` enters delegated mode and
routes execution through `submit()`. Local loop operations such as `train()`
remain available only for `backend="native_torch"`.

### Methods

#### train

```python
def train(
    self,
    data: BatchProvider | Any,
    num_steps: int | None = None,
    resume_from: str | None = None,
) -> nn.Module
```

Train the model.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `BatchProvider \| Any` | required | `BatchProvider` or iterable yielding `Batch`/`dict`. |
| `num_steps` | `int \| None` | `None` | Number of steps to train. Uses `config.total_steps` if `None`. |
| `resume_from` | `str \| None` | `None` | Path to checkpoint to resume from. |

**Returns:** `nn.Module` -- The trained model.

#### submit / status / logs / cancel

```python
def submit(self, *, resume_from: str | None = None) -> JobHandle
def status(self, handle: JobHandle) -> JobStatus
def logs(self, handle: JobHandle) -> Iterator[str]
def cancel(self, handle: JobHandle) -> None
```

Delegated execution helpers for non-native backends. `submit()` is only valid
when the trainer was constructed from an `OfficialBackendHandle`.

#### evaluate

```python
def evaluate(
    self,
    data: BatchProvider | Any,
    num_batches: int = 10,
) -> dict[str, float]
```

Evaluate the model on data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `BatchProvider \| Any` | required | Data source for evaluation. |
| `num_batches` | `int` | `10` | Number of batches to evaluate. |

**Returns:** `dict[str, float]` -- Dictionary of average metrics.

#### save_checkpoint / load_checkpoint

```python
def save_checkpoint(self, path: str) -> None
def load_checkpoint(self, path: str) -> None
```

Save/load training checkpoint. Save uses atomic write pattern (write to temp
file, validate, then rename) to prevent corrupted checkpoints. Load raises
`CheckpointError` if file is missing or corrupted.

#### add_callback

```python
def add_callback(self, callback: Callback) -> None
```

Register a callback after trainer construction.

#### runtime_profile

```python
def runtime_profile(self) -> dict[str, float | None]
```

Return lightweight runtime profiling metrics for DX instrumentation. Keys:
`"ttfi_sec"`, `"elapsed_sec"`, `"throughput_steps_per_sec"`.

### Example

```python
from worldflux import create_world_model
from worldflux.training import Trainer, TrainingConfig, ReplayBuffer

model = create_world_model("dreamerv3:size12m")
buffer = ReplayBuffer.load("data.npz")
config = TrainingConfig(total_steps=50_000, batch_size=32)

trainer = Trainer(model, config)
trainer.train(buffer)
```

---

## TrainingConfig

```python
@dataclass
class TrainingConfig
```

Configuration for training world models.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `total_steps` | `int` | `100_000` | Total number of training steps. |
| `batch_size` | `int` | `16` | Batch size for training. |
| `sequence_length` | `int` | `50` | Sequence length for trajectory sampling. |
| `instant_mode` | `bool` | `False` | Enable instant mode for fast testing. |
| `instant_total_steps` | `int` | `8` | Total steps when instant mode is active. |
| `instant_batch_size` | `int` | `8` | Batch size when instant mode is active. |
| `instant_sequence_length` | `int` | `10` | Sequence length when instant mode is active. |
| `learning_rate` | `float` | `3e-4` | Learning rate for optimizer. |
| `grad_clip` | `float` | `100.0` | Maximum gradient norm for clipping. |
| `weight_decay` | `float` | `0.0` | Weight decay for optimizer. |
| `warmup_steps` | `int` | `0` | Number of warmup steps for learning rate scheduler. |
| `log_interval` | `int` | `100` | Interval (in steps) for logging metrics. |
| `eval_interval` | `int` | `1000` | Interval (in steps) for evaluation. |
| `save_interval` | `int` | `10000` | Interval (in steps) for saving checkpoints. |
| `output_dir` | `str` | `"./outputs"` | Directory for saving outputs (checkpoints, logs). |
| `device` | `str` | `"auto"` | Device to train on (`"cuda"`, `"cpu"`, `"auto"`). |
| `seed` | `int` | `42` | Random seed for reproducibility. |
| `mixed_precision` | `bool` | `False` | Whether to use mixed precision training via `torch.amp`. |
| `num_workers` | `int` | `0` | Number of workers for data loading. |
| `prefetch_factor` | `int` | `2` | Number of batches to prefetch per worker. |
| `optimizer` | `str` | `"adamw"` | Optimizer type: `"adamw"`, `"adam"`, or `"sgd"`. |
| `scheduler` | `str` | `"none"` | LR scheduler: `"none"`, `"linear"`, `"cosine"`, or `"constant"`. |
| `gradient_accumulation_steps` | `int` | `1` | Number of gradient accumulation steps. Effective batch size becomes `batch_size * gradient_accumulation_steps`. |
| `auto_quality_check` | `bool` | `True` | Run a smoke-level quality check after training completes. |

### Advanced/Internal placeholders

The following config knobs exist in the broader type surface, but they are not
part of the supported MVP training path:

- `ema_decay`: reserved placeholder, rejected by `Trainer`
- `model_overrides`: reserved trainer-level patch surface, currently unsupported

### Methods

#### save / load

```python
def save(self, path: str | Path) -> None
def load(cls, path: str | Path) -> TrainingConfig
```

Save/load config to/from JSON file.

#### with_updates

```python
def with_updates(self, **kwargs: Any) -> TrainingConfig
```

Return a new config with updated values.

#### resolve_device

```python
def resolve_device(self) -> str
```

Resolve `"auto"` device to actual device (`"cuda"` or `"cpu"`).

#### effective_total_steps / effective_batch_size / effective_sequence_length

```python
def effective_total_steps(self) -> int
def effective_batch_size(self) -> int
def effective_sequence_length(self) -> int
```

Return the effective value under the current mode (normal or instant).

### Example

```python
config = TrainingConfig(total_steps=100_000, batch_size=32)
config.save("training_config.json")
loaded = TrainingConfig.load("training_config.json")
```

---

## ReplayBuffer

```python
class ReplayBuffer
```

Efficient trajectory storage for world model training. Stores episodes as
contiguous arrays and supports efficient random sampling of trajectory segments.

:::warning
**Not thread-safe.** Concurrent calls to `add_episode()` from multiple threads
may cause race conditions. Use a single writer thread or external synchronization.
:::

### Constructor

```python
def __init__(
    self,
    capacity: int,
    obs_shape: tuple[int, ...],
    action_dim: int,
    obs_dtype: type = np.float32,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capacity` | `int` | required | Maximum number of transitions to store. |
| `obs_shape` | `tuple[int, ...]` | required | Shape of observations (e.g. `(3, 64, 64)` for images). |
| `action_dim` | `int` | required | Dimension of action space. |
| `obs_dtype` | `type` | `np.float32` | NumPy dtype for observations. |

### Methods

#### add_episode

```python
def add_episode(
    self,
    obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray | None = None,
) -> None
```

Add a complete episode to the buffer.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `obs` | `np.ndarray` | required | Observations of shape `[episode_len, *obs_shape]`. |
| `actions` | `np.ndarray` | required | Actions of shape `[episode_len, action_dim]`. |
| `rewards` | `np.ndarray` | required | Rewards of shape `[episode_len]`. |
| `dones` | `np.ndarray \| None` | `None` | Done flags of shape `[episode_len]`. If `None`, last step is marked done. |

#### sample

```python
def sample(
    self,
    batch_size: int,
    seq_len: int,
    device: str | torch.device = "cpu",
) -> Batch
```

Sample random trajectory segments.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batch_size` | `int` | required | Number of trajectory segments. |
| `seq_len` | `int` | required | Length of each trajectory segment. |
| `device` | `str \| torch.device` | `"cpu"` | Device to place tensors on. |

**Returns:** `Batch` with keys `obs` `[B, T, *obs_shape]`, `actions` `[B, T, action_dim]`, `rewards` `[B, T]`, `terminations` `[B, T]`.

#### save / load

```python
def save(self, path: str | Path) -> None

@classmethod
def load(cls, path: str | Path) -> ReplayBuffer
```

Save/load buffer to/from `.npz` format with schema validation.

#### to_parquet / from_parquet

```python
def to_parquet(self, path: str | Path, *, compression: str = "zstd") -> None

@classmethod
def from_parquet(cls, path: str | Path) -> ReplayBuffer
```

Save/load buffer to/from Parquet format for cloud-native analytics pipelines.
Requires optional `pyarrow` dependency.

#### from_trajectories

```python
@classmethod
def from_trajectories(
    cls,
    trajectories: list[dict[str, np.ndarray]],
    capacity: int | None = None,
) -> ReplayBuffer
```

Create buffer from a list of trajectory dictionaries with keys `"obs"`, `"actions"`, `"rewards"`, `"dones"`.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_episodes` | `int` | Number of complete episodes stored. |

### Example

```python
buffer = ReplayBuffer(capacity=100_000, obs_shape=(3, 64, 64), action_dim=6)
buffer.add_episode(obs, actions, rewards, dones)
batch = buffer.sample(batch_size=32, seq_len=50)
```

---

## Callbacks

### Callback (Base Class)

```python
class Callback(ABC)
```

Base class for training callbacks. Override any of the following hooks:

| Hook | When Called |
|------|------------|
| `on_train_begin(trainer)` | At the start of training. |
| `on_train_end(trainer)` | At the end of training. |
| `on_epoch_begin(trainer)` | At the start of each epoch. |
| `on_epoch_end(trainer)` | At the end of each epoch. |
| `on_step_begin(trainer)` | Before each training step. |
| `on_step_end(trainer)` | After each training step. |

---

### LoggingCallback

```python
class LoggingCallback(Callback)
```

Callback for logging training metrics to console and optionally to wandb.

```python
def __init__(
    self,
    log_interval: int = 100,
    use_wandb: bool = False,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_interval` | `int` | `100` | Steps between log outputs. |
| `use_wandb` | `bool` | `False` | Whether to log to wandb. |
| `wandb_project` | `str \| None` | `None` | wandb project name. |
| `wandb_run_name` | `str \| None` | `None` | wandb run name. |

---

### CheckpointCallback

```python
class CheckpointCallback(Callback)
```

Callback for saving model checkpoints at regular intervals.

```python
def __init__(
    self,
    save_interval: int = 10000,
    output_dir: str = "./outputs",
    save_best: bool = True,
    max_checkpoints: int | None = 5,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_interval` | `int` | `10000` | Steps between checkpoint saves. |
| `output_dir` | `str` | `"./outputs"` | Directory to save checkpoints. |
| `save_best` | `bool` | `True` | Whether to save the best model (lowest loss). |
| `max_checkpoints` | `int \| None` | `5` | Maximum number of checkpoints to keep. `None` for unlimited. |

---

### EarlyStoppingCallback

```python
class EarlyStoppingCallback(Callback)
```

Callback for early stopping based on loss plateau detection.

```python
def __init__(
    self,
    patience: int = 5000,
    min_delta: float = 1e-4,
    monitor: str = "loss",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `patience` | `int` | `5000` | Number of steps to wait without improvement before stopping. |
| `min_delta` | `float` | `1e-4` | Minimum improvement required to reset patience counter. |
| `monitor` | `str` | `"loss"` | Metric name to monitor. |
