# Training Pipeline Guide

## Callback Architecture

Defined in `src/worldflux/training/callbacks.py`. Base class `Callback` provides 6 hooks:

```python
class Callback(ABC):
    def on_train_begin(self, trainer: Trainer) -> None: ...
    def on_train_end(self, trainer: Trainer) -> None: ...
    def on_epoch_begin(self, trainer: Trainer) -> None: ...
    def on_epoch_end(self, trainer: Trainer) -> None: ...
    def on_step_begin(self, trainer: Trainer) -> None: ...
    def on_step_end(self, trainer: Trainer) -> None: ...
```

`CallbackList` dispatches to all registered callbacks in order.

### Built-in Callbacks

| Callback | Purpose | Key Config |
|----------|---------|------------|
| `LoggingCallback` | Console + optional wandb logging | `log_interval`, `use_wandb` |
| `CheckpointCallback` | Periodic + best-model saves | `save_interval`, `save_best`, `max_checkpoints` |
| `EarlyStoppingCallback` | Stop on loss plateau | `patience`, `min_delta`, `monitor` |
| `ProgressCallback` | tqdm progress bar | `desc` |
| `HeartbeatCallback` | WASR telemetry heartbeat | `interval_steps`, `scenario` |
| `DiagnosisCallback` | Detect training pathologies | `check_interval`, `gradient_min_norm`, `latent_std_min` |

`LoggingCallback` and `CheckpointCallback` are auto-registered by default in the Trainer.

## ReplayBuffer Single-Writer Constraint

**`ReplayBuffer` is NOT thread-safe.** Use a single writer thread only. This is documented in the project CLAUDE.md as a gotcha. Do not attempt concurrent writes from multiple threads or processes.

## Checkpoint Atomicity Pattern

`Trainer.save_checkpoint()` in `src/worldflux/training/trainer.py` uses atomic writes:

1. Write to a temp file (`*.pt.tmp`) in the same directory
2. Validate by loading the temp file with `torch.load(..., weights_only=True)`
3. Verify essential keys exist (`model_state_dict`, `optimizer_state_dict`, `global_step`)
4. Atomic rename via `os.replace(temp_path, final_path)` (POSIX atomic)
5. On failure, clean up the temp file

This prevents corrupted checkpoints from disk-full or process-kill scenarios.

### Checkpoint Contents

```python
checkpoint = {
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "global_step": ...,
    "best_loss": ...,
    "config": ...,           # TrainingConfig
    "scheduler_state_dict": ...,  # if scheduler exists
    "scaler_state_dict": ...,     # if mixed precision
    "model_config": ...,          # if model has config.to_dict()
}
```

## WASR Telemetry Integration

Located in `src/worldflux/telemetry/wasr.py`. Local JSONL-based telemetry (no network calls).

### Core API

```python
from worldflux.telemetry.wasr import write_event, read_events, make_run_id

write_event(
    event="heartbeat",     # or "diagnostic", custom strings
    scenario="trainer",
    success=True,
    duration_sec=120.0,
    ttfi_sec=0.5,
    artifacts={},
    error="",
    run_id=make_run_id(),
    # Optional fields:
    epoch=1, step=1000,
    throughput_steps_per_sec=45.2,
    flops_estimate=1e12,
    watts_estimate=300.0,
    flops_per_watt=3.3e9,
    suggestions=["Consider lower LR"],
)
```

### Storage

- Default path: `.worldflux/metrics.jsonl` in CWD
- Override via `WORLDFLUX_METRICS_PATH` env var
- Format: JSON Lines (one event per line, compact separators)

### Integration Points

- `HeartbeatCallback`: Emits periodic `heartbeat` events with throughput and efficiency metrics
- `DiagnosisCallback`: Emits `diagnostic` events when training pathologies are detected (NaN/Inf gradients, vanishing gradients, latent collapse)

## weights_only=True Rule

**Always use `torch.load(path, weights_only=True)` for model weights.** The only exception is `Trainer.load_checkpoint()` which requires `weights_only=False` to deserialize optimizer states -- this is marked with `# nosec B614` and a trust comment. Do not add new `weights_only=False` usage elsewhere.
