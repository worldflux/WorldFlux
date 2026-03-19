# Scaling

WorldFlux currently supports a staged scaling path rather than one monolithic
distributed runtime.

## Replay Backends

The training package now exposes replay backend abstractions:

- `ReplayBackend`: protocol for replay storage backends
- `MemoryReplayBackend`: adapter over the existing in-memory `ReplayBuffer`
- `ParquetReplayBackend`: parquet-backed reconstruction path for exported replay data

Example:

```python
from worldflux.training import MemoryReplayBackend, ReplayBuffer

buffer = ReplayBuffer(capacity=1024, obs_shape=(4,), action_dim=2)
backend = MemoryReplayBackend(buffer)
batch = backend.sample(batch_size=8, seq_len=16)
```

## Current Scope

- `MemoryReplayBackend` is the default local path.
- `ParquetReplayBackend` is the first step toward larger analytics-oriented data
  workflows.
- Native distributed training is still tracked separately.

## Near-Term Roadmap

- `TrainingConfig(distributed_mode="ddp", distributed_world_size=N)` now exposes
  the first normalized launch surface for native distributed training work.
- `build_launch_config(...)` returns the normalized launch description used by
  upcoming distributed trainer entry points.
- add distributed training entry points
- add throughput and memory profiling
- extend replay backends beyond in-memory reconstruction
