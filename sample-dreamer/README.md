# my-world-model

Generated with `worldflux init`.

## Quick Start

```bash
# Preferred launcher:
# 1) uv run python
# 2) python
# 3) python3
# 4) py
uv run python train.py
```

When training starts, a local dashboard URL is printed:

```text
Dashboard: http://127.0.0.1:8765
```

If port `8765` is already in use, it automatically falls back to the next available port.

Run inference or imagination checks:

```bash
uv run python inference.py
```

## Project Files

- `worldflux.toml`: Main config (model, architecture, training, data).
- `train.py`: Training entrypoint using `create_world_model` + `Trainer`.
- `dataset.py`: Demo data provider (gym collection with random fallback).
- `inference.py`: Short rollout and prediction stats.

## Configuration

`worldflux.toml` drives this project. Common changes:

- `training.batch_size`
- `training.total_steps`
- `training.learning_rate`
- `data.source` (`"random"` or `"gym"`)
- `data.gym_env`
- `gameplay.enabled`
- `gameplay.fps`
- `gameplay.max_frames`
- `online_collection.enabled`
- `online_collection.warmup_transitions`
- `online_collection.collect_steps_per_update`
- `online_collection.max_episode_steps`
- `visualization.enabled`
- `visualization.port`
- `visualization.refresh_ms`
- `visualization.open_browser`

## Gym Data Collection (Default)

This sample uses `data.source = "gym"` and `online_collection.enabled = true` by default so live gameplay can keep updating during training.
Install gym dependencies:

```bash
uv pip install "gymnasium>=0.29.0,<2.0.0"
```

For Atari, also install:

```bash
uv pip install "gymnasium[atari]>=0.29.0,<2.0.0" "ale-py>=0.8.0,<1.0.0"
```

If Atari dependencies are missing, training falls back to random replay data and the gameplay panel shows an unavailable message.

For MuJoCo, also install:

```bash
uv pip install "gymnasium[mujoco]>=0.29.0,<2.0.0" "mujoco>=3.0.0,<4.0.0"
```
