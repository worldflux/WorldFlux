# Examples

Choose the right example for your use case.

## Getting Started

| Example | Description | Requirements |
|---------|-------------|-------------|
| [`quickstart_cpu_success.py`](quickstart_cpu_success.py) | Official CPU-first success path with machine-checkable criteria | CPU only |
| [`worldflux_quickstart.ipynb`](worldflux_quickstart.ipynb) | Interactive Colab notebook | CPU or GPU |

## Training

| Example | Description | Requirements |
|---------|-------------|-------------|
| [`train_dreamer.py`](train_dreamer.py) | DreamerV3 training basics | CPU or GPU |
| [`train_tdmpc2.py`](train_tdmpc2.py) | TD-MPC2 training basics | CPU or GPU |
| [`compare_unified_training.py`](compare_unified_training.py) | Side-by-side model comparison via unified API | CPU or GPU |
| [`train_dreamer_mujoco.py`](train_dreamer_mujoco.py) | DreamerV3 on MuJoCo environments | GPU recommended, `mujoco` extra |
| [`train_tdmpc2_mujoco.py`](train_tdmpc2_mujoco.py) | TD-MPC2 on MuJoCo environments | GPU recommended, `mujoco` extra |
| [`train_atari_dreamer.py`](train_atari_dreamer.py) | DreamerV3 on Atari environments | GPU recommended, `atari` extra |

## Advanced Models

| Example | Description | Requirements |
|---------|-------------|-------------|
| [`train_diffusion_model.py`](train_diffusion_model.py) | Diffusion-based world model training | CPU or GPU |
| [`train_jepa.py`](train_jepa.py) | JEPA world model training | CPU or GPU |
| [`train_token_model.py`](train_token_model.py) | Token-based world model training | CPU or GPU |

## Planning and Visualization

| Example | Description | Requirements |
|---------|-------------|-------------|
| [`plan_cem.py`](plan_cem.py) | Cross-entropy method planning with a trained model | CPU or GPU |
| [`visualize_imagination.py`](visualize_imagination.py) | Visualize rollouts, reward predictions, and latent dynamics | CPU or GPU, `viz` extra |
| [`validate_world_models_v23.py`](validate_world_models_v23.py) | V2/V3 API validation suite | CPU or GPU |

## Data Collection

| Example | Description | Requirements |
|---------|-------------|-------------|
| [`collect_mujoco.py`](collect_mujoco.py) | Collect trajectories from MuJoCo environments | `mujoco` extra |
| [`collect_atari.py`](collect_atari.py) | Collect trajectories from Atari environments | `atari` extra |

## Plugins

| Example | Description |
|---------|-------------|
| [`plugins/minimal_plugin/`](plugins/minimal_plugin/) | Minimal plugin with custom model and component registration |
| [`plugins/custom_encoder_plugin/`](plugins/custom_encoder_plugin/) | Custom `ObservationEncoder` component plugin |
| [`plugins/custom_dynamics_plugin/`](plugins/custom_dynamics_plugin/) | Custom residual `DynamicsModel` component plugin |
| [`plugins/smoke_minimal_plugin.py`](plugins/smoke_minimal_plugin.py) | Smoke test for the minimal plugin |

See [Extensibility docs](../docs/EXTENSIBILITY.md) for the full plugin development guide.

## Running Examples

```bash
# Install dev dependencies
uv sync --extra dev

# CPU quickstart (recommended first step)
uv run python examples/quickstart_cpu_success.py --quick

# Training examples
uv run python examples/train_dreamer.py
uv run python examples/train_tdmpc2.py

# MuJoCo examples (requires extra)
uv sync --extra mujoco
uv run python examples/train_dreamer_mujoco.py
```
