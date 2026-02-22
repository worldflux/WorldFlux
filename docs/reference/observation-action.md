# Observation Shape and Action Dim

When `worldflux init` asks for these values, it needs the exact I/O contract of your environment.

## Observation Shape

`Observation shape` is the per-step observation tensor shape.

- Atari/image tasks usually use channel-first shape like `3,64,64`
- Vector/state tasks usually use one dimension like `39`
- This value maps to `architecture.obs_shape` in `worldflux.toml`

Examples:

- `3,64,64`: RGB image observation (channels, height, width)
- `1,84,84`: grayscale frame stack with one channel
- `39`: flat state vector with 39 features

## Action Dim

`Action dim` is the action width expected by the world model.

- Discrete environments: use the number of discrete actions (for one-hot action vectors)
- Continuous environments: use the size of the continuous action vector
- This value maps to `architecture.action_dim` in `worldflux.toml`

Examples:

- Breakout-like discrete control with 6 actions: `action_dim = 6`
- MuJoCo HalfCheetah with 6-dim continuous action: `action_dim = 6`

## How To Pick Correct Values

1. Check your environment spec first.
2. Set `obs_shape` to the exact observation tensor shape used in training.
3. Set `action_dim` to the exact action width emitted by your policy/planner.
4. Keep these values consistent across `train.py`, `inference.py`, and datasets.

## Common Mistakes

- Using `64,64,3` while pipeline expects channel-first `3,64,64`
- Entering `action_dim=1` for a discrete environment with multiple actions
- Changing `obs_shape` after checkpoints are created (can break load/inference)

## Shape Reference Table

| Environment Type | obs_shape | action_dim | action_type | Notes |
|-----------------|-----------|------------|-------------|-------|
| Atari (standard) | `3, 64, 64` | 18 | discrete | Channel-first RGB |
| Atari (grayscale stack) | `4, 84, 84` | 18 | discrete | 4-frame stack |
| MuJoCo HalfCheetah | `17` | 6 | continuous | Flat state vector |
| MuJoCo Humanoid | `376` | 17 | continuous | High-dim state |
| DMControl Walker | `24` | 6 | continuous | Proprioceptive |
| DMControl Pixels | `3, 84, 84` | varies | continuous | Visual observation |
| Custom image env | `C, H, W` | varies | varies | Channel-first required |

## Multi-Modal Observations (v3 API)

WorldFlux v3 supports multi-modal observation specifications via `observation_modalities`:

```python
model = create_world_model(
    "dreamerv3:size12m",
    observation_modalities={
        "image": {"shape": (3, 64, 64), "kind": "image"},
        "proprio": {"shape": (12,), "kind": "vector"},
    },
    action_dim=6,
)
```

Each modality entry requires:

- `shape`: Tensor shape per time step (excluding batch dimension)
- `kind`: One of `"image"`, `"vector"`, `"video"`, `"tokens"`, `"text"`, `"other"`
- `dtype` (optional): `"float32"` (default), `"float16"`, or `"bfloat16"`

## Action Specifications (v3 API)

For explicit action control, use `action_spec`:

```python
model = create_world_model(
    "tdmpc2:5m",
    obs_shape=(39,),
    action_spec={"kind": "continuous", "dim": 6},
)
```

Supported `kind` values: `"continuous"`, `"discrete"`, `"token"`, `"latent"`, `"none"`.

## Programmatic Inspection

```python
from worldflux import get_config

config = get_config("dreamerv3:size12m")
print(f"obs_shape: {config.obs_shape}")        # (3, 64, 64)
print(f"action_dim: {config.action_dim}")      # 6
print(f"action_type: {config.action_type}")    # continuous
print(f"modalities: {config.observation_modalities}")
```

## Related Docs

- [Quick Start](../getting-started/quickstart.md)
- [Model Choice in `worldflux init`](../getting-started/quickstart.md#4-choosing-a-model-in-worldflux-init)
- [Configuration Reference](config.md)
- [State Reference](../api/state.md)
- [Troubleshooting](troubleshooting.md)
