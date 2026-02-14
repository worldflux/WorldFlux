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

## Related Docs

- [Quick Start](../getting-started/quickstart.md)
- [Model Choice in `worldflux init`](../getting-started/quickstart.md#4-choosing-a-model-in-worldflux-init)
- [Configuration Reference](config.md)
- [Troubleshooting](troubleshooting.md)
