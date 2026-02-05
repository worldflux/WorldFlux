# Reproduce DreamerV3 and TD-MPC2 (Bundled Data)

This page provides the **shortest, real-data reproduction steps** using the
bundled datasets in the repository.

## Prerequisites

```bash
uv sync --extra training
```

## Bundled Datasets

The repo already includes:

- `atari_data.npz` (image observations)
- `mujoco_data.npz` (state observations)

## Quick Verification (CPU-friendly)

These commands are intended as **short, reproducible checks**. Increase steps
for longer training runs.

### DreamerV3 (Atari)

```bash
uv run python examples/train_atari_dreamer.py --data atari_data.npz --steps 2000
```

Expected output:

- Model saved to `outputs/atari_dreamer/atari_dreamer_final`

### DreamerV3 (MuJoCo)

```bash
uv run python examples/train_dreamer_mujoco.py --data mujoco_data.npz --steps 2000
```

Expected output:

- Model saved to `outputs/dreamer_mujoco/dreamer_mujoco_final`
- Validation image in `outputs/dreamer_mujoco/dreamer_mujoco_validation.png`

### TD-MPC2 (MuJoCo)

```bash
uv run python examples/train_tdmpc2_mujoco.py --data mujoco_data.npz --steps 2000
```

Expected output:

- Model saved to `outputs/tdmpc2_mujoco/tdmpc2_final`
- Validation image in `outputs/tdmpc2_mujoco/tdmpc2_validation.png`

## Notes

- If you want **faster smoke checks**, use `--test` on each script.
- For stronger validation, increase `--steps` to 10k+.
