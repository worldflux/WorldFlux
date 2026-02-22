# CPU Success Path

This is the shortest official path to a successful WorldFlux run on CPU-only machines.

## Prerequisites

Before running, make sure your environment meets these requirements:

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10 | 3.11 or 3.12 |
| RAM | 2 GB free | 4 GB free |
| Disk | 500 MB | 1 GB |
| PyTorch | 2.0.0 | 2.x latest |

!!! tip "Check your Python version"
    ```bash
    python --version  # must be 3.10, 3.11, or 3.12
    ```

## Run

```bash
uv sync --extra dev
uv run python examples/quickstart_cpu_success.py --quick
```

The `--quick` flag uses a minimal configuration (8 training steps, horizon 10) so the
run finishes in under a minute on most machines.

## What Happens Step by Step

The script performs the following sequence:

1. **Create a replay buffer** -- Generates random transitions with `obs_shape=(8,)` and
   `action_dim=2`. With `--quick`, 2 000 transitions across 40 episodes are created.

2. **Build a DreamerV3 CI model** -- Instantiates a tiny DreamerV3 model
   (`dreamerv3:ci`) with MLP encoder/decoder, suitable for vector observations on CPU.

3. **Measure initial loss** -- Evaluates the untrained model on 2 random batches from
   the buffer to establish a baseline loss value.

4. **Train** -- Runs the `Trainer` for 8 steps (or 48 without `--quick`) with
   `batch_size=16` and `sequence_length=10`.

5. **Measure final loss** -- Evaluates again after training. The test checks that
   `final_loss < initial_loss`, confirming the model learned.

6. **Imagination rollout** -- Encodes one observation and rolls out the model for the
   configured horizon, generating predicted rewards.

7. **Write artifacts** -- Saves `summary.json` and `imagination.ppm` (a reward heatmap
   image) to the output directory.

## Success Criteria

The run is considered successful when all of the following are true:

- `initial_loss` and `final_loss` are finite (no NaN or Inf)
- `final_loss < initial_loss` (training improved the model)
- imagination rollout horizon matches the requested length
- artifacts are written to disk

Artifacts:

- `outputs/quickstart_cpu/summary.json`
- `outputs/quickstart_cpu/imagination.ppm`

## Reading the Output: `summary.json`

After a successful run, `summary.json` contains the following fields:

```json
{
  "scenario": "quickstart_cpu",
  "run_id": "20260221T012345Z-abc123",
  "seed": 42,
  "quick": true,
  "initial_loss": 12.345,
  "final_loss": 8.901,
  "loss_improved": true,
  "finite": true,
  "horizon": 10,
  "rollout_horizon": 10,
  "artifacts": {
    "summary": "outputs/quickstart_cpu/summary.json",
    "imagination": "outputs/quickstart_cpu/imagination.ppm"
  },
  "success": true
}
```

| Field | Meaning |
|-------|---------|
| `scenario` | Identifies this as the CPU quickstart scenario |
| `run_id` | Unique timestamped identifier for the run |
| `seed` | Random seed used for reproducibility |
| `quick` | Whether `--quick` mode was active |
| `initial_loss` | Model loss before training (baseline) |
| `final_loss` | Model loss after training (should be lower) |
| `loss_improved` | `true` if `final_loss < initial_loss` |
| `finite` | `true` if both losses are finite numbers |
| `horizon` | Requested imagination rollout horizon |
| `rollout_horizon` | Actual horizon produced by the model |
| `artifacts` | Paths to the generated output files |
| `success` | `true` only if all criteria pass |

!!! note "The imagination.ppm file"
    This is a PPM-format reward heatmap showing predicted rewards across the rollout
    horizon. You can view it with any image viewer that supports PPM, or convert it
    with ImageMagick: `convert imagination.ppm imagination.png`.

## Troubleshooting

### Dependencies missing or stale

If `uv` dependencies are missing, re-run `uv sync --extra dev`.

If the run is too slow on your machine, keep `--quick` enabled.

If your environment has stale outputs, remove `outputs/quickstart_cpu` and retry:

```bash
rm -rf outputs/quickstart_cpu
```

### Memory issues (OOM or slow performance)

!!! warning "Low memory environments"
    The `--quick` configuration uses approximately 500 MB of RAM. If you are on a
    constrained machine (e.g. CI runner, small VM), ensure at least 2 GB of free
    memory.

If you encounter `MemoryError` or the system becomes unresponsive:

1. Make sure `--quick` is enabled (it reduces buffer size from 8 000 to 2 000
   transitions).
2. Close other memory-intensive applications.
3. On Linux, check available memory:
   ```bash
   free -h
   ```
4. If the problem persists, reduce the buffer further by editing the script or running
   a manual test:
   ```python
   from worldflux import create_world_model
   import torch

   model = create_world_model(
       "dreamerv3:ci",
       obs_shape=(8,),
       action_dim=2,
       encoder_type="mlp",
       decoder_type="mlp",
       device="cpu",
   )
   obs = torch.randn(1, 8)
   state = model.encode(obs)
   actions = torch.randn(5, 1, 2)
   trajectory = model.rollout(state, actions)
   print(f"Success: rewards shape = {trajectory.rewards.shape}")
   ```

### PyTorch version incompatibility

WorldFlux requires PyTorch >= 2.0.0 and < 3.0.0.

```bash
python -c "import torch; print(torch.__version__)"
```

If your version is outdated or mismatched:

```bash
# With uv (recommended)
uv sync --extra dev

# Or reinstall PyTorch explicitly
uv pip install "torch>=2.0.0,<3.0.0"
```

!!! warning "Apple Silicon (M1/M2/M3)"
    Make sure you have a native ARM PyTorch build, not an x86 version under Rosetta.
    Install with: `uv pip install torch` -- uv will select the correct platform wheel.

### `uv sync` fails

If `uv sync` fails due to resolver conflicts or network issues, fall back to pip:

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

If specific optional dependencies fail (e.g. gymnasium environments), you can still
run the CPU success path since it only needs the `dev` extra:

```bash
pip install -e ".[dev]"
python examples/quickstart_cpu_success.py --quick
```

### NaN or Inf in losses

If `summary.json` shows `"finite": false`:

1. **Check PyTorch version** -- Older builds may have numerical issues with certain
   operations. Update to the latest 2.x release.
2. **Try a different seed**:
   ```bash
   uv run python examples/quickstart_cpu_success.py --quick --seed 123
   ```
3. **Verify installation integrity**:
   ```bash
   uv sync --extra dev --reinstall
   ```

### Loss did not improve

If `"loss_improved": false` but losses are finite:

- This can happen with very few training steps. Remove `--quick` to run 48 steps
  instead of 8:
  ```bash
  uv run python examples/quickstart_cpu_success.py
  ```
- Try a different random seed with `--seed`.

## Next Steps: Moving to GPU

Once the CPU success path passes, you can scale up to GPU training:

```python
import torch
from worldflux import create_world_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Use a larger model preset on GPU
model = create_world_model(
    "dreamerv3:size12m",    # 12M params (vs ~0.1M for CI preset)
    obs_shape=(3, 64, 64),  # image observations
    action_dim=4,
    device=device,
)
```

!!! tip "Scaling recommendations"
    | Environment | Recommended Model | GPU Memory |
    |-------------|-------------------|------------|
    | Simple vector tasks | `tdmpc2:5m` | < 2 GB |
    | Image tasks (64x64) | `dreamerv3:size12m` | ~4 GB |
    | Complex image tasks | `dreamerv3:size50m` | ~8 GB |
    | Large-scale training | `dreamerv3:size200m` | ~16 GB |

For full training workflows, see:

- [Quick Start](quickstart.md) -- model creation, rollout, and configuration
- [Training API Guide](../api/training.md) -- `Trainer`, `TrainingConfig`, callbacks
- [Benchmarks](../reference/benchmarks.md) -- performance numbers across environments
