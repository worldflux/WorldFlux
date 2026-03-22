# Train Your First Model

This is the first supported end-to-end WorldFlux tutorial after installation.

It uses the scaffolded project flow that already exists in the repository today:

1. create a project with `worldflux init`
2. run a short local training job with `worldflux train`
3. validate outputs with `worldflux verify --mode quick`
4. inspect the generated helper files such as `inference.py`

If you have not completed the installation smoke test yet, do that first:
[CPU Success Path](../getting-started/cpu-success.md).

## 1. Create a Project

Install the CLI if needed:

```bash
uv tool install worldflux
```

Generate a new project:

```bash
worldflux init my-first-worldflux-model
cd my-first-worldflux-model
```

The scaffold creates these files for you:

- `worldflux.toml`
- `train.py`
- `dataset.py`
- `local_dashboard.py`
- `dashboard/index.html`
- `inference.py`

## 2. Check the Generated Config

Open `worldflux.toml` and confirm the newcomer-safe defaults:

- `training.backend = "native_torch"`
- `verify.mode = "quick"`
- `data.source` matches the scaffolded environment (`gym` for the default Atari/MuJoCo path)

You do not need parity/proof setup for this tutorial.

## 3. Run the Contract-Smoke Lane

Start with a small CPU run:

```bash
worldflux train --steps 5 --device cpu
```

What to expect in this lane:

- a training summary panel in the terminal
- an `outputs/` directory with checkpoints and `run_manifest.json`
- a local dashboard URL such as `http://127.0.0.1:8765` because the scaffolded
  project includes `local_dashboard.py` and `dashboard/index.html`

If you want a longer run later, increase `training.total_steps` in `worldflux.toml`
or pass a larger `--steps` value.

The contract-smoke lane is about command success, artifact creation, and
manifest structure. It is not the same thing as meaningful local training.

## 4. Verify the Contract-Smoke Result Locally

Run the quick compatibility check:

```bash
worldflux verify --target ./outputs --mode quick
```

This is the supported first verification path for local projects.
It is intentionally different from proof-mode parity workflows.

Quick verify may still miss the synthetic threshold if the run is too short.
In the contract-smoke lane, that should be interpreted as a workflow warning:
the command executed, artifacts are interpretable, but the run is not yet a
strong quality signal.

Check `outputs/run_manifest.json` after the run. In the smoke lane you should
expect:

- `run_classification = "contract_smoke"`
- `data_mode = "random"` unless you already configured a real environment-backed dataset
- `degraded_modes = []` for the happy path, or explicit fallback markers if the
  scaffold had to degrade

## 5. Promote to Meaningful Local Training

Once the smoke lane passes, switch to a real environment-backed run:

1. For the guaranteed DreamerV3 lane, install Atari extras with `uv sync --extra training --extra atari`
2. Set `data.source = "gym"` in `worldflux.toml`
3. Set `data.gym_env = "ALE/Breakout-v5"`
4. Increase `training.total_steps` to a value that is meaningful for your local test
5. Rerun `worldflux train`

This lane is successful only if `outputs/run_manifest.json` shows:

- `run_classification = "meaningful_local_training"`
- `data_mode = "offline"` or `data_mode = "online"`
- no `degraded_modes`

If `random_replay_fallback`, `scaffold_runtime_fallback`, or
`env_collection_unavailable` appears, treat the run as smoke only and fix the
data path before using it as evidence.

## 6. Inspect the Scaffolded Helper Script

The scaffold also creates `inference.py`.
Use it as the starting point for short rollout and imagination checks once you are ready
to inspect a trained checkpoint with the same environment you use for development.

## 7. What to Do Next

- Tune `training.total_steps`, `training.batch_size`, and `training.learning_rate` in `worldflux.toml`
- Tune `data.source`, `gameplay.enabled`, and `visualization.enabled` in `worldflux.toml`
  when you want to change how `worldflux train` collects data and displays the local dashboard
- Read [Quick Start](../getting-started/quickstart.md) for core API usage
- Read [Training API Guide](../api/training.md) for trainer and replay buffer details
- Read [Parity](../reference/parity.md) only when you need advanced proof-oriented workflows
