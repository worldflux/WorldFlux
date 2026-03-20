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

You do not need parity/proof setup for this tutorial.

## 3. Run a Short Training Job

Start with a small CPU run:

```bash
worldflux train --steps 5 --device cpu
```

What to expect:

- a training summary panel in the terminal
- an `outputs/` directory with checkpoints and `run_manifest.json`
- a local dashboard URL such as `http://127.0.0.1:8765`

If you want a longer run later, increase `training.total_steps` in `worldflux.toml`
or pass a larger `--steps` value.

## 4. Verify the Result Locally

Run the quick compatibility check:

```bash
worldflux verify --target ./outputs --mode quick
```

This is the supported first verification path for local projects.
It is intentionally different from proof-mode parity workflows.

Quick verify may still return a failing non-inferiority verdict if the run is too short.
For a first pass, the important thing is that the command executes and produces a structured result.

## 5. Inspect the Scaffolded Helper Script

The scaffold also creates `inference.py`.
Use it as the starting point for short rollout and imagination checks once you are ready
to inspect a trained checkpoint with the same environment you use for development.

## 6. What to Do Next

- Tune `training.total_steps`, `training.batch_size`, and `training.learning_rate` in `worldflux.toml`
- Switch `data.source` from random data to gym-backed collection when your environment setup is ready
- Read [Quick Start](../getting-started/quickstart.md) for core API usage
- Read [Training API Guide](../api/training.md) for trainer and replay buffer details
- Read [Parity](../reference/parity.md) only when you need advanced proof-oriented workflows
