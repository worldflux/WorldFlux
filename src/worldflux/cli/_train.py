"""The ``train`` command."""

from __future__ import annotations

from pathlib import Path

import torch
import typer

from worldflux.config_loader import load_config

from ._app import app, console
from ._rich_output import key_value_panel, result_banner


@app.command(rich_help_panel="Training")
def train(
    config: Path = typer.Option(
        Path("worldflux.toml"),
        "--config",
        "-c",
        help="Path to worldflux.toml configuration file.",
    ),
    steps: int | None = typer.Option(
        None,
        "--steps",
        help="Override total training steps from config.",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Override device from config (cpu, cuda, auto).",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        help="Override output directory from config.",
    ),
    resume_from: str | None = typer.Option(
        None,
        "--resume-from",
        help="Path to checkpoint to resume training from.",
    ),
    cloud: bool = typer.Option(
        False,
        "--cloud",
        help="Submit a cloud training job instead of running local training.",
    ),
    gpu: str | None = typer.Option(
        None,
        "--gpu",
        help="Cloud GPU override (for example: a10g, a100, h100).",
    ),
) -> None:
    """Train a world model using worldflux.toml configuration.

    [dim]Examples:[/dim]
      worldflux train
      worldflux train --steps 50000 --device cuda
      worldflux train --cloud --gpu a100
    """
    from worldflux import create_world_model
    from worldflux.training import Trainer, TrainingConfig
    from worldflux.training.data import create_random_buffer

    try:
        cfg = load_config(config)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[wf.fail]Configuration error:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    # Apply CLI overrides
    effective_steps = steps if steps is not None else cfg.training.total_steps
    effective_device = device if device is not None else cfg.training.device
    effective_output_dir = output_dir if output_dir is not None else cfg.training.output_dir

    if cloud:
        from worldflux.cloud import ModalBackend, WorldFluxCloudClient

        client = WorldFluxCloudClient.from_env()
        if not client.api_key:
            console.print(
                "[wf.fail]Cloud auth missing:[/wf.fail] run `worldflux login --api-key <key>` "
                "or set WORLDFLUX_CLOUD_API_KEY."
            )
            raise typer.Exit(code=1)

        cloud_gpu = gpu if gpu is not None else cfg.cloud.gpu_type
        cloud_region = cfg.cloud.region
        payload = {
            "project_name": cfg.project_name,
            "model": cfg.model,
            "architecture": {
                "obs_shape": list(cfg.architecture.obs_shape),
                "action_dim": cfg.architecture.action_dim,
                "hidden_dim": cfg.architecture.hidden_dim,
            },
            "training": {
                "total_steps": effective_steps,
                "batch_size": cfg.training.batch_size,
                "sequence_length": cfg.training.sequence_length,
                "learning_rate": cfg.training.learning_rate,
                "output_dir": effective_output_dir,
            },
            "cloud": {
                "gpu_type": cloud_gpu,
                "spot": cfg.cloud.spot,
                "region": cloud_region,
                "timeout_hours": cfg.cloud.timeout_hours,
            },
            "flywheel": {
                "opt_in": cfg.flywheel.opt_in,
                "privacy_epsilon": cfg.flywheel.privacy_epsilon,
                "privacy_delta": cfg.flywheel.privacy_delta,
            },
        }
        backend = ModalBackend(client)
        try:
            handle = backend.submit(payload)
        except RuntimeError as exc:
            console.print(f"[wf.fail]Cloud submission failed:[/wf.fail] {exc}")
            raise typer.Exit(code=1) from None

        console.print(
            key_value_panel(
                {
                    "Backend": handle.backend,
                    "Job ID": handle.job_id,
                    "GPU": cloud_gpu,
                    "Region": cloud_region,
                    "Next": f"worldflux logs {handle.job_id}",
                },
                title="Cloud Training Submitted",
                border="wf.border.success",
            )
        )
        return

    # Resolve auto device
    if effective_device == "auto":
        effective_device = "cuda" if torch.cuda.is_available() else "cpu"
    if effective_device == "cuda" and not torch.cuda.is_available():
        console.print("[wf.caution]CUDA is not available. Falling back to CPU.[/wf.caution]")
        effective_device = "cpu"

    console.print(
        key_value_panel(
            {
                "Project": cfg.project_name,
                "Model": cfg.model,
                "Obs shape": str(cfg.architecture.obs_shape),
                "Action dim": str(cfg.architecture.action_dim),
                "Steps": f"{effective_steps:,}",
                "Batch size": str(cfg.training.batch_size),
                "Device": effective_device,
                "Output": effective_output_dir,
            },
            title="WorldFlux Train",
            border="wf.border",
        )
    )

    try:
        model = create_world_model(
            model=cfg.model,
            obs_shape=cfg.architecture.obs_shape,
            action_dim=cfg.architecture.action_dim,
            hidden_dim=cfg.architecture.hidden_dim,
            device=effective_device,
        )
    except (ValueError, RuntimeError) as exc:
        console.print(f"[wf.fail]Model creation failed:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    training_config = TrainingConfig(
        total_steps=effective_steps,
        batch_size=cfg.training.batch_size,
        sequence_length=cfg.training.sequence_length,
        learning_rate=cfg.training.learning_rate,
        device=effective_device,
        output_dir=effective_output_dir,
    )

    # Create data source
    console.print("[wf.info]Preparing training data...[/wf.info]")
    data = create_random_buffer(
        obs_shape=cfg.architecture.obs_shape,
        action_dim=cfg.architecture.action_dim,
    )
    console.print(f"[wf.ok]Training data ready:[/wf.ok] {len(data)} transitions")

    trainer = Trainer(model, training_config)
    console.print(f"[wf.info]Starting training for {effective_steps:,} steps...[/wf.info]")

    try:
        trainer.train(data, resume_from=resume_from)
    except (RuntimeError, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            console.print("\n[wf.caution]Training interrupted by user.[/wf.caution]")
        else:
            console.print(f"[wf.fail]Training failed:[/wf.fail] {exc}")
            raise typer.Exit(code=1) from None

    profile = trainer.runtime_profile()
    elapsed = profile.get("elapsed_sec")
    throughput = profile.get("throughput_steps_per_sec")
    final_step = trainer.state.global_step

    summary_lines = [
        f"[wf.label]Final step:[/wf.label]  {final_step:,}",
        f"[wf.label]Output:[/wf.label]      {Path(effective_output_dir).resolve()}",
    ]
    if elapsed is not None:
        summary_lines.append(f"[wf.label]Elapsed:[/wf.label]     {elapsed:.1f}s")
    if throughput is not None:
        summary_lines.append(f"[wf.label]Throughput:[/wf.label]  {throughput:.1f} steps/s")
    summary_lines.append("")
    summary_lines.append("Next: worldflux verify --target " + effective_output_dir)

    console.print(
        result_banner(
            passed=True,
            title="Training Complete",
            lines=summary_lines,
        )
    )
