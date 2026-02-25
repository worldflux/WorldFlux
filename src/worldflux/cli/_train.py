"""The ``train`` command."""

from __future__ import annotations

from pathlib import Path

import torch
import typer
from rich.panel import Panel

from worldflux.config_loader import load_config

from ._app import app, console


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
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
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
                "[bold red]Cloud auth missing:[/bold red] run `worldflux login --api-key <key>` "
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
            console.print(f"[bold red]Cloud submission failed:[/bold red] {exc}")
            raise typer.Exit(code=1) from None

        console.print(
            Panel.fit(
                "\n".join(
                    [
                        f"[bold]Backend:[/bold] {handle.backend}",
                        f"[bold]Job ID:[/bold] {handle.job_id}",
                        f"[bold]GPU:[/bold] {cloud_gpu}",
                        f"[bold]Region:[/bold] {cloud_region}",
                        "",
                        "Next:",
                        f"  worldflux logs {handle.job_id}",
                        f"  worldflux pull {handle.job_id}",
                    ]
                ),
                title="Cloud Training Submitted",
                border_style="green",
            )
        )
        return

    # Resolve auto device
    if effective_device == "auto":
        effective_device = "cuda" if torch.cuda.is_available() else "cpu"
    if effective_device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]CUDA is not available. Falling back to CPU.[/yellow]")
        effective_device = "cpu"

    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]Project:[/bold] {cfg.project_name}",
                    f"[bold]Model:[/bold] {cfg.model}",
                    f"[bold]Obs shape:[/bold] {cfg.architecture.obs_shape}",
                    f"[bold]Action dim:[/bold] {cfg.architecture.action_dim}",
                    f"[bold]Steps:[/bold] {effective_steps:,}",
                    f"[bold]Batch size:[/bold] {cfg.training.batch_size}",
                    f"[bold]Device:[/bold] {effective_device}",
                    f"[bold]Output:[/bold] {effective_output_dir}",
                ]
            ),
            title="WorldFlux Train",
            border_style="cyan",
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
        console.print(f"[bold red]Model creation failed:[/bold red] {exc}")
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
    console.print("[cyan]Preparing training data...[/cyan]")
    data = create_random_buffer(
        obs_shape=cfg.architecture.obs_shape,
        action_dim=cfg.architecture.action_dim,
    )
    console.print(f"[green]Training data ready:[/green] {len(data)} transitions")

    trainer = Trainer(model, training_config)
    console.print(f"[cyan]Starting training for {effective_steps:,} steps...[/cyan]")

    try:
        trainer.train(data, resume_from=resume_from)
    except (RuntimeError, KeyboardInterrupt) as exc:
        if isinstance(exc, KeyboardInterrupt):
            console.print("\n[yellow]Training interrupted by user.[/yellow]")
        else:
            console.print(f"[bold red]Training failed:[/bold red] {exc}")
            raise typer.Exit(code=1) from None

    profile = trainer.runtime_profile()
    elapsed = profile.get("elapsed_sec")
    throughput = profile.get("throughput_steps_per_sec")
    final_step = trainer.state.global_step

    summary_lines = [
        f"[bold]Final step:[/bold] {final_step:,}",
        f"[bold]Output:[/bold] {Path(effective_output_dir).resolve()}",
    ]
    if elapsed is not None:
        summary_lines.append(f"[bold]Elapsed:[/bold] {elapsed:.1f}s")
    if throughput is not None:
        summary_lines.append(f"[bold]Throughput:[/bold] {throughput:.1f} steps/s")
    summary_lines.append("")
    summary_lines.append("Next: worldflux verify --target " + effective_output_dir)

    console.print(
        Panel.fit(
            "\n".join(summary_lines),
            title="Training Complete",
            border_style="green",
        )
    )
