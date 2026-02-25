"""The ``report`` command — display a training report."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.panel import Panel
from rich.table import Table

from ._app import app, console


@app.command(rich_help_panel="Quality & Evaluation")
def report(
    path: Path = typer.Argument(..., help="Path to training_report.json."),
    format: str = typer.Option("rich", "--format", "-f"),
) -> None:
    """Display a training report from a completed training run.

    [dim]Examples:[/dim]
      worldflux report ./outputs/training_report.json
      worldflux report ./outputs/training_report.json --format json
    """
    if not path.exists():
        console.print(f"[bold red]File not found:[/bold red] {path}")
        raise typer.Exit(code=1)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        console.print(f"[bold red]Failed to read report:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    if format == "json":
        console.print(json.dumps(data, indent=2))
        return

    # Summary panel
    model_id = data.get("model_id", "unknown")
    total_steps = data.get("total_steps", "-")
    wall_time = data.get("wall_time_sec")
    final_loss = data.get("final_loss")
    best_loss = data.get("best_loss")
    throughput = data.get("throughput_steps_per_sec")
    health_score = data.get("health_score")

    summary_lines = [
        f"[bold]Model:[/bold] {model_id}",
        f"[bold]Total steps:[/bold] {total_steps}",
    ]
    if wall_time is not None:
        summary_lines.append(f"[bold]Wall time:[/bold] {wall_time:.1f}s")
    if throughput is not None:
        summary_lines.append(f"[bold]Throughput:[/bold] {throughput:.1f} steps/s")
    if final_loss is not None:
        summary_lines.append(f"[bold]Final loss:[/bold] {final_loss:.6f}")
    if best_loss is not None:
        summary_lines.append(f"[bold]Best loss:[/bold] {best_loss:.6f}")
    if health_score is not None:
        summary_lines.append(f"[bold]Health score:[/bold] {health_score:.2f}")

    console.print(Panel.fit("\n".join(summary_lines), title="Training Report", border_style="cyan"))

    # Health signals
    health_signals = data.get("health_signals", {})
    if health_signals:
        table = Table(title="Health Signals")
        table.add_column("Signal", style="bold")
        table.add_column("Status")
        table.add_column("Value", justify="right")
        table.add_column("Message")

        for name, signal in health_signals.items():
            status = signal.get("status", "unknown")
            if status == "healthy":
                styled = "[bold green]healthy[/bold green]"
            elif status == "warning":
                styled = "[bold yellow]warning[/bold yellow]"
            elif status == "critical":
                styled = "[bold red]critical[/bold red]"
            else:
                styled = status
            value = signal.get("value")
            value_str = f"{value:.4f}" if isinstance(value, int | float) else str(value)
            table.add_row(name, styled, value_str, signal.get("message", ""))

        console.print(table)

    # Loss curve summary
    loss_curve = data.get("loss_curve_summary", {})
    if loss_curve:
        lc_lines = []
        for key in (
            "initial_loss",
            "final_loss",
            "best_loss",
            "best_step",
            "convergence_slope",
            "plateau_detected",
        ):
            if key in loss_curve:
                value = loss_curve[key]
                if isinstance(value, float):
                    lc_lines.append(f"[bold]{key}:[/bold] {value:.6f}")
                else:
                    lc_lines.append(f"[bold]{key}:[/bold] {value}")
        if lc_lines:
            console.print(
                Panel.fit("\n".join(lc_lines), title="Loss Curve Summary", border_style="blue")
            )

    # Recommendations
    recs = data.get("recommendations", [])
    if recs:
        console.print("\n[bold]Recommendations:[/bold]")
        for rec in recs:
            console.print(f"  - {rec}")
