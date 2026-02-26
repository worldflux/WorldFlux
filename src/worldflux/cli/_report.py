"""The ``report`` command — display a training report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from ._app import app, console
from ._rich_output import key_value_panel, status_table


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
        console.print(f"[wf.fail]File not found:[/wf.fail] {path}")
        raise typer.Exit(code=1)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        console.print(f"[wf.fail]Failed to read report:[/wf.fail] {exc}")
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

    summary_data: dict[str, Any] = {
        "Model": model_id,
        "Total steps": total_steps,
    }
    if wall_time is not None:
        summary_data["Wall time"] = f"{wall_time:.1f}s"
    if throughput is not None:
        summary_data["Throughput"] = f"{throughput:.1f} steps/s"
    if final_loss is not None:
        summary_data["Final loss"] = f"{final_loss:.6f}"
    if best_loss is not None:
        summary_data["Best loss"] = f"{best_loss:.6f}"
    if health_score is not None:
        summary_data["Health score"] = f"{health_score:.2f}"

    console.print(key_value_panel(summary_data, title="Training Report", border="wf.border"))

    # Health signals
    health_signals = data.get("health_signals", {})
    if health_signals:
        status_map = {
            "healthy": "pass",
            "warning": "warn",
            "critical": "fail",
        }
        style_map = {
            "healthy": "[wf.pass]healthy[/wf.pass]",
            "warning": "[wf.warn]warning[/wf.warn]",
            "critical": "[wf.fail]critical[/wf.fail]",
        }

        rows: list[tuple[str, str, str]] = []
        for name, signal in health_signals.items():
            status = signal.get("status", "unknown")
            styled = style_map.get(status, status)
            value = signal.get("value")
            value_str = f"{value:.4f}" if isinstance(value, int | float) else str(value)
            msg = signal.get("message", "")
            rows.append((status_map.get(status, "info"), name, f"{styled}  {value_str}  {msg}"))

        console.print(status_table(rows, title="Health Signals"))

    # Loss curve summary
    loss_curve = data.get("loss_curve_summary", {})
    if loss_curve:
        lc_data: dict[str, Any] = {}
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
                    lc_data[key] = f"{value:.6f}"
                else:
                    lc_data[key] = str(value)
        if lc_data:
            console.print(
                key_value_panel(lc_data, title="Loss Curve Summary", border="wf.border.info")
            )

    # Recommendations
    recs = data.get("recommendations", [])
    if recs:
        console.print("\n[wf.label]Recommendations:[/wf.label]")
        for rec in recs:
            console.print(f"  - {rec}")
