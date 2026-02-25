"""The ``eval`` command — run evaluation metrics on a world model."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.table import Table

from ._app import app, console


@app.command("eval", rich_help_panel="Quality & Evaluation")
def eval_cmd(
    model_or_path: str = typer.Argument(..., help="Checkpoint path or model ID."),
    suite: str = typer.Option(
        "quick", "--suite", "-s", help="quick (~5s), standard (~30s), comprehensive (~5min)."
    ),
    device: str = typer.Option("cpu", "--device"),
    output: Path | None = typer.Option(None, "--output", "-o"),
    format: str = typer.Option("rich", "--format", "-f"),
) -> None:
    """Run evaluation metrics on a world model.

    [dim]Examples:[/dim]
      worldflux eval dreamer:ci --suite quick
      worldflux eval ./outputs --suite standard --device cuda
      worldflux eval tdmpc2:ci --suite comprehensive -o results.json
    """
    from rich.status import Status

    from worldflux.evals import SUITE_CONFIGS, run_eval_suite

    if suite not in SUITE_CONFIGS:
        console.print(
            f"[bold red]Unknown suite:[/bold red] {suite}. "
            f"Available: {', '.join(sorted(SUITE_CONFIGS))}."
        )
        raise typer.Exit(code=1)

    # Load or create model
    target = Path(model_or_path)
    model_id = model_or_path

    with Status("[bold cyan]Loading model...[/bold cyan]", console=console, spinner="dots"):
        if target.exists():
            from worldflux.verify.quick import _load_model_from_target

            try:
                model = _load_model_from_target(target, device=device)
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                console.print(f"[bold red]Failed to load model:[/bold red] {exc}")
                raise typer.Exit(code=1) from None
            model_id = str(target)
        else:
            from worldflux import create_world_model

            try:
                model = create_world_model(model_or_path, device=device)
            except (ValueError, RuntimeError) as exc:
                console.print(f"[bold red]Failed to create model:[/bold red] {exc}")
                raise typer.Exit(code=1) from None

    with Status(
        f"[bold cyan]Running {suite} evaluation...[/bold cyan]",
        console=console,
        spinner="dots",
    ):
        report = run_eval_suite(
            model,  # type: ignore[arg-type]
            suite=suite,
            device=device,
            model_id=model_id,
            output_path=output,
        )

    if format == "json":
        json_str = json.dumps(report.to_dict(), indent=2)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json_str + "\n", encoding="utf-8")
            console.print(f"[green]Results written to:[/green] {output.resolve()}")
        else:
            console.print(json_str)
        if report.all_passed is False:
            raise typer.Exit(code=1)
        return

    # Rich table output
    table = Table(title=f"Eval: {suite} | {model_id}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Status")

    for result in report.results:
        threshold_str = f"{result.threshold:.4f}" if result.threshold is not None else "-"
        if result.passed is True:
            status = "[green]✓[/green] [bold green]PASS[/bold green]"
        elif result.passed is False:
            status = "[red]✗[/red] [bold red]FAIL[/bold red]"
        else:
            status = "[dim]·[/dim] [dim]info[/dim]"
        table.add_row(result.metric, f"{result.value:.4f}", threshold_str, status)

    console.print(table)
    console.print(f"[dim]Wall time: {report.wall_time_sec:.2f}s[/dim]")

    if report.all_passed is True:
        console.print("[bold green]All metrics passed.[/bold green]")
    elif report.all_passed is False:
        console.print("[bold red]Some metrics failed.[/bold red]")
        raise typer.Exit(code=1)
