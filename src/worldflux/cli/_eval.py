"""The ``eval`` command — run evaluation metrics on a world model."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ._app import app, console
from ._rich_output import metric_table


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
            f"[wf.fail]Unknown suite:[/wf.fail] {suite}. "
            f"Available: {', '.join(sorted(SUITE_CONFIGS))}."
        )
        raise typer.Exit(code=1)

    # Load or create model
    target = Path(model_or_path)
    model_id = model_or_path

    with Status("[wf.brand]Loading model...[/wf.brand]", console=console, spinner="dots"):
        if target.exists():
            from worldflux.verify.quick import _load_model_from_target

            try:
                model = _load_model_from_target(target, device=device)
            except (FileNotFoundError, ValueError, RuntimeError) as exc:
                console.print(f"[wf.fail]Failed to load model:[/wf.fail] {exc}")
                raise typer.Exit(code=1) from None
            model_id = str(target)
        else:
            from worldflux import create_world_model

            try:
                model = create_world_model(model_or_path, device=device)
            except (ValueError, RuntimeError) as exc:
                console.print(f"[wf.fail]Failed to create model:[/wf.fail] {exc}")
                raise typer.Exit(code=1) from None

    with Status(
        f"[wf.brand]Running {suite} evaluation...[/wf.brand]",
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
            console.print(f"[wf.ok]Results written to:[/wf.ok] {output.resolve()}")
        else:
            console.print(json_str)
        if report.all_passed is False:
            raise typer.Exit(code=1)
        return

    # Rich table output
    rows: list[tuple[str, str, str, bool | None]] = []
    for result in report.results:
        threshold_str = f"{result.threshold:.4f}" if result.threshold is not None else "-"
        rows.append((result.metric, f"{result.value:.4f}", threshold_str, result.passed))

    console.print(metric_table(rows, title=f"Eval: {suite} | {model_id}"))
    console.print(f"[wf.muted]Wall time: {report.wall_time_sec:.2f}s[/wf.muted]")

    if report.all_passed is True:
        console.print("[wf.pass]All metrics passed.[/wf.pass]")
    elif report.all_passed is False:
        console.print("[wf.fail]Some metrics failed.[/wf.fail]")
        raise typer.Exit(code=1)
