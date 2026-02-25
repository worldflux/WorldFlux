"""Cloud commands: login, jobs, logs, pull."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.panel import Panel

from ._app import app, console


@app.command(rich_help_panel="Cloud")
def login(
    api_key: str = typer.Option(
        ...,
        "--api-key",
        help="WorldFlux Cloud API key.",
        prompt=True,
        hide_input=True,
    ),
) -> None:
    """Store WorldFlux Cloud API credentials for CLI usage.

    [dim]Examples:[/dim]
      worldflux login --api-key sk-...
    """
    from worldflux.cloud import WorldFluxCloudClient

    client = WorldFluxCloudClient.from_env()
    try:
        client.login(api_key=api_key)
    except RuntimeError as exc:
        console.print(f"[bold red]Login failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from None
    console.print("[green]Cloud credentials saved.[/green]")


@app.command(rich_help_panel="Cloud")
def jobs() -> None:
    """List cloud training jobs.

    [dim]Examples:[/dim]
      worldflux jobs
    """
    from worldflux.cloud import WorldFluxCloudClient

    client = WorldFluxCloudClient.from_env()
    if not client.api_key:
        console.print(
            "[bold red]Cloud auth missing:[/bold red] run `worldflux login --api-key <key>`."
        )
        raise typer.Exit(code=1)

    try:
        job_rows = client.list_jobs()
    except RuntimeError as exc:
        console.print(f"[bold red]Failed to list jobs:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    if not job_rows:
        console.print("[yellow]No cloud jobs found.[/yellow]")
        return

    lines = []
    for row in job_rows:
        lines.append(
            f"{row.get('job_id', '-')}: status={row.get('status', '-')}, "
            f"model={row.get('model', '-')}, gpu={row.get('gpu_type', '-')}"
        )
    console.print(Panel.fit("\n".join(lines), title="Cloud Jobs", border_style="cyan"))


@app.command(rich_help_panel="Cloud")
def logs(
    job_id: str = typer.Argument(..., help="Cloud job ID."),
    limit: int = typer.Option(200, "--limit", help="Maximum log lines to fetch."),
) -> None:
    """Show cloud job logs.

    [dim]Examples:[/dim]
      worldflux logs <job-id>
      worldflux logs <job-id> --limit 500
    """
    from worldflux.cloud import WorldFluxCloudClient

    client = WorldFluxCloudClient.from_env()
    if not client.api_key:
        console.print(
            "[bold red]Cloud auth missing:[/bold red] run `worldflux login --api-key <key>`."
        )
        raise typer.Exit(code=1)

    try:
        lines = client.get_job_logs(job_id, limit=limit)
    except RuntimeError as exc:
        console.print(f"[bold red]Failed to fetch logs:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    if not lines:
        console.print("[yellow]No logs available for this job.[/yellow]")
        return
    for line in lines:
        console.print(line)


@app.command(rich_help_panel="Cloud")
def pull(
    job_id: str = typer.Argument(..., help="Cloud job ID."),
    output_dir: Path = typer.Option(Path("./outputs/cloud"), "--output-dir", "-o"),
) -> None:
    """Pull cloud artifact manifest for a job.

    [dim]Examples:[/dim]
      worldflux pull <job-id>
      worldflux pull <job-id> -o ./my-outputs
    """
    from worldflux.cloud import WorldFluxCloudClient

    client = WorldFluxCloudClient.from_env()
    if not client.api_key:
        console.print(
            "[bold red]Cloud auth missing:[/bold red] run `worldflux login --api-key <key>`."
        )
        raise typer.Exit(code=1)
    try:
        payload = client.pull_job_artifacts(job_id, output_dir=output_dir)
    except RuntimeError as exc:
        console.print(f"[bold red]Failed to pull artifacts:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    manifest = payload.get("manifest", "-")
    console.print(f"[green]Saved artifact manifest:[/green] {manifest}")
