"""The ``verify`` command and helpers."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from rich.panel import Panel

from worldflux.verify import ParityVerifier, VerifyResult

from ._app import app, console
from ._utils import _hash_file


@app.command(rich_help_panel="Quality & Evaluation")
def verify(
    target: str = typer.Option(..., "--target", help="Path or registry ID of your custom model."),
    baseline: str = typer.Option(
        "official/dreamerv3", "--baseline", help="Baseline to compare against."
    ),
    env: str = typer.Option("atari/pong", "--env", help="Target simulation environment."),
    demo: bool = typer.Option(False, "--demo", help="Mock mode for demonstrations and VC pitches."),
    device: str = typer.Option("cpu", "--device", help="Execution device."),
    mode: str = typer.Option(
        "auto",
        "--mode",
        help="Verification mode: auto, quick, proof, or cloud.",
    ),
    format: str = typer.Option(
        "rich", "--format", help="Output format: rich (default), json, or badge."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output file path for json/badge format."
    ),
    episodes: int = typer.Option(
        10, "--episodes", help="Number of evaluation episodes (quick mode)."
    ),
    evidence_bundle: Path | None = typer.Option(
        None,
        "--evidence-bundle",
        help="Output directory for verification evidence bundle (manifest + artifacts).",
    ),
    tier: str | None = typer.Option(
        None,
        "--tier",
        help="Quality tier check: smoke, baseline, or production.",
    ),
) -> None:
    """Verify your model against an official baseline.

    [dim]Examples:[/dim]
      worldflux verify --target ./outputs
      worldflux verify --target ./outputs --mode quick --episodes 20
      worldflux verify --target ./outputs --tier smoke
    """
    from rich.status import Status

    # Quality tier check mode
    if tier is not None:
        _run_quality_tier_check(target=target, tier=tier, device=device)
        return

    # Determine verification mode
    effective_mode = mode.strip().lower()
    if effective_mode == "auto":
        # Use quick mode when scripts/parity is not available (pip install users)
        scripts_root = Path(__file__).resolve().parents[3] / "scripts" / "parity"
        effective_mode = "proof" if scripts_root.exists() else "quick"

    if effective_mode == "quick":
        _run_quick_verify(
            target=target,
            env=env,
            device=device,
            episodes=episodes,
            format=format,
            output=output,
            evidence_bundle=evidence_bundle,
        )
        return
    if effective_mode == "cloud":
        _run_cloud_verify(
            target=target,
            baseline=baseline,
            env=env,
            device=device,
            format=format,
            output=output,
            evidence_bundle=evidence_bundle,
        )
        return
    if effective_mode != "proof":
        console.print(
            f"[bold red]Unsupported verify mode:[/bold red] {effective_mode}. "
            "Use one of: auto, quick, proof, cloud."
        )
        raise typer.Exit(code=1)

    # Proof mode (original behavior)
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Mode: {'demo (synthetic)' if demo else 'proof'}",
                    f"Target: {target}",
                    f"Baseline: {baseline}",
                    f"Env: {env}",
                    f"Device: {device}",
                ]
            ),
            title="Verify - Configuration",
            border_style="cyan",
        )
    )

    with Status(
        "[bold cyan]Running Bayesian Equivalence Engine (TOST)...[/bold cyan]",
        console=console,
        spinner="dots",
    ):
        try:
            result: VerifyResult = ParityVerifier.run(
                target=target,
                baseline=baseline,
                env=env,
                demo=demo,
                device=device,
            )
        except (NotImplementedError, RuntimeError, OSError, ValueError) as exc:
            console.print(f"[bold red]Verification unavailable:[/bold red] {exc}")
            raise typer.Exit(code=1) from None

    if format == "json":
        _emit_verify_json(result, output)
        if evidence_bundle is not None:
            _write_proof_evidence_bundle(
                output_dir=evidence_bundle,
                result=result,
                mode="proof" if not demo else "demo",
            )
        return

    stats = result.stats
    if result.passed:
        title = "\u2705 PASS: Mathematically Guaranteed Parity"
        border = "green"
    else:
        title = "\u274c FAIL: Parity Threshold Not Met"
        border = "red"
    lines = [
        f"Bayesian Equivalence HDI: [bold]{stats.get('bayesian_equivalence_hdi', '-')}[/bold]",
        f"TOST p-value: [bold]{stats.get('tost_p_value', '-')}[/bold]",
        f"Samples: {stats.get('samples', '-')}",
        f"Mean drop ratio: {stats.get('mean_drop_ratio', '-')}",
        f"CI upper (one-sided): {stats.get('ci_upper_ratio', '-')}",
        f"Margin: {stats.get('margin_ratio', '-')}",
        f"Elapsed: {result.elapsed_seconds:.3f}s",
    ]
    console.print(
        Panel.fit(
            "\n".join(lines),
            title=title,
            border_style=border,
        )
    )
    if result.demo:
        console.print("[dim]Results are synthetic (--demo mode)[/dim]")
    if evidence_bundle is not None:
        _write_proof_evidence_bundle(
            output_dir=evidence_bundle,
            result=result,
            mode="proof" if not demo else "demo",
        )
    if not result.passed:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_quality_tier_check(
    *,
    target: str,
    tier: str,
    device: str,
) -> None:
    """Run quality tier check on a model checkpoint."""
    from worldflux.verify.quick import QualityTier, _load_model_from_target, quality_check

    tier_map = {
        "smoke": QualityTier.SMOKE,
        "baseline": QualityTier.BASELINE,
        "production": QualityTier.PRODUCTION,
    }
    quality_tier = tier_map.get(tier.strip().lower())
    if quality_tier is None:
        console.print(
            f"[bold red]Unknown tier:[/bold red] {tier}. Use one of: smoke, baseline, production."
        )
        raise typer.Exit(code=1)

    console.print(f"[cyan]Quality check:[/cyan] tier={quality_tier.value}, target={target}")

    try:
        model = _load_model_from_target(Path(target), device=device)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Failed to load model:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    result = quality_check(model, tier=quality_tier, device=device)

    if result.passed:
        console.print(
            f"[bold green]PASSED[/bold green] — "
            f"achieved: {result.achieved_tier.value}, score: {result.score:.2f}"
        )
    else:
        console.print(
            f"[bold red]FAILED[/bold red] — "
            f"target: {quality_tier.value}, achieved: {result.achieved_tier.value}, "
            f"score: {result.score:.2f}"
        )
        raise typer.Exit(code=1)


def _run_quick_verify(
    *,
    target: str,
    env: str,
    device: str,
    episodes: int,
    format: str,
    output: Path | None,
    evidence_bundle: Path | None,
) -> None:
    """Execute quick verify mode for pip-install users."""
    from rich.status import Status

    from worldflux.parity import save_badge
    from worldflux.verify.quick import quick_verify

    console.print(
        Panel.fit(
            "\n".join(
                [
                    "Mode: quick (lightweight evaluation)",
                    f"Target: {target}",
                    f"Env: {env}",
                    f"Episodes: {episodes}",
                    f"Device: {device}",
                ]
            ),
            title="Verify - Quick Mode",
            border_style="cyan",
        )
    )

    with Status(
        "[bold cyan]Running quick verification...[/bold cyan]",
        console=console,
        spinner="dots",
    ):
        try:
            qr = quick_verify(
                target=target,
                env=env,
                episodes=episodes,
                device=device,
            )
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
            console.print(f"[bold red]Quick verification failed:[/bold red] {exc}")
            raise typer.Exit(code=1) from None

    if format == "json":
        payload = qr.to_dict()
        json_str = json.dumps(payload, indent=2)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json_str + "\n", encoding="utf-8")
            console.print(f"[green]Results written to:[/green] {output.resolve()}")
        else:
            console.print(json_str)
        if evidence_bundle is not None:
            _write_quick_evidence_bundle(output_dir=evidence_bundle, payload=payload)
        if not qr.passed:
            raise typer.Exit(code=1)
        return

    if format == "badge":
        family = env.split("/")[0] if "/" in env else env
        save_badge(
            path=output or Path("verify-badge.svg"),
            family=family,
            passed=qr.passed,
            confidence=0.95,
            margin=qr.stats.get("margin_ratio", 0.15),
        )
        badge_path = (output or Path("verify-badge.svg")).resolve()
        console.print(f"[green]Badge written to:[/green] {badge_path}")
        if evidence_bundle is not None:
            _write_quick_evidence_bundle(output_dir=evidence_bundle, payload=qr.to_dict())
        if not qr.passed:
            raise typer.Exit(code=1)
        return

    # Rich panel output (default)
    if qr.passed:
        title = "\u2705 PASS: Model Meets Baseline"
        border = "green"
    else:
        title = "\u274c FAIL: Below Baseline Threshold"
        border = "red"

    stats = qr.stats
    lines = [
        f"Mean score: [bold]{qr.mean_score:.4f}[/bold]",
        f"Baseline mean: [bold]{qr.baseline_mean:.4f}[/bold]",
        f"Episodes: {qr.episodes}",
        f"Mean drop ratio: {stats.get('mean_drop_ratio', '-')}",
        f"CI upper: {stats.get('ci_upper_ratio', '-')}",
        f"Margin: {stats.get('margin_ratio', '-')}",
        f"Protocol: v{qr.protocol_version}",
        f"Elapsed: {qr.elapsed_seconds:.3f}s",
    ]
    console.print(
        Panel.fit(
            "\n".join(lines),
            title=title,
            border_style=border,
        )
    )
    if evidence_bundle is not None:
        _write_quick_evidence_bundle(output_dir=evidence_bundle, payload=qr.to_dict())
    if not qr.passed:
        raise typer.Exit(code=1)


def _write_evidence_manifest(
    *,
    output_dir: Path,
    mode: str,
    request_payload: dict[str, Any],
    result_payload: dict[str, Any],
    local_artifacts: list[Path],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, Any]] = []
    for src in local_artifacts:
        if not src.exists():
            continue
        dest = artifacts_dir / src.name
        shutil.copy2(src, dest)
        copied.append(
            {
                "source": str(src.resolve()),
                "path": str(dest.resolve()),
                "sha256": _hash_file(dest),
                "bytes": dest.stat().st_size,
            }
        )

    manifest = {
        "schema_version": "worldflux.verify.evidence.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "request": request_payload,
        "result": result_payload,
        "artifacts": copied,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    console.print(f"[green]Evidence bundle written:[/green] {manifest_path.resolve()}")


def _write_quick_evidence_bundle(*, output_dir: Path, payload: dict[str, Any]) -> None:
    local_payload = output_dir / "quick_result.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    local_payload.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_evidence_manifest(
        output_dir=output_dir,
        mode="quick",
        request_payload={
            "target": payload.get("target"),
            "env": payload.get("env"),
            "episodes": payload.get("episodes"),
        },
        result_payload=payload,
        local_artifacts=[local_payload],
    )


def _write_proof_evidence_bundle(*, output_dir: Path, result: VerifyResult, mode: str) -> None:
    stats = result.stats
    report_paths: list[Path] = []
    for key in ("runs_jsonl", "equivalence_report_json", "equivalence_report_md"):
        raw_value = str(stats.get(key, "")).strip()
        if not raw_value:
            continue
        report_paths.append(Path(raw_value).expanduser())
    result_payload = {
        "passed": result.passed,
        "target": result.target,
        "baseline": result.baseline,
        "env": result.env,
        "demo": result.demo,
        "elapsed_seconds": result.elapsed_seconds,
        "stats": dict(result.stats),
        "verdict_reason": result.verdict_reason,
    }
    _write_evidence_manifest(
        output_dir=output_dir,
        mode=mode,
        request_payload={
            "target": result.target,
            "baseline": result.baseline,
            "env": result.env,
            "device": result.stats.get("device"),
        },
        result_payload=result_payload,
        local_artifacts=report_paths,
    )


def _run_cloud_verify(
    *,
    target: str,
    baseline: str,
    env: str,
    device: str,
    format: str,
    output: Path | None,
    evidence_bundle: Path | None,
) -> None:
    """Execute cloud verification mode through WorldFlux Cloud API."""
    from worldflux.cloud import WorldFluxCloudClient

    client = WorldFluxCloudClient.from_env()
    if not client.api_key:
        console.print(
            "[bold red]Cloud auth missing:[/bold red] run `worldflux login --api-key <key>`."
        )
        raise typer.Exit(code=1)

    request_payload = {
        "target": target,
        "baseline": baseline,
        "env": env,
        "device": device,
    }
    try:
        response = client.verify_cloud(request_payload)
    except RuntimeError as exc:
        console.print(f"[bold red]Cloud verification failed:[/bold red] {exc}")
        raise typer.Exit(code=1) from None

    passed = bool(response.get("passed", False))
    response.setdefault("mode", "cloud")
    response.setdefault("target", target)
    response.setdefault("baseline", baseline)
    response.setdefault("env", env)
    response.setdefault("stats", {})

    if format == "json":
        json_payload = json.dumps(response, indent=2)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json_payload + "\n", encoding="utf-8")
            console.print(f"[green]Results written to:[/green] {output.resolve()}")
        else:
            console.print(json_payload)
    else:
        title = (
            "\u2705 PASS: Cloud verification passed"
            if passed
            else "\u274c FAIL: Cloud verification failed"
        )
        border = "green" if passed else "red"
        console.print(
            Panel.fit(
                "\n".join(
                    [
                        f"Target: {response.get('target')}",
                        f"Baseline: {response.get('baseline')}",
                        f"Env: {response.get('env')}",
                        f"Status: {response.get('status', '-')}",
                        f"Verdict: {response.get('verdict_reason', '-')}",
                    ]
                ),
                title=title,
                border_style=border,
            )
        )

    if evidence_bundle is not None:
        _write_evidence_manifest(
            output_dir=evidence_bundle,
            mode="cloud",
            request_payload=request_payload,
            result_payload=response,
            local_artifacts=[],
        )

    if not passed:
        raise typer.Exit(code=1)


def _emit_verify_json(result: VerifyResult, output: Path | None) -> None:
    """Emit VerifyResult as JSON."""
    payload = {
        "passed": result.passed,
        "target": result.target,
        "baseline": result.baseline,
        "env": result.env,
        "demo": result.demo,
        "elapsed_seconds": result.elapsed_seconds,
        "stats": result.stats,
        "verdict_reason": result.verdict_reason,
    }
    json_str = json.dumps(payload, indent=2)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json_str + "\n", encoding="utf-8")
        console.print(f"[green]Results written to:[/green] {output.resolve()}")
    else:
        console.print(json_str)
    if not result.passed:
        raise typer.Exit(code=1)
