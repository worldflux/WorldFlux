"""The ``verify`` command and helpers."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

from worldflux.verify import ParityVerifier, VerifyResult

from ._app import app, console
from ._rich_output import key_value_panel, result_banner
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
            f"[wf.fail]Unsupported verify mode:[/wf.fail] {effective_mode}. "
            "Use one of: auto, quick, proof, cloud."
        )
        raise typer.Exit(code=1)

    # Proof mode (original behavior)
    console.print(
        key_value_panel(
            {
                "Mode": "demo (synthetic)" if demo else "proof",
                "Target": target,
                "Baseline": baseline,
                "Env": env,
                "Device": device,
            },
            title="Verify - Configuration",
            border="wf.border",
        )
    )

    with Status(
        "[wf.brand]Running Bayesian Equivalence Engine (TOST)...[/wf.brand]",
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
            console.print(f"[wf.fail]Verification unavailable:[/wf.fail] {exc}")
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
        title = "\u2713 PASS: Mathematically Guaranteed Parity"
    else:
        title = "\u2717 FAIL: Parity Threshold Not Met"
    lines = [
        f"[wf.label]Bayesian Equivalence HDI:[/wf.label]  {stats.get('bayesian_equivalence_hdi', '-')}",
        f"[wf.label]TOST p-value:[/wf.label]              {stats.get('tost_p_value', '-')}",
        f"[wf.label]Samples:[/wf.label]                    {stats.get('samples', '-')}",
        f"[wf.label]Mean drop ratio:[/wf.label]            {stats.get('mean_drop_ratio', '-')}",
        f"[wf.label]CI upper (one-sided):[/wf.label]       {stats.get('ci_upper_ratio', '-')}",
        f"[wf.label]Margin:[/wf.label]                     {stats.get('margin_ratio', '-')}",
        f"[wf.label]Elapsed:[/wf.label]                    {result.elapsed_seconds:.3f}s",
    ]
    console.print(result_banner(passed=result.passed, title=title, lines=lines))
    if result.demo:
        console.print("[wf.muted]Results are synthetic (--demo mode)[/wf.muted]")
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
            f"[wf.fail]Unknown tier:[/wf.fail] {tier}. Use one of: smoke, baseline, production."
        )
        raise typer.Exit(code=1)

    console.print(f"[wf.info]Quality check:[/wf.info] tier={quality_tier.value}, target={target}")

    try:
        model = _load_model_from_target(Path(target), device=device)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[wf.fail]Failed to load model:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    result = quality_check(model, tier=quality_tier, device=device)

    if result.passed:
        console.print(
            f"[wf.pass]PASSED[/wf.pass] \u2014 "
            f"achieved: {result.achieved_tier.value}, score: {result.score:.2f}"
        )
    else:
        console.print(
            f"[wf.fail]FAILED[/wf.fail] \u2014 "
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
        key_value_panel(
            {
                "Mode": "quick (lightweight evaluation)",
                "Target": target,
                "Env": env,
                "Episodes": str(episodes),
                "Device": device,
            },
            title="Verify - Quick Mode",
            border="wf.border",
        )
    )

    with Status(
        "[wf.brand]Running quick verification...[/wf.brand]",
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
            console.print(f"[wf.fail]Quick verification failed:[/wf.fail] {exc}")
            raise typer.Exit(code=1) from None

    if format == "json":
        payload = qr.to_dict()
        json_str = json.dumps(payload, indent=2)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json_str + "\n", encoding="utf-8")
            console.print(f"[wf.ok]Results written to:[/wf.ok] {output.resolve()}")
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
        console.print(f"[wf.ok]Badge written to:[/wf.ok] {badge_path}")
        if evidence_bundle is not None:
            _write_quick_evidence_bundle(output_dir=evidence_bundle, payload=qr.to_dict())
        if not qr.passed:
            raise typer.Exit(code=1)
        return

    # Rich panel output (default)
    if qr.passed:
        title = "\u2713 PASS: Model Meets Baseline"
    else:
        title = "\u2717 FAIL: Below Baseline Threshold"

    stats = qr.stats
    lines = [
        f"[wf.label]Mean score:[/wf.label]       {qr.mean_score:.4f}",
        f"[wf.label]Baseline mean:[/wf.label]    {qr.baseline_mean:.4f}",
        f"[wf.label]Episodes:[/wf.label]         {qr.episodes}",
        f"[wf.label]Mean drop ratio:[/wf.label]  {stats.get('mean_drop_ratio', '-')}",
        f"[wf.label]CI upper:[/wf.label]         {stats.get('ci_upper_ratio', '-')}",
        f"[wf.label]Margin:[/wf.label]           {stats.get('margin_ratio', '-')}",
        f"[wf.label]Protocol:[/wf.label]         v{qr.protocol_version}",
        f"[wf.label]Elapsed:[/wf.label]          {qr.elapsed_seconds:.3f}s",
    ]
    console.print(result_banner(passed=qr.passed, title=title, lines=lines))
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
    console.print(f"[wf.ok]Evidence bundle written:[/wf.ok] {manifest_path.resolve()}")


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
            "[wf.fail]Cloud auth missing:[/wf.fail] run `worldflux login --api-key <key>`."
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
        console.print(f"[wf.fail]Cloud verification failed:[/wf.fail] {exc}")
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
            console.print(f"[wf.ok]Results written to:[/wf.ok] {output.resolve()}")
        else:
            console.print(json_payload)
    else:
        title = (
            "\u2713 PASS: Cloud verification passed"
            if passed
            else "\u2717 FAIL: Cloud verification failed"
        )
        lines = [
            f"[wf.label]Target:[/wf.label]    {response.get('target')}",
            f"[wf.label]Baseline:[/wf.label]  {response.get('baseline')}",
            f"[wf.label]Env:[/wf.label]       {response.get('env')}",
            f"[wf.label]Status:[/wf.label]    {response.get('status', '-')}",
            f"[wf.label]Verdict:[/wf.label]   {response.get('verdict_reason', '-')}",
        ]
        console.print(result_banner(passed=passed, title=title, lines=lines))

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
        console.print(f"[wf.ok]Results written to:[/wf.ok] {output.resolve()}")
    else:
        console.print(json_str)
    if not result.passed:
        raise typer.Exit(code=1)
