"""The ``verify`` command and helpers."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
from click.core import ParameterSource

from worldflux.config_loader import load_config
from worldflux.execution import resolve_proof_backend_defaults
from worldflux.verify import ParityVerifier, VerifyResult

from ._app import app, console
from ._rich_output import key_value_panel, result_banner
from ._utils import _hash_file


def _resolve_effective_verify_mode(
    *,
    cli_mode: str,
    config_mode: str | None,
    proof_claim: str,
) -> str:
    explicit_mode = str(cli_mode).strip().lower()
    if explicit_mode and explicit_mode != "auto":
        return explicit_mode

    configured_mode = str(config_mode or "").strip().lower()
    if configured_mode and configured_mode != "auto":
        return configured_mode

    env_mode = os.getenv("WORLDFLUX_VERIFY_MODE", "").strip().lower()
    if env_mode == "proof":
        return "proof"
    if env_mode == "quick":
        return "quick"

    if proof_claim in {"compare", "official_only"}:
        return "proof"
    return "quick"


def _was_cli_option_provided(ctx: typer.Context, param_name: str) -> bool:
    return ctx.get_parameter_source(param_name) == ParameterSource.COMMANDLINE


@app.command(rich_help_panel="Quality & Evaluation")
def verify(
    ctx: typer.Context,
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Optional worldflux.toml path used for verify defaults.",
    ),
    target: str = typer.Option(..., "--target", help="Path or registry ID of your custom model."),
    baseline: str = typer.Option(
        "official/dreamerv3", "--baseline", help="Baseline to compare against."
    ),
    env: str = typer.Option("atari/pong", "--env", help="Target simulation environment."),
    demo: bool = typer.Option(
        False,
        "--demo",
        help="Synthetic demonstration mode for pitches and screenshots. Never use as proof.",
    ),
    device: str = typer.Option("cpu", "--device", help="Execution device."),
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="Execution backend override for proof mode. Quick mode stays local-native.",
    ),
    backend_profile: str | None = typer.Option(
        None,
        "--backend-profile",
        help="Backend profile override for proof mode.",
    ),
    allow_official_only: bool | None = typer.Option(
        None,
        "--allow-official-only/--no-allow-official-only",
        help="Allow Dreamer official-only bootstrap routing.",
    ),
    proof_claim: str | None = typer.Option(
        None,
        "--proof-claim",
        help="Proof routing intent override (for example: compare).",
    ),
    mode: str = typer.Option(
        "auto",
        "--mode",
        help="Verification mode: auto, quick, proof, or cloud (experimental).",
    ),
    format: str = typer.Option(
        "rich", "--format", help="Output format: rich (default), json, or badge."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Output file path for json/badge format."
    ),
    episodes: int = typer.Option(
        10, "--episodes", help="Number of evaluation episodes (synthetic smoke mode)."
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

    config_payload = None
    config_path = config
    if config_path is not None and not config_path.exists():
        console.print(
            f"[wf.fail]Configuration error:[/wf.fail] Configuration file not found: {config_path}"
        )
        raise typer.Exit(code=1) from None
    if config_path is None:
        candidate = Path("worldflux.toml")
        if candidate.exists():
            config_path = candidate
    if config_path is not None and config_path.exists():
        try:
            config_payload = load_config(config_path)
        except (FileNotFoundError, ValueError) as exc:
            console.print(f"[wf.fail]Configuration error:[/wf.fail] {exc}")
            raise typer.Exit(code=1) from None

    effective_baseline = baseline
    effective_env = env
    effective_backend = backend or ""
    effective_backend_profile = backend_profile or ""
    effective_allow_official_only = False if allow_official_only is None else allow_official_only
    effective_proof_claim = proof_claim or "compare"
    baseline_provided = _was_cli_option_provided(ctx, "baseline")
    env_provided = _was_cli_option_provided(ctx, "env")
    config_verify_raw: dict[str, Any] = {}

    if config_payload is not None:
        raw_verify = config_payload.raw.get("verify", {})
        if isinstance(raw_verify, dict):
            config_verify_raw = raw_verify
        if not baseline_provided:
            effective_baseline = config_payload.verify.baseline
        if not env_provided:
            effective_env = config_payload.verify.env
        if backend is None and "backend" in config_verify_raw:
            effective_backend = config_payload.verify.backend
        if backend_profile is None and "backend_profile" in config_verify_raw:
            effective_backend_profile = config_payload.verify.backend_profile
        if allow_official_only is None:
            effective_allow_official_only = config_payload.verify.allow_official_only
        if proof_claim is None:
            effective_proof_claim = config_payload.verify.proof_claim

    # Quality tier check mode
    if tier is not None:
        _run_quality_tier_check(target=target, tier=tier, device=device)
        return

    # Determine verification mode
    configured_mode = str(config_payload.verify.mode).strip().lower() if config_payload else None
    effective_mode = _resolve_effective_verify_mode(
        cli_mode=mode,
        config_mode=configured_mode,
        proof_claim=effective_proof_claim,
    )

    if effective_mode == "proof":
        inferred_family = ParityVerifier._infer_family(
            baseline=effective_baseline,
            target=target,
        )
        effective_backend, effective_backend_profile = resolve_proof_backend_defaults(
            inferred_family,
            backend=effective_backend or None,
            backend_profile=effective_backend_profile or None,
        )
    else:
        effective_backend = str(effective_backend or "native_torch").strip() or "native_torch"
        effective_backend_profile = str(effective_backend_profile or "").strip()

    if effective_mode == "quick":
        _run_quick_verify(
            target=target,
            env=effective_env,
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
            baseline=effective_baseline,
            env=effective_env,
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
                "Mode": "demo (synthetic, not proof)" if demo else "proof",
                "Target": target,
                "Baseline": effective_baseline,
                "Env": effective_env,
                "Device": device,
                "Backend": effective_backend,
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
                baseline=effective_baseline,
                env=effective_env,
                demo=demo,
                device=device,
                backend=effective_backend,
                backend_profile=effective_backend_profile,
                allow_official_only=effective_allow_official_only,
                proof_claim=effective_proof_claim,
            )
        except (NotImplementedError, RuntimeError, OSError, ValueError) as exc:
            error_msg = str(exc)
            if "\n" in error_msg:
                from rich.panel import Panel

                console.print(
                    Panel(
                        error_msg,
                        title="[wf.fail]Verification failed[/wf.fail]",
                        border_style="red",
                        expand=False,
                    )
                )
            else:
                console.print(f"[wf.fail]Verification unavailable:[/wf.fail] {error_msg}")
            console.print(
                "[wf.muted]Hint: use --mode quick for synthetic smoke verification, "
                "or --demo for a synthetic demonstration.[/wf.muted]"
            )
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
    execution_result = stats.get("execution_result")
    if isinstance(execution_result, dict) and not result.demo:
        status = str(execution_result.get("status", "failed"))
        if status == "succeeded":
            title = "\u2713 PASS: Proof-mode parity checks satisfied"
        elif status == "blocked":
            title = "\u26a0 BLOCKED: Proof execution blocked"
        elif status == "incomplete":
            title = "\u23f3 INCOMPLETE: Proof prerequisites not met"
        else:
            title = "\u2717 FAIL: Proof execution failed"
        lines = [
            f"[wf.label]Status:[/wf.label]      {status}",
            f"[wf.label]Reason:[/wf.label]      {execution_result.get('reason_code', '-')}",
            f"[wf.label]Message:[/wf.label]     {execution_result.get('message', '-')}",
            f"[wf.label]Phase:[/wf.label]       {execution_result.get('proof_phase', '-')}",
            f"[wf.label]Profile:[/wf.label]     {execution_result.get('profile', '-')}",
            f"[wf.label]Next:[/wf.label]        {execution_result.get('next_action', '-')}",
            f"[wf.label]Elapsed:[/wf.label]     {result.elapsed_seconds:.3f}s",
        ]
        if "bayesian_equivalence_hdi" in stats:
            lines.insert(
                4,
                f"[wf.label]Bayesian HDI:[/wf.label] {stats.get('bayesian_equivalence_hdi', '-')}",
            )
        console.print(result_banner(passed=status == "succeeded", title=title, lines=lines))
        if evidence_bundle is not None:
            _write_proof_evidence_bundle(
                output_dir=evidence_bundle,
                result=result,
                mode="proof" if not demo else "demo",
            )
        if status != "succeeded":
            raise typer.Exit(code=_execution_exit_code(execution_result))
        return

    if result.demo:
        title = (
            "\u2713 SYNTHETIC DEMO: Example parity report"
            if result.passed
            else "\u2717 SYNTHETIC DEMO: Example parity failure"
        )
    elif result.passed:
        title = "\u2713 PASS: Proof-mode parity checks satisfied"
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
    if result.demo:
        lines.insert(
            0,
            "[wf.label]Synthetic provenance:[/wf.label]       demo mode (not proof, not publishable evidence)",
        )
    console.print(result_banner(passed=result.passed, title=title, lines=lines))
    if result.demo:
        console.print("[wf.muted]Results are synthetic (--demo mode)[/wf.muted]")
    else:
        console.print(
            "[wf.muted]Note: local proof-mode runs are not by themselves a public "
            "proof claim. Publish an evidence bundle before making public parity "
            "claims.[/wf.muted]"
        )
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
    """Execute synthetic smoke verify mode for pip-install users."""
    from rich.status import Status

    from worldflux.parity import save_badge
    from worldflux.verify.quick import quick_verify

    console.print(
        key_value_panel(
            {
                "Mode": "synthetic smoke (quick compatibility mode)",
                "Target": target,
                "Env": env,
                "Episodes": str(episodes),
                "Device": device,
            },
            title="Verify - Synthetic Smoke",
            border="wf.border",
        )
    )

    with Status(
        "[wf.brand]Running synthetic smoke verification...[/wf.brand]",
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
            console.print(f"[wf.fail]Synthetic smoke verification failed:[/wf.fail] {exc}")
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
        title = "\u2713 PASS: Synthetic Smoke Baseline Met"
    else:
        title = "\u2717 FAIL: Synthetic Smoke Threshold Not Met"

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
        "[wf.label]Semantics:[/wf.label]        synthetic workload only; not proof-grade evidence",
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
        mode="synthetic-smoke",
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
    if result.demo:
        result_payload["synthetic_provenance"] = {
            "kind": "demo",
            "not_for_proof": True,
            "label": "synthetic demonstration output",
        }
    execution_result = stats.get("execution_result")
    proof_phase = stats.get("execution_phase")
    backend = stats.get("execution_backend")
    profile = stats.get("execution_profile")
    if isinstance(execution_result, dict):
        proof_phase = execution_result.get("proof_phase", proof_phase)
        backend = execution_result.get("backend", backend)
        profile = execution_result.get("profile", profile)
    _write_evidence_manifest(
        output_dir=output_dir,
        mode=mode,
        request_payload={
            "target": result.target,
            "baseline": result.baseline,
            "env": result.env,
            "device": result.stats.get("device"),
            "backend": backend,
            "backend_profile": profile,
            "proof_phase": proof_phase,
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
    """Execute experimental cloud verification mode through WorldFlux Cloud API."""
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
            "\u2713 PASS: Experimental cloud verification passed"
            if passed
            else "\u2717 FAIL: Experimental cloud verification failed"
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
    execution_result = result.stats.get("execution_result")
    if isinstance(execution_result, dict) and not result.demo:
        json_str = json.dumps(execution_result, indent=2)
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json_str + "\n", encoding="utf-8")
            console.print(f"[wf.ok]Results written to:[/wf.ok] {output.resolve()}")
        else:
            console.print(json_str)
        status = str(execution_result.get("status", "failed"))
        if status != "succeeded":
            raise typer.Exit(code=_execution_exit_code(execution_result))
        return

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
    if result.demo:
        payload["synthetic_provenance"] = {
            "kind": "demo",
            "not_for_proof": True,
            "label": "synthetic demonstration output",
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


def _execution_exit_code(payload: dict[str, Any]) -> int:
    status = str(payload.get("status", "failed"))
    if status == "blocked":
        return 2
    if status == "incomplete":
        return 3
    return 1
