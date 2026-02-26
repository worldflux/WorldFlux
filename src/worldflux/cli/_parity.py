"""Parity commands: run, aggregate, report, proof-run, proof-report, proof, badge, campaign."""

from __future__ import annotations

import glob as glob_module
import json
import subprocess
import sys
from pathlib import Path

import typer
from rich.box import ROUNDED
from rich.panel import Panel

from worldflux.parity import (
    CampaignRunOptions,
    parse_seed_csv,
    save_badge,
)
from worldflux.parity.errors import ParityError
from worldflux.parity.fmt_utils import fmt_bool as _fmt_bool

from ._app import console, parity_app, parity_campaign_app
from ._rich_output import key_value_panel
from ._theme import PANEL_PADDING

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_parity_script_path(script_name: str) -> Path:
    candidates = (
        Path(__file__).resolve().parents[3] / "scripts" / "parity" / script_name,
        Path.cwd().resolve() / "scripts" / "parity" / script_name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise ParityError(
        f"Unable to locate scripts/parity/{script_name}. "
        "Run this command from a WorldFlux source checkout."
    )


def _run_parity_proof_script(script_name: str, args: list[str]) -> str:
    script_path = _resolve_parity_script_path(script_name)
    completed = subprocess.run(
        [sys.executable, str(script_path), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    if completed.returncode != 0:
        details = "\n".join(part for part in (stdout, stderr) if part)
        raise ParityError(
            f"Proof pipeline step failed ({script_name}, exit={completed.returncode}).\n{details}"
        )
    return stdout


def _resolve_campaign_seeds(
    spec_default: tuple[int, ...], seeds_option: str | None
) -> tuple[int, ...]:
    parsed = parse_seed_csv(seeds_option)
    if parsed:
        return parsed
    if spec_default:
        return spec_default
    raise ParityError("No seeds provided. Pass --seeds or define campaign.default_seeds.")


# ---------------------------------------------------------------------------
# parity run / aggregate / report
# ---------------------------------------------------------------------------


@parity_app.command("run", rich_help_panel="Legacy Harness")
def parity_run(
    suite: Path = typer.Argument(..., help="Path to parity suite specification file."),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Output path for normalized run artifact.",
    ),
    candidate: Path | None = typer.Option(
        None,
        "--candidate",
        help="Override worldflux source path defined in suite file.",
    ),
    candidate_format: str | None = typer.Option(
        None,
        "--candidate-format",
        help="Override worldflux source format defined in suite file.",
    ),
    oracle: Path | None = typer.Option(
        None,
        "--oracle",
        help="Override upstream/oracle source path defined in suite file.",
    ),
    oracle_format: str | None = typer.Option(
        None,
        "--oracle-format",
        help="Override upstream/oracle source format defined in suite file.",
    ),
    upstream_lock: Path | None = typer.Option(
        None,
        "--upstream-lock",
        help="Override upstream lock path used for suite_lock_ref metadata.",
    ),
    enforce: bool = typer.Option(
        False,
        "--enforce/--no-enforce",
        help="Exit non-zero when non-inferiority verdict fails.",
    ),
) -> None:
    """Run legacy non-inferiority parity harness and emit comparison artifact."""
    import worldflux.cli as _cli  # support monkeypatch on cli namespace

    try:
        payload = _cli.run_suite(
            suite,
            output_path=output,
            upstream_path=oracle,
            upstream_format=oracle_format,
            worldflux_path=candidate,
            worldflux_format=candidate_format,
            upstream_lock_path=upstream_lock,
        )
    except (ParityError, ValueError, OSError, json.JSONDecodeError) as exc:
        console.print(f"[wf.fail]Parity run failed:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    suite_meta = payload["suite"]
    stats = payload["stats"]
    passed = bool(stats["pass_non_inferiority"])
    verdict = "[wf.pass]PASS[/wf.pass]" if passed else "[wf.fail]FAIL[/wf.fail]"
    console.print(
        key_value_panel(
            {
                "Mode": "legacy non-inferiority harness",
                "Suite": f"{suite_meta['suite_id']} ({suite_meta['family']})",
                "Samples": str(stats["sample_size"]),
                "Mean drop ratio": f"{stats['mean_drop_ratio']:.6f}",
                "Upper CI (one-sided)": f"{stats['ci_upper_ratio']:.6f}",
                "Margin": f"{stats['margin_ratio']:.6f}",
                "Verdict": verdict,
            },
            title="Parity Run (Legacy)",
            border="wf.border",
        )
    )
    if enforce and not passed:
        raise typer.Exit(code=1)


@parity_app.command("aggregate", rich_help_panel="Legacy Harness")
def parity_aggregate(
    run_paths: list[Path] = typer.Option(
        [],
        "--run",
        help="Run artifact path (repeat to pass multiple files).",
    ),
    runs_glob: str = typer.Option(
        "reports/parity/runs/*.json",
        "--runs-glob",
        help="Glob used when --run is omitted.",
    ),
    output: Path = typer.Option(
        Path("reports/parity/aggregate.json"),
        "--output",
        help="Output path for aggregate artifact.",
    ),
    enforce: bool = typer.Option(
        False,
        "--enforce/--no-enforce",
        help="Exit non-zero when any suite fails aggregate verdict.",
    ),
) -> None:
    """Aggregate legacy parity run artifacts."""
    import worldflux.cli as _cli  # support monkeypatch on cli namespace

    paths = list(run_paths)
    if not paths:
        paths = [Path(path) for path in sorted(glob_module.glob(runs_glob))]
    if not paths:
        console.print("[wf.fail]No run artifacts found to aggregate.[/wf.fail]")
        raise typer.Exit(code=1)

    try:
        payload = _cli.aggregate_runs(paths, output_path=output)
    except (ParityError, ValueError, OSError, json.JSONDecodeError) as exc:
        console.print(f"[wf.fail]Parity aggregate failed:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    passed = bool(payload["all_suites_pass"])
    verdict = "[wf.pass]PASS[/wf.pass]" if passed else "[wf.fail]FAIL[/wf.fail]"
    console.print(
        key_value_panel(
            {
                "Mode": "legacy non-inferiority harness",
                "Runs": str(payload["run_count"]),
                "Suite pass": str(payload["suite_pass_count"]),
                "Suite fail": str(payload["suite_fail_count"]),
                "Verdict": verdict,
                "Written": str(output.resolve()),
            },
            title="Parity Aggregate (Legacy)",
            border="wf.border",
        )
    )
    if enforce and not passed:
        raise typer.Exit(code=1)


@parity_app.command("report", rich_help_panel="Legacy Harness")
def parity_report(
    aggregate: Path = typer.Option(
        Path("reports/parity/aggregate.json"),
        "--aggregate",
        help="Aggregate parity artifact path.",
    ),
    output: Path = typer.Option(
        Path("reports/parity/report.md"),
        "--output",
        help="Markdown report output path.",
    ),
) -> None:
    """Render legacy parity markdown report from aggregate artifact."""
    import worldflux.cli as _cli  # support monkeypatch on cli namespace

    try:
        payload = json.loads(aggregate.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ParityError("aggregate payload must be a JSON object")
        markdown = _cli.render_markdown_report(payload)
    except (ParityError, ValueError, OSError, json.JSONDecodeError) as exc:
        console.print(f"[wf.fail]Parity report failed:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    console.print(f"[wf.ok]Wrote parity report:[/wf.ok] {output.resolve()}")


# ---------------------------------------------------------------------------
# proof-run / proof-report / proof (combined)
# ---------------------------------------------------------------------------


@parity_app.command("proof-run", rich_help_panel="Proof-Grade Pipeline")
def parity_proof_run(
    manifest: Path = typer.Argument(
        ..., help="Proof manifest (parity.manifest.v1 or parity.suite.v2)."
    ),
    run_id: str = typer.Option(
        "",
        "--run-id",
        help="Run identifier. If omitted, scripts/parity/run_parity_matrix.py generates one.",
    ),
    output_dir: Path = typer.Option(
        Path("reports/parity"),
        "--output-dir",
        help="Output directory root for proof artifacts.",
    ),
    device: str = typer.Option("cuda", "--device", help="Execution device."),
    seed_list: str = typer.Option("", "--seed-list", help="Optional seed override, e.g. 0,1,2."),
    max_retries: int = typer.Option(1, "--max-retries", help="Max retries per task/system pair."),
    task_filter: str = typer.Option(
        "",
        "--task-filter",
        help="Comma-separated task filters (supports fnmatch patterns).",
    ),
    shard_index: int = typer.Option(0, "--shard-index", help="Shard index (0-based)."),
    num_shards: int = typer.Option(1, "--num-shards", help="Total shard count."),
    resume: bool = typer.Option(
        True, "--resume/--no-resume", help="Resume from existing parity_runs.jsonl."
    ),
) -> None:
    """Run proof-grade parity matrix execution (official path backed by scripts/parity)."""
    import worldflux.cli as _cli  # support monkeypatch on cli namespace

    args = [
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--max-retries",
        str(max_retries),
        "--shard-index",
        str(shard_index),
        "--num-shards",
        str(num_shards),
    ]
    if run_id.strip():
        args.extend(["--run-id", run_id.strip()])
    if seed_list.strip():
        args.extend(["--seed-list", seed_list.strip()])
    if task_filter.strip():
        args.extend(["--task-filter", task_filter.strip()])
    if resume:
        args.append("--resume")

    try:
        stdout = _cli._run_parity_proof_script("run_parity_matrix.py", args)
    except ParityError as exc:
        console.print(f"[wf.fail]Parity proof-run failed:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    if stdout:
        console.print(stdout)
    console.print(
        key_value_panel(
            {
                "Mode": "proof-grade official equivalence path",
                "Manifest": str(manifest.resolve()),
                "Output dir": str(output_dir.resolve()),
                "Next": "run `worldflux parity proof-report --manifest ... --runs .../parity_runs.jsonl`",
            },
            title="Parity Proof Run",
            border="wf.border.success",
        )
    )


@parity_app.command("proof-report", rich_help_panel="Proof-Grade Pipeline")
def parity_proof_report(
    manifest: Path = typer.Argument(..., help="Proof manifest used for the run."),
    runs: Path = typer.Option(
        ...,
        "--runs",
        help="Path to parity_runs.jsonl produced by proof-run or distributed orchestration.",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        help="Output directory for coverage/equivalence reports (defaults to runs parent).",
    ),
) -> None:
    """Generate proof-grade completeness + equivalence + markdown reports."""
    import worldflux.cli as _cli  # support monkeypatch on cli namespace

    resolved_runs = runs.resolve()
    run_root = resolved_runs.parent
    report_root = output_dir.resolve() if output_dir is not None else run_root
    report_root.mkdir(parents=True, exist_ok=True)

    coverage_report = report_root / "coverage_report.json"
    validity_report = report_root / "validity_report.json"
    equivalence_report = report_root / "equivalence_report.json"
    markdown_report = report_root / "equivalence_report.md"

    seed_plan = run_root / "seed_plan.json"
    run_context = run_root / "run_context.json"
    try:
        coverage_args = [
            "--manifest",
            str(manifest),
            "--runs",
            str(resolved_runs),
            "--output",
            str(coverage_report),
            "--max-missing-pairs",
            "0",
        ]
        if seed_plan.exists():
            coverage_args.extend(["--seed-plan", str(seed_plan)])
        if run_context.exists():
            coverage_args.extend(["--run-context", str(run_context)])
        _cli._run_parity_proof_script("validate_matrix_completeness.py", coverage_args)

        _cli._run_parity_proof_script(
            "stats_equivalence.py",
            [
                "--input",
                str(resolved_runs),
                "--output",
                str(equivalence_report),
                "--manifest",
                str(manifest),
                "--strict-completeness",
                "--strict-validity",
                "--proof-mode",
                "--validity-report",
                str(validity_report),
            ],
        )
        _cli._run_parity_proof_script(
            "report_markdown.py",
            [
                "--input",
                str(equivalence_report),
                "--output",
                str(markdown_report),
            ],
        )
    except ParityError as exc:
        console.print(f"[wf.fail]Parity proof-report failed:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    report_payload = json.loads(equivalence_report.read_text(encoding="utf-8"))
    global_block = report_payload.get("global", {})
    console.print(
        key_value_panel(
            {
                "Mode": "proof-grade official equivalence path",
                "Final verdict": _fmt_bool(global_block.get("parity_pass_final")),
                "Validity pass": _fmt_bool(global_block.get("validity_pass")),
                "Missing pairs": str(global_block.get("missing_pairs", "-")),
                "JSON": str(equivalence_report),
                "Markdown": str(markdown_report),
            },
            title="Parity Proof Report",
            border="wf.border.success",
        )
    )


@parity_app.command("proof", rich_help_panel="Proof-Grade Pipeline")
def parity_proof_combined(
    manifest: Path = typer.Argument(
        ..., help="Proof manifest (parity.manifest.v1 or parity.suite.v2)."
    ),
    device: str = typer.Option("cpu", "--device", help="Execution device."),
    output_dir: Path = typer.Option(
        Path("reports/parity"),
        "--output-dir",
        help="Output directory root for proof artifacts.",
    ),
    seed_list: str = typer.Option("", "--seed-list", help="Optional seed override, e.g. 0,1,2."),
    max_retries: int = typer.Option(1, "--max-retries", help="Max retries per task/system pair."),
    enforce: bool = typer.Option(
        False,
        "--enforce/--no-enforce",
        help="Exit non-zero when final parity verdict fails.",
    ),
) -> None:
    """Run proof-grade parity verification (proof-run + proof-report) in a single step."""
    import worldflux.cli as _cli  # support monkeypatch on cli namespace

    # --- Phase 1: proof-run ---------------------------------------------------
    run_args = [
        "--manifest",
        str(manifest),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--max-retries",
        str(max_retries),
        "--resume",
    ]
    if seed_list.strip():
        run_args.extend(["--seed-list", seed_list.strip()])

    try:
        run_stdout = _cli._run_parity_proof_script("run_parity_matrix.py", run_args)
    except ParityError as exc:
        console.print(f"[wf.fail]Verify proof-run failed:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    if run_stdout:
        console.print(run_stdout)

    console.print(
        key_value_panel(
            {
                "Phase": "1/2: proof-run complete",
                "Manifest": str(manifest.resolve()),
                "Output dir": str(output_dir.resolve()),
            },
            title="Verify - Proof Run",
            border="wf.border",
        )
    )

    # --- Phase 2: proof-report ------------------------------------------------
    resolved_output_dir = output_dir.resolve()
    runs_path = resolved_output_dir / "parity_runs.jsonl"
    if not runs_path.exists():
        console.print(
            f"[wf.fail]Expected run log not found:[/wf.fail] {runs_path}\n"
            "proof-run may have written output to a different location."
        )
        raise typer.Exit(code=1)

    report_root = resolved_output_dir
    report_root.mkdir(parents=True, exist_ok=True)

    coverage_report = report_root / "coverage_report.json"
    validity_report = report_root / "validity_report.json"
    equivalence_report = report_root / "equivalence_report.json"
    markdown_report = report_root / "equivalence_report.md"

    seed_plan = resolved_output_dir / "seed_plan.json"
    run_context = resolved_output_dir / "run_context.json"

    try:
        coverage_args = [
            "--manifest",
            str(manifest),
            "--runs",
            str(runs_path),
            "--output",
            str(coverage_report),
            "--max-missing-pairs",
            "0",
        ]
        if seed_plan.exists():
            coverage_args.extend(["--seed-plan", str(seed_plan)])
        if run_context.exists():
            coverage_args.extend(["--run-context", str(run_context)])
        _cli._run_parity_proof_script("validate_matrix_completeness.py", coverage_args)

        _cli._run_parity_proof_script(
            "stats_equivalence.py",
            [
                "--input",
                str(runs_path),
                "--output",
                str(equivalence_report),
                "--manifest",
                str(manifest),
                "--strict-completeness",
                "--strict-validity",
                "--proof-mode",
                "--validity-report",
                str(validity_report),
            ],
        )
        _cli._run_parity_proof_script(
            "report_markdown.py",
            [
                "--input",
                str(equivalence_report),
                "--output",
                str(markdown_report),
            ],
        )
    except ParityError as exc:
        console.print(f"[wf.fail]Verify proof-report failed:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    # --- Combined summary -----------------------------------------------------
    report_payload = json.loads(equivalence_report.read_text(encoding="utf-8"))
    global_block = report_payload.get("global", {})
    final_pass = bool(global_block.get("parity_pass_final"))
    verdict = "[wf.pass]PASS[/wf.pass]" if final_pass else "[wf.fail]FAIL[/wf.fail]"
    console.print(
        key_value_panel(
            {
                "Mode": "proof-grade official equivalence path",
                "Manifest": str(manifest.resolve()),
                "Device": device,
                "Final verdict": verdict,
                "Validity pass": _fmt_bool(global_block.get("validity_pass")),
                "Missing pairs": str(global_block.get("missing_pairs", "-")),
                "Equivalence JSON": str(equivalence_report),
                "Markdown report": str(markdown_report),
            },
            title="Verify - Combined Summary",
            border="wf.border.success",
        )
    )
    if enforce and not final_pass:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# badge
# ---------------------------------------------------------------------------


@parity_app.command("badge", rich_help_panel="Utilities")
def parity_badge(
    family: str = typer.Option(..., "--family", help="Model family name (e.g. DreamerV3)."),
    passed: bool = typer.Option(True, "--passed/--no-passed", help="Whether parity passed."),
    confidence: float = typer.Option(0.95, "--confidence", help="Confidence level (0-1)."),
    margin: float = typer.Option(0.05, "--margin", help="Margin ratio (0-1)."),
    output: Path = typer.Option(
        Path("parity-badge.svg"),
        "--output",
        help="Output SVG file path.",
    ),
) -> None:
    """Generate a shields.io-style SVG badge for parity proof results."""
    try:
        save_badge(
            path=output,
            family=family,
            passed=passed,
            confidence=confidence,
            margin=margin,
        )
    except OSError as exc:
        console.print(f"[wf.fail]Badge generation failed:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    status = "PASS" if passed else "FAIL"
    console.print(
        key_value_panel(
            {
                "Family": family,
                "Status": status,
                "Confidence": f"{confidence:.0%}",
                "Margin": f"{margin:.0%}",
                "Written": str(output.resolve()),
            },
            title="Parity Badge",
            border="wf.border.success" if passed else "wf.border.error",
        )
    )


# ---------------------------------------------------------------------------
# campaign run / resume / export
# ---------------------------------------------------------------------------


@parity_campaign_app.command("run")
def parity_campaign_run(
    campaign: Path = typer.Argument(..., help="Path to campaign specification file."),
    mode: str = typer.Option(
        "worldflux",
        "--mode",
        help="Execution mode: worldflux | oracle | both.",
    ),
    seeds: str | None = typer.Option(
        None,
        "--seeds",
        help="Comma-separated seeds (e.g. 0,1,2). Uses campaign.default_seeds when omitted.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Device string propagated to command template placeholders.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Override worldflux output path.",
    ),
    oracle_output: Path | None = typer.Option(
        None,
        "--oracle-output",
        help="Override oracle output path.",
    ),
    workdir: Path = typer.Option(
        Path.cwd(),
        "--workdir",
        help="Working directory for command template execution.",
    ),
    pair_output_root: Path | None = typer.Option(
        None,
        "--pair-output-root",
        help="Directory for per-task/seed temporary command outputs.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Reuse existing canonical outputs and skip already available task/seed pairs.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run/--no-dry-run",
        help="Print command templates without executing them.",
    ),
) -> None:
    """Run parity campaign and emit canonical artifacts."""
    import worldflux.cli as _cli  # support monkeypatch on cli namespace

    try:
        spec = _cli.load_campaign_spec(campaign)
        run_options = CampaignRunOptions(
            mode=mode,
            seeds=_resolve_campaign_seeds(spec.default_seeds, seeds),
            device=device,
            output=output.resolve() if output is not None else None,
            oracle_output=oracle_output.resolve() if oracle_output is not None else None,
            resume=resume,
            dry_run=dry_run,
            workdir=workdir.resolve(),
            pair_output_root=pair_output_root.resolve() if pair_output_root is not None else None,
        )
        summary = _cli.run_campaign(spec, run_options)
    except (ParityError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        console.print(f"[wf.fail]Parity campaign failed:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    console.print(
        Panel(
            _cli.render_campaign_summary(summary),
            title="Parity Campaign",
            border_style="wf.border",
            box=ROUNDED,
            padding=PANEL_PADDING,
        )
    )


@parity_campaign_app.command("resume")
def parity_campaign_resume(
    campaign: Path = typer.Argument(..., help="Path to campaign specification file."),
    mode: str = typer.Option(
        "worldflux",
        "--mode",
        help="Execution mode: worldflux | oracle | both.",
    ),
    seeds: str | None = typer.Option(
        None,
        "--seeds",
        help="Comma-separated seeds (e.g. 0,1,2). Uses campaign.default_seeds when omitted.",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Device string propagated to command template placeholders.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Override worldflux output path.",
    ),
    oracle_output: Path | None = typer.Option(
        None,
        "--oracle-output",
        help="Override oracle output path.",
    ),
    workdir: Path = typer.Option(
        Path.cwd(),
        "--workdir",
        help="Working directory for command template execution.",
    ),
    pair_output_root: Path | None = typer.Option(
        None,
        "--pair-output-root",
        help="Directory for per-task/seed temporary command outputs.",
    ),
) -> None:
    """Resume parity campaign generation from existing outputs."""
    import worldflux.cli as _cli  # support monkeypatch on cli namespace

    _cli.parity_campaign_run(
        campaign=campaign,
        mode=mode,
        seeds=seeds,
        device=device,
        output=output,
        oracle_output=oracle_output,
        workdir=workdir,
        pair_output_root=pair_output_root,
        resume=True,
        dry_run=False,
    )


@parity_campaign_app.command("export")
def parity_campaign_export(
    campaign: Path = typer.Argument(..., help="Path to campaign specification file."),
    source: str = typer.Option(
        "worldflux",
        "--source",
        help="Source to export: worldflux | oracle.",
    ),
    seeds: str | None = typer.Option(
        None,
        "--seeds",
        help="Comma-separated seeds (e.g. 0,1,2). Uses campaign.default_seeds when omitted.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Override output path for exported canonical artifact.",
    ),
    resume: bool = typer.Option(
        True,
        "--resume/--no-resume",
        help="Reuse existing canonical output rows when output already exists.",
    ),
) -> None:
    """Export canonical artifact from campaign input source without command execution."""
    import worldflux.cli as _cli  # support monkeypatch on cli namespace

    source_normalized = source.strip().lower()
    if source_normalized not in {"worldflux", "oracle"}:
        console.print("[wf.fail]--source must be one of: worldflux, oracle[/wf.fail]")
        raise typer.Exit(code=1)

    try:
        spec = _cli.load_campaign_spec(campaign)
        resolved_seeds = _resolve_campaign_seeds(spec.default_seeds, seeds)
        summary = _cli.export_campaign_source(
            spec,
            source_name=source_normalized,
            seeds=resolved_seeds,
            output_path=output.resolve() if output is not None else None,
            resume=resume,
        )
    except (ParityError, ValueError, OSError) as exc:
        console.print(f"[wf.fail]Parity campaign export failed:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    console.print(
        Panel(
            _cli.render_campaign_summary(summary),
            title="Parity Campaign Export",
            border_style="wf.border",
            box=ROUNDED,
            padding=PANEL_PADDING,
        )
    )
