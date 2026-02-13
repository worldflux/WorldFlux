"""Parity campaign orchestration for repeatable oracle vs WorldFlux experiments."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import resolve_adapter
from .errors import ParityError
from .loaders import load_score_points
from .types import ScorePoint

CAMPAIGN_SCHEMA_VERSION = "worldflux.parity.campaign.v1"
CAMPAIGN_OUTPUT_SCHEMA_VERSION = "worldflux.parity.campaign.output.v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _round_float(value: float) -> float:
    return float(f"{value:.10f}")


def _read_json_or_yaml(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found,import-untyped]
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise ParityError(
                f"Failed to parse campaign file {path}. Use JSON-compatible YAML or install pyyaml."
            ) from exc
        loaded = yaml.safe_load(raw)

    if not isinstance(loaded, dict):
        raise ParityError(f"Campaign file must decode to object, got {type(loaded).__name__}")
    return loaded


@dataclass(frozen=True)
class CampaignSourceSpec:
    """Source configuration for one campaign side (`oracle` or `worldflux`)."""

    name: str
    adapter: str | None
    repo: str | None
    commit: str | None
    input_path: Path | None
    input_format: str | None
    output_path: Path | None
    command_template: str | None
    result_path_template: str | None
    result_format: str | None
    env: dict[str, str]


@dataclass(frozen=True)
class CampaignSpec:
    """Campaign configuration payload."""

    path: Path
    suite_id: str
    family: str
    tasks: tuple[str, ...]
    default_step: int
    default_seeds: tuple[int, ...]
    sources: dict[str, CampaignSourceSpec]

    def source(self, name: str) -> CampaignSourceSpec:
        key = name.strip().lower()
        source = self.sources.get(key)
        if source is None:
            raise ParityError(f"campaign source is missing: {name!r}")
        return source


@dataclass(frozen=True)
class CampaignRunOptions:
    """Runtime options for campaign execution."""

    mode: str
    seeds: tuple[int, ...]
    device: str
    output: Path | None
    oracle_output: Path | None
    resume: bool
    dry_run: bool
    workdir: Path
    pair_output_root: Path | None = None


def parse_seed_csv(value: str | None) -> tuple[int, ...]:
    """Parse comma-separated seeds string to sorted unique tuple."""
    if value is None:
        return ()
    parsed: set[int] = set()
    for token in value.split(","):
        item = token.strip()
        if not item:
            continue
        try:
            parsed.add(int(item))
        except ValueError as exc:
            raise ParityError(f"Invalid seed value: {item!r}") from exc
    return tuple(sorted(parsed))


def _parse_path(raw: Any, *, base_dir: Path, key: str) -> Path | None:
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ParityError(f"{key} must be a non-empty string when provided")
    path = Path(raw.strip())
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _parse_source(
    raw: Any,
    *,
    name: str,
    base_dir: Path,
) -> CampaignSourceSpec:
    if not isinstance(raw, dict):
        raise ParityError(f"campaign.sources.{name} must be an object")

    adapter = raw.get("adapter")
    adapter_value = str(adapter).strip() if adapter is not None else None
    repo_raw = raw.get("repo")
    commit_raw = raw.get("commit")
    repo = str(repo_raw).strip() if repo_raw is not None else None
    commit = str(commit_raw).strip() if commit_raw is not None else None

    input_path = _parse_path(raw.get("input_path"), base_dir=base_dir, key=f"{name}.input_path")
    output_path = _parse_path(raw.get("output_path"), base_dir=base_dir, key=f"{name}.output_path")
    input_format_raw = raw.get("input_format")
    input_format = str(input_format_raw).strip() if input_format_raw is not None else None
    command_template_raw = raw.get("command_template")
    command_template = (
        str(command_template_raw).strip() if command_template_raw is not None else None
    )
    if command_template == "":
        command_template = None
    result_path_template_raw = raw.get("result_path_template")
    result_path_template = (
        str(result_path_template_raw).strip() if result_path_template_raw is not None else None
    )
    if result_path_template == "":
        result_path_template = None
    result_format_raw = raw.get("result_format")
    result_format = str(result_format_raw).strip() if result_format_raw is not None else None

    env_payload = raw.get("env", {})
    if env_payload is None:
        env_payload = {}
    if not isinstance(env_payload, dict):
        raise ParityError(f"campaign.sources.{name}.env must be an object")
    env = {str(key): str(value) for key, value in env_payload.items()}

    return CampaignSourceSpec(
        name=name,
        adapter=adapter_value,
        repo=repo,
        commit=commit,
        input_path=input_path,
        input_format=input_format,
        output_path=output_path,
        command_template=command_template,
        result_path_template=result_path_template,
        result_format=result_format,
        env=env,
    )


def load_campaign_spec(path: Path) -> CampaignSpec:
    """Load and validate campaign specification file."""
    path = path.resolve()
    data = _read_json_or_yaml(path)
    schema_version = str(data.get("schema_version", "")).strip()
    if schema_version and schema_version != CAMPAIGN_SCHEMA_VERSION:
        raise ParityError(
            f"Unsupported campaign schema_version {schema_version!r}, "
            f"expected {CAMPAIGN_SCHEMA_VERSION!r}"
        )

    suite_id = str(data.get("suite_id", "")).strip()
    if not suite_id:
        raise ParityError("campaign.suite_id is required")

    family = str(data.get("family", "")).strip().lower()
    if not family:
        raise ParityError("campaign.family is required")

    tasks_raw = data.get("tasks", [])
    if not isinstance(tasks_raw, list):
        raise ParityError("campaign.tasks must be a list")
    tasks = tuple(sorted({str(task).strip() for task in tasks_raw if str(task).strip()}))
    if not tasks:
        raise ParityError("campaign.tasks must include at least one task")

    default_step = int(data.get("default_step", 0))

    default_seeds_raw = data.get("default_seeds", [])
    if not isinstance(default_seeds_raw, list):
        raise ParityError("campaign.default_seeds must be a list")
    default_seeds = tuple(sorted({int(seed) for seed in default_seeds_raw}))

    sources_raw = data.get("sources")
    if not isinstance(sources_raw, dict):
        raise ParityError("campaign.sources must be an object")

    base_dir = path.parent
    sources: dict[str, CampaignSourceSpec] = {}
    for source_name in ("oracle", "worldflux"):
        source_payload = sources_raw.get(source_name)
        if source_payload is None:
            continue
        sources[source_name] = _parse_source(source_payload, name=source_name, base_dir=base_dir)

    if not sources:
        raise ParityError("campaign.sources must define at least one of: oracle, worldflux")

    return CampaignSpec(
        path=path,
        suite_id=suite_id,
        family=family,
        tasks=tasks,
        default_step=default_step,
        default_seeds=default_seeds,
        sources=sources,
    )


def _canonicalize_points(
    points: list[ScorePoint], *, family: str, source: CampaignSourceSpec
) -> list[ScorePoint]:
    adapter = resolve_adapter(family, source.adapter)
    normalized: list[ScorePoint] = []
    for point in points:
        normalized.append(
            ScorePoint(
                task=adapter.to_canonical_task(point.task),
                seed=int(point.seed),
                step=int(point.step),
                score=float(point.score),
            )
        )
    return normalized


def _dedupe_latest(points: list[ScorePoint]) -> dict[tuple[str, int], ScorePoint]:
    deduped: dict[tuple[str, int], ScorePoint] = {}
    for point in points:
        key = point.key
        current = deduped.get(key)
        if current is None or point.step >= current.step:
            deduped[key] = point
    return deduped


def _load_existing_output(
    path: Path, *, family: str, source: CampaignSourceSpec
) -> dict[tuple[str, int], ScorePoint]:
    if not path.exists():
        return {}
    points = _canonicalize_points(
        load_score_points(path, "canonical_json"),
        family=family,
        source=source,
    )
    return _dedupe_latest(points)


def _resolve_output_path(
    spec: CampaignSpec,
    source_name: str,
    source: CampaignSourceSpec,
    options: CampaignRunOptions,
) -> Path:
    if source_name == "worldflux" and options.output is not None:
        return options.output.resolve()
    if source_name == "oracle" and options.oracle_output is not None:
        return options.oracle_output.resolve()
    if source.output_path is not None:
        return source.output_path
    if source.input_path is not None:
        return source.input_path
    if source_name == "worldflux":
        return (spec.path.parent / f"../../reports/parity/worldflux/{spec.suite_id}.json").resolve()
    return (spec.path.parent / f"../../artifacts/upstream/{spec.suite_id}.json").resolve()


def _resolve_result_path(
    spec: CampaignSpec,
    source: CampaignSourceSpec,
    *,
    task: str,
    seed: int,
    pair_output: Path,
    workdir: Path,
) -> Path:
    if source.result_path_template is None:
        return pair_output

    adapter = resolve_adapter(spec.family, source.adapter)
    context = {
        "suite_id": spec.suite_id,
        "family": spec.family,
        "source": source.name,
        "task": task,
        "adapter_task": adapter.to_adapter_task(task),
        "seed": int(seed),
        "pair_output": str(pair_output),
        "pair_output_dir": str(pair_output.parent),
        "campaign_path": str(spec.path),
        "workdir": str(workdir),
    }
    rendered = source.result_path_template.format(**context)
    path = Path(rendered)
    if not path.is_absolute():
        path = (spec.path.parent / path).resolve()
    return path


def _load_pair_score(
    result_path: Path,
    *,
    result_format: str,
    spec: CampaignSpec,
    source: CampaignSourceSpec,
    task: str,
    seed: int,
) -> ScorePoint:
    if not result_path.exists():
        raise ParityError(f"Command result file does not exist: {result_path}")

    points = _canonicalize_points(
        load_score_points(result_path, result_format),
        family=spec.family,
        source=source,
    )

    candidates = [point for point in points if point.task == task and point.seed == seed]
    if not candidates:
        raise ParityError(
            "No matching task/seed point found in result artifact "
            f"{result_path} for task={task!r}, seed={seed}"
        )
    candidates.sort(key=lambda point: point.step)
    return candidates[-1]


def _run_pair_command(
    spec: CampaignSpec,
    source: CampaignSourceSpec,
    *,
    task: str,
    seed: int,
    device: str,
    workdir: Path,
    pair_output: Path,
    dry_run: bool,
) -> None:
    if source.command_template is None:
        raise ParityError(
            f"campaign source {source.name!r} does not define command_template "
            "and cannot execute task/seed generation"
        )

    adapter = resolve_adapter(spec.family, source.adapter)
    context = {
        "suite_id": spec.suite_id,
        "family": spec.family,
        "source": source.name,
        "task": task,
        "adapter_task": adapter.to_adapter_task(task),
        "seed": int(seed),
        "device": device,
        "pair_output": str(pair_output),
        "pair_output_dir": str(pair_output.parent),
        "campaign_path": str(spec.path),
        "workdir": str(workdir),
    }
    command = source.command_template.format(**context)
    rendered_env = os.environ.copy()
    rendered_env.update(source.env)
    pair_output.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"[campaign:dry-run] {command}")
        return

    subprocess.run(
        command,
        cwd=workdir,
        env=rendered_env,
        shell=True,
        check=True,
    )


def _write_canonical_output(
    *,
    spec: CampaignSpec,
    source_name: str,
    output_path: Path,
    selected: dict[tuple[str, int], ScorePoint],
    seeds: tuple[int, ...],
) -> None:
    rows = [
        {
            "task": point.task,
            "seed": int(point.seed),
            "step": int(point.step),
            "score": _round_float(point.score),
        }
        for _, point in sorted(
            selected.items(), key=lambda item: (item[0][0], item[0][1], item[1].step)
        )
    ]
    payload = {
        "schema_version": CAMPAIGN_OUTPUT_SCHEMA_VERSION,
        "generated_at_utc": _utc_now_iso(),
        "suite_id": spec.suite_id,
        "family": spec.family,
        "source": source_name,
        "seeds": list(seeds),
        "tasks": list(spec.tasks),
        "score_count": len(rows),
        "scores": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_from_source_artifact(
    spec: CampaignSpec,
    source: CampaignSourceSpec,
    *,
    seeds: tuple[int, ...],
) -> dict[tuple[str, int], ScorePoint]:
    if source.input_path is None or source.input_format is None:
        raise ParityError(
            f"campaign source {source.name!r} must define input_path/input_format when "
            "command_template is not used"
        )
    points = _canonicalize_points(
        load_score_points(source.input_path, source.input_format),
        family=spec.family,
        source=source,
    )
    selected: dict[tuple[str, int], ScorePoint] = {}
    allowed_tasks = set(spec.tasks)
    allowed_seeds = set(seeds)
    for point in points:
        if point.task not in allowed_tasks or point.seed not in allowed_seeds:
            continue
        current = selected.get(point.key)
        if current is None or point.step >= current.step:
            selected[point.key] = point
    return selected


def run_campaign(spec: CampaignSpec, options: CampaignRunOptions) -> dict[str, Any]:
    """Execute campaign generation for oracle/worldflux outputs."""
    mode = options.mode.strip().lower()
    if mode not in {"worldflux", "oracle", "both"}:
        raise ParityError("mode must be one of: worldflux, oracle, both")

    selected_modes = ("oracle", "worldflux") if mode == "both" else (mode,)
    seeds = options.seeds or spec.default_seeds
    if not seeds:
        raise ParityError("No seeds provided. Pass --seeds or define campaign.default_seeds.")
    if any(seed < 0 for seed in seeds):
        raise ParityError("seeds must be non-negative integers")

    summary_rows: list[dict[str, Any]] = []
    for source_name in selected_modes:
        source = spec.source(source_name)
        output_path = _resolve_output_path(spec, source_name, source, options)
        selected = (
            _load_existing_output(output_path, family=spec.family, source=source)
            if options.resume
            else {}
        )

        if source.command_template is not None:
            pair_root = options.pair_output_root
            if pair_root is None:
                pair_root = (spec.path.parent / "../../reports/parity/tmp").resolve()
            pair_dir = (pair_root / spec.suite_id / source_name).resolve()

            result_format = (
                source.result_format
                or source.input_format
                or resolve_adapter(spec.family, source.adapter).default_result_format(source_name)
                or "canonical_json"
            )

            for task in spec.tasks:
                for seed in seeds:
                    key = (task, seed)
                    if options.resume and key in selected:
                        continue
                    safe_task = task.replace("/", "_")
                    pair_output = pair_dir / f"{safe_task}__seed{seed}.json"
                    _run_pair_command(
                        spec,
                        source,
                        task=task,
                        seed=seed,
                        device=options.device,
                        workdir=options.workdir,
                        pair_output=pair_output,
                        dry_run=options.dry_run,
                    )
                    if options.dry_run:
                        continue
                    result_path = _resolve_result_path(
                        spec,
                        source,
                        task=task,
                        seed=seed,
                        pair_output=pair_output,
                        workdir=options.workdir,
                    )
                    point = _load_pair_score(
                        result_path,
                        result_format=result_format,
                        spec=spec,
                        source=source,
                        task=task,
                        seed=seed,
                    )
                    selected[key] = point
        else:
            loaded = _load_from_source_artifact(spec, source, seeds=seeds)
            for key, point in loaded.items():
                current = selected.get(key)
                if current is None or point.step >= current.step:
                    selected[key] = point

        expected_pairs = {(task, seed) for task in spec.tasks for seed in seeds}
        missing_pairs = sorted(expected_pairs - set(selected))
        if missing_pairs:
            preview = ", ".join(f"{task}:{seed}" for task, seed in missing_pairs[:5])
            raise ParityError(
                f"campaign source {source_name!r} is missing {len(missing_pairs)} task/seed pairs "
                f"(first: {preview})"
            )

        if not options.dry_run:
            _write_canonical_output(
                spec=spec,
                source_name=source_name,
                output_path=output_path,
                selected=selected,
                seeds=seeds,
            )

        summary_rows.append(
            {
                "source": source_name,
                "output_path": str(output_path),
                "score_count": len(selected),
                "task_count": len(spec.tasks),
                "seed_count": len(seeds),
                "resume": options.resume,
                "dry_run": options.dry_run,
            }
        )

    return {
        "schema_version": "worldflux.parity.campaign.summary.v1",
        "generated_at_utc": _utc_now_iso(),
        "suite_id": spec.suite_id,
        "family": spec.family,
        "mode": mode,
        "seeds": list(seeds),
        "rows": summary_rows,
    }


def export_campaign_source(
    spec: CampaignSpec,
    *,
    source_name: str,
    seeds: tuple[int, ...],
    output_path: Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Export filtered canonical artifact for a single campaign source."""
    options = CampaignRunOptions(
        mode=source_name,
        seeds=seeds,
        device="cpu",
        output=output_path if source_name == "worldflux" else None,
        oracle_output=output_path if source_name == "oracle" else None,
        resume=resume,
        dry_run=False,
        workdir=Path.cwd(),
        pair_output_root=None,
    )
    return run_campaign(spec, options)


def render_campaign_summary(summary: dict[str, Any]) -> str:
    """Render text summary for CLI and script output."""
    rows = summary.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    lines = [
        f"Suite: {summary.get('suite_id', 'unknown')} ({summary.get('family', 'unknown')})",
        f"Mode: {summary.get('mode', 'unknown')}",
        f"Seeds: {','.join(str(seed) for seed in summary.get('seeds', []))}",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            " - {source}: scores={scores} tasks={tasks} seeds={seeds} output={output}".format(
                source=row.get("source", "unknown"),
                scores=row.get("score_count", 0),
                tasks=row.get("task_count", 0),
                seeds=row.get("seed_count", 0),
                output=row.get("output_path", "unknown"),
            )
        )
    return "\n".join(lines)


def build_command_preview(command: str) -> str:
    """Return shell command preview with safe quoting for logs."""
    return " ".join(shlex.quote(part) for part in shlex.split(command))
