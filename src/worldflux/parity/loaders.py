"""Load suite specs and score artifacts for parity evaluation."""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ParityError
from .types import ScorePoint


@dataclass(frozen=True)
class SourceSpec:
    """Source metadata for upstream or candidate score files."""

    path: Path
    format: str
    repo: str | None = None
    commit: str | None = None


@dataclass(frozen=True)
class SuiteSpec:
    """Configuration for one parity benchmark suite."""

    suite_id: str
    family: str
    metric: str
    higher_is_better: bool
    margin_ratio: float
    confidence: float
    tasks: tuple[str, ...]
    upstream: SourceSpec
    worldflux: SourceSpec


def _read_json_or_yaml(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found,import-untyped]
        except ImportError as exc:  # pragma: no cover - depends on optional dependency
            raise ParityError(
                f"Failed to parse suite file {path}. Use JSON-compatible YAML or install pyyaml."
            ) from exc
        loaded = yaml.safe_load(raw)

    if not isinstance(loaded, dict):
        raise ParityError(f"Suite file must decode to object, got {type(loaded).__name__}")
    return loaded


def _parse_source(raw: Any, *, key: str, base_dir: Path) -> SourceSpec:
    if not isinstance(raw, dict):
        raise ParityError(f"suite.{key} must be an object")

    path_raw = raw.get("path")
    if not isinstance(path_raw, str) or not path_raw.strip():
        raise ParityError(f"suite.{key}.path must be non-empty string")

    format_raw = raw.get("format")
    if not isinstance(format_raw, str) or not format_raw.strip():
        raise ParityError(f"suite.{key}.format must be non-empty string")

    source_path = Path(path_raw)
    if not source_path.is_absolute():
        source_path = (base_dir / source_path).resolve()

    repo_raw = raw.get("repo")
    commit_raw = raw.get("commit")
    repo = str(repo_raw) if repo_raw is not None else None
    commit = str(commit_raw) if commit_raw is not None else None

    return SourceSpec(path=source_path, format=format_raw.strip(), repo=repo, commit=commit)


def load_suite_spec(path: Path) -> SuiteSpec:
    """Load and validate a parity suite specification."""
    data = _read_json_or_yaml(path)

    suite_id = str(data.get("suite_id", "")).strip()
    if not suite_id:
        raise ParityError("suite_id is required")

    family = str(data.get("family", "")).strip()
    if not family:
        raise ParityError("family is required")

    metric = str(data.get("metric", "episode_return")).strip() or "episode_return"
    higher_is_better = bool(data.get("higher_is_better", True))
    margin_ratio = float(data.get("margin_ratio", 0.05))
    confidence = float(data.get("confidence", 0.95))

    tasks_raw = data.get("tasks", [])
    if tasks_raw is None:
        tasks_raw = []
    if not isinstance(tasks_raw, list):
        raise ParityError("tasks must be a list of task identifiers")
    tasks = tuple(sorted({str(task).strip() for task in tasks_raw if str(task).strip()}))

    base_dir = path.parent
    upstream = _parse_source(data.get("upstream"), key="upstream", base_dir=base_dir)

    worldflux_raw = data.get("worldflux")
    if worldflux_raw is None:
        raise ParityError("suite.worldflux source must be defined")
    worldflux = _parse_source(worldflux_raw, key="worldflux", base_dir=base_dir)

    return SuiteSpec(
        suite_id=suite_id,
        family=family,
        metric=metric,
        higher_is_better=higher_is_better,
        margin_ratio=margin_ratio,
        confidence=confidence,
        tasks=tasks,
        upstream=upstream,
        worldflux=worldflux,
    )


def _parse_step(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ParityError(f"Invalid step value: {value!r}") from exc


def _parse_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ParityError(f"Invalid score value: {value!r}") from exc
    if score != score:
        raise ParityError("NaN score encountered in parity data")
    return score


def _points_from_canonical_json(path: Path) -> list[ScorePoint]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]]
    if isinstance(payload, dict):
        raw_scores = payload.get("scores", [])
        if not isinstance(raw_scores, list):
            raise ParityError(f"Expected list under {path}:scores")
        rows = [row for row in raw_scores if isinstance(row, dict)]
    elif isinstance(payload, list):
        rows = [row for row in payload if isinstance(row, dict)]
    else:
        raise ParityError(f"Canonical JSON must be object or list, got {type(payload).__name__}")

    points: list[ScorePoint] = []
    for row in rows:
        task = str(row.get("task", "")).strip()
        if not task:
            continue
        seed = int(row.get("seed", 0))
        step = _parse_step(row.get("step", 0))
        score = _parse_score(row.get("score"))
        points.append(ScorePoint(task=task, seed=seed, step=step, score=score))
    return points


def _points_from_canonical_jsonl(path: Path) -> list[ScorePoint]:
    points: list[ScorePoint] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            continue
        task = str(item.get("task", "")).strip()
        if not task:
            continue
        seed = int(item.get("seed", 0))
        step = _parse_step(item.get("step", 0))
        score = _parse_score(item.get("score"))
        points.append(ScorePoint(task=task, seed=seed, step=step, score=score))
    return points


def _points_from_dreamerv3_scores(path: Path) -> list[ScorePoint]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ParityError("DreamerV3 score file must contain a list")

    points: list[ScorePoint] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        task = str(row.get("task", "")).strip()
        if not task:
            continue

        seed = int(row.get("seed", 0))
        xs = row.get("xs")
        ys = row.get("ys")
        if not isinstance(ys, list) or not ys:
            continue

        if isinstance(xs, list) and xs:
            step = _parse_step(xs[-1])
        else:
            step = len(ys) - 1
        score = _parse_score(ys[-1])
        points.append(ScorePoint(task=task, seed=seed, step=step, score=score))
    return points


def _rows_from_tdmpc_csv(path: Path, *, default_task: str | None = None) -> list[ScorePoint]:
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if not fields:
            return []

        task_col = "task" if "task" in fields else None
        score_col = "reward" if "reward" in fields else ("score" if "score" in fields else None)
        step_col = "step" if "step" in fields else ("timesteps" if "timesteps" in fields else None)
        seed_col = "seed" if "seed" in fields else None

        if score_col is None:
            raise ParityError(f"TD-MPC2 CSV missing reward/score column: {path}")

        points: list[ScorePoint] = []
        for row in reader:
            task = str(row.get(task_col, default_task or "")).strip()
            if not task:
                continue
            seed = int(row.get(seed_col, 0) if seed_col else 0)
            step = _parse_step(row.get(step_col, 0) if step_col else 0)
            score = _parse_score(row.get(score_col))
            points.append(ScorePoint(task=task, seed=seed, step=step, score=score))
    return points


def _points_from_tdmpc_results(path: Path) -> list[ScorePoint]:
    if path.is_dir():
        points: list[ScorePoint] = []
        for csv_path in sorted(path.glob("*.csv")):
            points.extend(_rows_from_tdmpc_csv(csv_path, default_task=csv_path.stem))
        return points
    if path.suffix.lower() != ".csv":
        raise ParityError(f"Expected TD-MPC2 csv file or directory, got: {path}")
    return _rows_from_tdmpc_csv(path)


def load_score_points(path: Path, format_name: str) -> list[ScorePoint]:
    """Load score points from supported source formats."""
    if not path.exists():
        raise ParityError(f"Score source does not exist: {path}")

    normalized = format_name.strip().lower()

    if normalized == "canonical_json":
        return _points_from_canonical_json(path)
    if normalized == "canonical_jsonl":
        return _points_from_canonical_jsonl(path)
    if normalized in {"dreamerv3_scores_json", "dreamerv3_scores_json_gz"}:
        return _points_from_dreamerv3_scores(path)
    if normalized in {"tdmpc2_results_csv", "tdmpc2_results_csv_dir"}:
        return _points_from_tdmpc_results(path)

    raise ParityError(f"Unsupported score format: {format_name!r}")
