"""Backend-neutral parity contracts for proof/parity execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def stable_recipe_hash(recipe: dict[str, Any]) -> str:
    canonical = json.dumps(recipe, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ArtifactManifest:
    backend_kind: str
    adapter_id: str
    recipe_hash: str
    config_snapshot: str | None = None
    command_argv: list[str] = field(default_factory=list)
    source_commit: str | None = None
    checkpoint_paths: list[str] = field(default_factory=list)
    score_paths: list[str] = field(default_factory=list)
    metrics_paths: list[str] = field(default_factory=list)
    eval_protocol_hash: str | None = None
    component_match_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "backend_kind": self.backend_kind,
            "adapter_id": self.adapter_id,
            "recipe_hash": self.recipe_hash,
            "config_snapshot_present": self.config_snapshot is not None,
            "checkpoint_count": len(self.checkpoint_paths),
            "score_count": len(self.score_paths),
            "metrics_count": len(self.metrics_paths),
            "component_match_present": self.component_match_path is not None,
        }


@dataclass(frozen=True)
class BackendRunSpec:
    adapter_id: str
    backend_kind: str
    recipe_hash: str
    command: list[str]
    cwd: str
    env: dict[str, str] = field(default_factory=dict)
    config_snapshot: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_artifacts(
    *,
    run_root: Path,
    backend_kind: str,
    adapter_id: str,
    recipe_hash: str,
    command_argv: list[str],
    source_commit: str | None,
    eval_protocol_hash: str | None,
    config_candidates: tuple[str, ...] = ("config.yaml", "config.json"),
    score_candidates: tuple[str, ...] = ("scores.jsonl", "metrics.jsonl", "eval.csv"),
    checkpoint_suffixes: tuple[str, ...] = (".pkl", ".pt", ".pth", ".ckpt", ".npz"),
) -> ArtifactManifest:
    files = [path for path in run_root.rglob("*") if path.is_file()]
    config_path = next((str(path) for path in files if path.name in config_candidates), None)
    checkpoint_paths = sorted(
        str(path) for path in files if path.suffix.lower() in checkpoint_suffixes
    )
    score_paths = sorted(str(path) for path in files if path.name in score_candidates)
    metrics_paths = sorted(str(path) for path in files if path.name == "metrics.json")
    component_match_path = next(
        (str(path) for path in files if path.name == "component_match_report.json"),
        None,
    )
    return ArtifactManifest(
        backend_kind=backend_kind,
        adapter_id=adapter_id,
        recipe_hash=recipe_hash,
        config_snapshot=config_path,
        command_argv=list(command_argv),
        source_commit=source_commit,
        checkpoint_paths=checkpoint_paths,
        score_paths=score_paths,
        metrics_paths=metrics_paths,
        eval_protocol_hash=eval_protocol_hash,
        component_match_path=component_match_path,
    )


def resolve_latest_checkpoint_dir(ckpt_root: Path) -> Path | None:
    if not ckpt_root.exists():
        return None
    latest_file = ckpt_root / "latest"
    if latest_file.exists() and latest_file.is_file():
        target = latest_file.read_text(encoding="utf-8").strip()
        if target:
            candidate = ckpt_root / target
            if candidate.exists() and candidate.is_dir():
                return candidate
    dirs = [path for path in ckpt_root.iterdir() if path.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda path: path.stat().st_mtime)


__all__ = [
    "ArtifactManifest",
    "BackendRunSpec",
    "discover_artifacts",
    "resolve_latest_checkpoint_dir",
    "stable_recipe_hash",
]
