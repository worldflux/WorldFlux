"""Backend adapter registry for proof/parity execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .backend_contract import (
    ArtifactManifest,
    BackendRunSpec,
    discover_artifacts,
    resolve_latest_checkpoint_dir,
    stable_recipe_hash,
)


class BackendAdapter(Protocol):
    adapter_id: str
    backend_kind: str

    def prepare_run(
        self,
        *,
        recipe: dict[str, Any],
        env_spec: dict[str, Any],
        seed: int,
        run_dir: Path,
        repo_root: Path,
        python_executable: str,
        device: str,
    ) -> BackendRunSpec: ...

    def collect_artifacts(
        self,
        *,
        run_dir: Path,
        source_commit: str | None = None,
        eval_protocol_hash: str | None = None,
        command_argv: list[str] | None = None,
        recipe: dict[str, Any] | None = None,
    ) -> ArtifactManifest: ...

    def monitor_run(self, *, run_dir: Path) -> dict[str, Any]: ...

    def artifact_requirements(self) -> dict[str, Any]: ...


@dataclass
class BackendAdapterRegistry:
    _adapters: dict[str, BackendAdapter]

    def __init__(self) -> None:
        self._adapters = {}

    def register(self, adapter: BackendAdapter) -> None:
        self._adapters[str(adapter.adapter_id)] = adapter

    def get(self, adapter_id: str) -> BackendAdapter | None:
        return self._adapters.get(str(adapter_id))

    def require(self, adapter_id: str) -> BackendAdapter:
        adapter = self.get(adapter_id)
        if adapter is None:
            raise KeyError(f"Unknown backend adapter: {adapter_id}")
        return adapter

    def list_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


class DreamerOfficialJAXSubprocessAdapter:
    adapter_id = "official_dreamerv3_jax_subprocess"
    backend_kind = "jax_subprocess"

    def prepare_run(
        self,
        *,
        recipe: dict[str, Any],
        env_spec: dict[str, Any],
        seed: int,
        run_dir: Path,
        repo_root: Path,
        python_executable: str,
        device: str,
    ) -> BackendRunSpec:
        recipe_hash = stable_recipe_hash(recipe)
        jax_platform = "cpu" if str(device).lower() == "cpu" else "cuda"
        command = [
            python_executable,
            "dreamerv3/main.py",
            "--logdir",
            str((run_dir / "dreamerv3_logdir").resolve()),
            "--configs",
            "atari100k",
            "--task",
            str(env_spec["task_id"]),
            "--seed",
            str(int(seed)),
            "--run.steps",
            str(int(recipe["steps"])),
            "--jax.platform",
            jax_platform,
            "--logger.outputs",
            "jsonl",
        ]
        return BackendRunSpec(
            adapter_id=self.adapter_id,
            backend_kind=self.backend_kind,
            recipe_hash=recipe_hash,
            command=command,
            cwd=str(repo_root.resolve()),
        )

    def collect_artifacts(
        self,
        *,
        run_dir: Path,
        source_commit: str | None = None,
        eval_protocol_hash: str | None = None,
        command_argv: list[str] | None = None,
        recipe: dict[str, Any] | None = None,
    ) -> ArtifactManifest:
        manifest = discover_artifacts(
            run_root=run_dir,
            backend_kind=self.backend_kind,
            adapter_id=self.adapter_id,
            recipe_hash=stable_recipe_hash(dict(recipe or {})),
            command_argv=list(command_argv or []),
            source_commit=source_commit,
            eval_protocol_hash=eval_protocol_hash,
        )
        latest_ckpt_dir = resolve_latest_checkpoint_dir(run_dir / "dreamerv3_logdir" / "ckpt")
        checkpoint_paths = list(manifest.checkpoint_paths)
        if latest_ckpt_dir is not None:
            checkpoint_paths = sorted(
                {*checkpoint_paths, *(str(path) for path in latest_ckpt_dir.rglob("*.pkl"))}
            )
        return ArtifactManifest(
            **{
                **manifest.to_dict(),
                "checkpoint_paths": checkpoint_paths,
            }
        )

    def monitor_run(self, *, run_dir: Path) -> dict[str, Any]:
        logdir = run_dir / "dreamerv3_logdir"
        scores = logdir / "scores.jsonl"
        latest_ckpt_dir = resolve_latest_checkpoint_dir(logdir / "ckpt")
        return {
            "backend_kind": self.backend_kind,
            "adapter_id": self.adapter_id,
            "config_present": (logdir / "config.yaml").exists(),
            "scores_present": scores.exists(),
            "scores_lines": len(scores.read_text(encoding="utf-8").splitlines())
            if scores.exists()
            else 0,
            "latest_checkpoint_dir": str(latest_ckpt_dir) if latest_ckpt_dir is not None else None,
            "agent_present": bool(latest_ckpt_dir and (latest_ckpt_dir / "agent.pkl").exists()),
        }

    def artifact_requirements(self) -> dict[str, Any]:
        return {
            "config_snapshot": ["dreamerv3_logdir/config.yaml"],
            "score_paths": ["dreamerv3_logdir/scores.jsonl"],
            "metrics_paths": ["dreamerv3_logdir/metrics.jsonl"],
            "checkpoint_paths": [
                "dreamerv3_logdir/ckpt/*/agent.pkl",
                "dreamerv3_logdir/ckpt/*/step.pkl",
            ],
        }


class DreamerWorldFluxJAXSubprocessAdapter:
    adapter_id = "worldflux_dreamerv3_jax_subprocess"
    backend_kind = "jax_subprocess"

    def prepare_run(
        self,
        *,
        recipe: dict[str, Any],
        env_spec: dict[str, Any],
        seed: int,
        run_dir: Path,
        repo_root: Path,
        python_executable: str,
        device: str,
    ) -> BackendRunSpec:
        effective_recipe = {
            **recipe,
            "steps": int(recipe.get("steps", 110_000)),
        }
        recipe_hash = stable_recipe_hash(effective_recipe)
        eval_protocol = dict(env_spec.get("eval_protocol", {}))
        command = [
            python_executable,
            str(
                (
                    repo_root / "scripts" / "parity" / "wrappers" / "worldflux_dreamerv3_jax.py"
                ).resolve()
            ),
            "--repo-root",
            str(repo_root.resolve()),
            "--task-id",
            str(env_spec["task_id"]),
            "--seed",
            str(int(seed)),
            "--steps",
            str(int(effective_recipe["steps"])),
            "--device",
            str(device),
            "--run-dir",
            str(run_dir.resolve()),
            "--metrics-out",
            str((run_dir / "metrics.json").resolve()),
            "--eval-window",
            str(int(eval_protocol.get("eval_window", 10))),
            "--eval-interval",
            str(int(eval_protocol.get("eval_interval", 5_000))),
            "--eval-episodes",
            str(int(eval_protocol.get("eval_episodes", 1))),
        ]
        return BackendRunSpec(
            adapter_id=self.adapter_id,
            backend_kind=self.backend_kind,
            recipe_hash=recipe_hash,
            command=command,
            cwd=str(repo_root.resolve()),
        )

    def collect_artifacts(
        self,
        *,
        run_dir: Path,
        source_commit: str | None = None,
        eval_protocol_hash: str | None = None,
        command_argv: list[str] | None = None,
        recipe: dict[str, Any] | None = None,
    ) -> ArtifactManifest:
        manifest = discover_artifacts(
            run_root=run_dir,
            backend_kind=self.backend_kind,
            adapter_id=self.adapter_id,
            recipe_hash=stable_recipe_hash(dict(recipe or {})),
            command_argv=list(command_argv or []),
            source_commit=source_commit,
            eval_protocol_hash=eval_protocol_hash,
        )
        latest_ckpt_dir = resolve_latest_checkpoint_dir(run_dir / "dreamerv3_logdir" / "ckpt")
        checkpoint_paths = list(manifest.checkpoint_paths)
        if latest_ckpt_dir is not None:
            checkpoint_paths = sorted(
                {*checkpoint_paths, *(str(path) for path in latest_ckpt_dir.rglob("*.pkl"))}
            )
        return ArtifactManifest(
            **{
                **manifest.to_dict(),
                "checkpoint_paths": checkpoint_paths,
            }
        )

    def monitor_run(self, *, run_dir: Path) -> dict[str, Any]:
        logdir = run_dir / "dreamerv3_logdir"
        scores = logdir / "scores.jsonl"
        latest_ckpt_dir = resolve_latest_checkpoint_dir(logdir / "ckpt")
        return {
            "backend_kind": self.backend_kind,
            "adapter_id": self.adapter_id,
            "config_present": (logdir / "config.yaml").exists(),
            "scores_present": scores.exists(),
            "scores_lines": len(scores.read_text(encoding="utf-8").splitlines())
            if scores.exists()
            else 0,
            "latest_checkpoint_dir": str(latest_ckpt_dir) if latest_ckpt_dir is not None else None,
            "agent_present": bool(latest_ckpt_dir and (latest_ckpt_dir / "agent.pkl").exists()),
        }

    def artifact_requirements(self) -> dict[str, Any]:
        return {
            "config_snapshot": ["dreamerv3_logdir/config.yaml"],
            "score_paths": ["dreamerv3_logdir/scores.jsonl"],
            "metrics_paths": ["dreamerv3_logdir/metrics.jsonl"],
            "checkpoint_paths": [
                "dreamerv3_logdir/ckpt/*/agent.pkl",
                "dreamerv3_logdir/ckpt/*/step.pkl",
            ],
        }


class TDMPC2OfficialTorchSubprocessAdapter:
    adapter_id = "official_tdmpc2_torch_subprocess"
    backend_kind = "torch_subprocess"

    def prepare_run(
        self,
        *,
        recipe: dict[str, Any],
        env_spec: dict[str, Any],
        seed: int,
        run_dir: Path,
        repo_root: Path,
        python_executable: str,
        device: str,
    ) -> BackendRunSpec:
        recipe_hash = stable_recipe_hash(recipe)
        model_size = int(recipe.get("model_size", 5))
        eval_episodes = int(env_spec.get("eval_protocol", {}).get("eval_episodes", 10))
        eval_interval = int(env_spec.get("eval_protocol", {}).get("eval_interval", 50_000))
        command = [
            python_executable,
            "tdmpc2/train.py",
            f"task={env_spec['task_id']}",
            f"steps={int(recipe['steps'])}",
            f"seed={int(seed)}",
            f"model_size={model_size}",
            f"eval_episodes={eval_episodes}",
            f"eval_freq={eval_interval}",
            "enable_wandb=false",
            "save_csv=true",
            "save_video=false",
            "save_agent=false",
            "compile=false",
            "hydra/launcher=basic",
            f"exp_name={run_dir.name}",
        ]
        return BackendRunSpec(
            adapter_id=self.adapter_id,
            backend_kind=self.backend_kind,
            recipe_hash=recipe_hash,
            command=command,
            cwd=str(repo_root.resolve()),
        )

    def collect_artifacts(
        self,
        *,
        run_dir: Path,
        source_commit: str | None = None,
        eval_protocol_hash: str | None = None,
        command_argv: list[str] | None = None,
        recipe: dict[str, Any] | None = None,
    ) -> ArtifactManifest:
        return discover_artifacts(
            run_root=run_dir,
            backend_kind=self.backend_kind,
            adapter_id=self.adapter_id,
            recipe_hash=stable_recipe_hash(dict(recipe or {})),
            command_argv=list(command_argv or []),
            source_commit=source_commit,
            eval_protocol_hash=eval_protocol_hash,
            score_candidates=("eval.csv",),
        )

    def monitor_run(self, *, run_dir: Path) -> dict[str, Any]:
        eval_csv = next(iter(sorted(run_dir.rglob("eval.csv"))), None)
        latest_ckpt = next(iter(sorted(run_dir.rglob("*.pt"))), None)
        return {
            "backend_kind": self.backend_kind,
            "adapter_id": self.adapter_id,
            "eval_csv_present": eval_csv is not None,
            "checkpoint_present": latest_ckpt is not None,
            "eval_csv_path": str(eval_csv) if eval_csv is not None else None,
            "checkpoint_path": str(latest_ckpt) if latest_ckpt is not None else None,
        }

    def artifact_requirements(self) -> dict[str, Any]:
        return {
            "score_paths": ["**/eval.csv"],
            "metrics_paths": ["**/metrics.json"],
            "checkpoint_paths": ["**/*.pt"],
        }


_DEFAULT_REGISTRY = BackendAdapterRegistry()
_DEFAULT_REGISTRY.register(DreamerOfficialJAXSubprocessAdapter())
_DEFAULT_REGISTRY.register(DreamerWorldFluxJAXSubprocessAdapter())
_DEFAULT_REGISTRY.register(TDMPC2OfficialTorchSubprocessAdapter())


def get_backend_adapter_registry() -> BackendAdapterRegistry:
    return _DEFAULT_REGISTRY


__all__ = [
    "BackendAdapter",
    "BackendAdapterRegistry",
    "DreamerOfficialJAXSubprocessAdapter",
    "DreamerWorldFluxJAXSubprocessAdapter",
    "get_backend_adapter_registry",
    "TDMPC2OfficialTorchSubprocessAdapter",
]
