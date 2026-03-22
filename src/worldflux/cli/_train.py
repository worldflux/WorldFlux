# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""The ``train`` command."""

from __future__ import annotations

import hashlib
import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import typer

from worldflux.config_loader import load_config
from worldflux.core.backend_bridge import canonical_backend_profile

from ._app import app, console
from ._rich_output import key_value_panel, result_banner


@dataclass(frozen=True)
class ScaffoldRuntime:
    dataset_module: ModuleType
    dashboard_module: ModuleType | None
    dashboard_root: Path | None


def _normalize_degraded_modes(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        key = str(value).strip().lower()
        if key and key not in normalized:
            normalized.append(key)
    return normalized


def _training_support_surface(model_id: str, backend: str) -> str:
    from worldflux.factory import get_model_info

    if str(backend).strip().lower() != "native_torch":
        return "advanced"

    try:
        info = get_model_info(model_id)
    except ValueError:
        return "internal"
    return str(info.get("support_tier", "internal")).strip().lower() or "internal"


def _classify_local_run(*, support_surface: str, data_mode: str, degraded_modes: list[str]) -> str:
    if support_surface == "advanced":
        return "advanced_evidence"
    if support_surface == "supported" and data_mode in {"offline", "online"} and not degraded_modes:
        return "meaningful_local_training"
    return "contract_smoke"


def _env_to_task_filter(env: str) -> str:
    value = str(env).strip().lower()
    if not value:
        return ""
    if value.startswith("atari/"):
        game = value.split("/", 1)[1].strip().replace("-", "_")
        return f"atari100k_{game}" if game else ""
    if value.startswith("dmcontrol/"):
        return value.split("/", 1)[1].strip().replace("/", "-")
    if value.startswith("mujoco/"):
        return value.split("/", 1)[1].strip().replace("_", "-")
    return value


def _degraded_guidance_lines(cfg: Any, degraded_modes: list[str]) -> list[str]:
    normalized = {str(mode).strip().lower() for mode in degraded_modes}
    if "env_collection_unavailable" not in normalized:
        return []

    environment = str(getattr(cfg, "environment", "")).strip().lower()
    if environment == "atari":
        return [
            "Cause: live Atari environment collection was unavailable, so training fell back to smoke-only data.",
            "Next: install `gymnasium[atari]` and `ale-py`, or run `uv sync --extra training --extra atari` before retrying.",
        ]
    if environment == "mujoco":
        return [
            "Cause: MuJoCo environment collection was unavailable, so training fell back to smoke-only data.",
            "Next: install `gymnasium[mujoco]` and `mujoco`, or run `uv sync --extra training --extra mujoco` before retrying.",
        ]
    return [
        "Cause: environment-backed data collection was unavailable, so training fell back to smoke-only data.",
        "Next: install the required environment extras and rerun before treating this as meaningful local training.",
    ]


def _load_module_from_file(path: Path, *, namespace: str) -> ModuleType:
    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    module_name = f"worldflux_cli_{namespace}_{digest}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_scaffold_runtime(project_root: Path) -> ScaffoldRuntime | None:
    dataset_path = project_root / "dataset.py"
    if not dataset_path.exists():
        return None

    dataset_module = _load_module_from_file(dataset_path, namespace="dataset")

    dashboard_path = project_root / "local_dashboard.py"
    dashboard_root = project_root / "dashboard"
    dashboard_module: ModuleType | None = None
    if dashboard_path.exists() and (dashboard_root / "index.html").exists():
        dashboard_module = _load_module_from_file(dashboard_path, namespace="dashboard")

    return ScaffoldRuntime(
        dataset_module=dataset_module,
        dashboard_module=dashboard_module,
        dashboard_root=dashboard_root if dashboard_module is not None else None,
    )


def _prepare_scaffold_dashboard(
    *,
    cfg: Any,
    effective_steps: int,
    effective_output_dir: str,
    runtime: ScaffoldRuntime,
) -> tuple[dict[str, Any] | None, list[Any]]:
    dashboard_module = runtime.dashboard_module
    dashboard_root = runtime.dashboard_root
    if dashboard_module is None or dashboard_root is None or not cfg.visualization.enabled:
        return None, []

    dashboard_buffer = dashboard_module.MetricBuffer(
        max_points=cfg.visualization.history_max_points
    )
    if hasattr(dashboard_buffer, "set_target_steps"):
        dashboard_buffer.set_target_steps(effective_steps)

    gameplay_buffer = None
    gameplay_enabled = bool(cfg.gameplay.enabled)
    unavailable_detail = (
        "Gameplay stream disabled in worldflux.toml." if not gameplay_enabled else None
    )

    if gameplay_enabled:
        gameplay_buffer = dashboard_module.GameplayBuffer(
            max_frames=cfg.gameplay.max_frames,
            fps=cfg.gameplay.fps,
        )
        if hasattr(dashboard_buffer, "set_gameplay_available"):
            dashboard_buffer.set_gameplay_available(True)
    else:
        if hasattr(dashboard_buffer, "set_gameplay_available"):
            dashboard_buffer.set_gameplay_available(False)
        if hasattr(dashboard_buffer, "set_phase"):
            dashboard_buffer.set_phase("unavailable", unavailable_detail)

    dashboard_server = dashboard_module.MetricsDashboardServer(
        metric_buffer=dashboard_buffer,
        gameplay_buffer=gameplay_buffer,
        host=cfg.visualization.host,
        start_port=cfg.visualization.port,
        dashboard_root=dashboard_root,
        refresh_ms=cfg.visualization.refresh_ms,
        max_port_tries=100,
    )
    dashboard_server.start()
    console.print(f"[wf.info]Dashboard:[/wf.info] {dashboard_server.url}")
    if cfg.visualization.open_browser and hasattr(dashboard_server, "open_browser"):
        dashboard_server.open_browser()

    dashboard_callback = dashboard_module.DashboardCallback(
        dashboard_buffer,
        Path(effective_output_dir) / "metrics.jsonl",
    )
    phase_events: list[tuple[str, str | None]] = []

    def publish_phase(phase: str, detail: str | None = None) -> None:
        phase_events.append((phase, detail))
        if hasattr(dashboard_buffer, "set_phase"):
            dashboard_buffer.set_phase(phase, detail)
        if gameplay_buffer is not None:
            if hasattr(gameplay_buffer, "set_phase"):
                gameplay_buffer.set_phase(phase, detail)
            if hasattr(gameplay_buffer, "set_status"):
                if phase == "unavailable":
                    gameplay_buffer.set_status("unavailable")
                elif phase in {"finished", "error"}:
                    gameplay_buffer.set_status(phase)
                else:
                    gameplay_buffer.set_status("running")
        elif hasattr(dashboard_buffer, "set_gameplay_available") and phase == "unavailable":
            dashboard_buffer.set_gameplay_available(False)

    def publish_frame(
        frame: Any,
        episode: int,
        episode_step: int,
        reward: float,
        done: bool,
    ) -> None:
        if gameplay_buffer is None or not hasattr(gameplay_buffer, "append_frame"):
            return
        gameplay_buffer.append_frame(
            frame,
            episode=episode,
            episode_step=episode_step,
            reward=reward,
            done=done,
        )

    return {
        "dashboard_buffer": dashboard_buffer,
        "gameplay_buffer": gameplay_buffer,
        "dashboard_server": dashboard_server,
        "dashboard_callback": dashboard_callback,
        "publish_phase": publish_phase,
        "publish_frame": publish_frame,
        "unavailable_detail": unavailable_detail,
        "phase_events": phase_events,
    }, [dashboard_callback]


@app.command(rich_help_panel="Training")
def train(
    config: Path = typer.Option(
        Path("worldflux.toml"),
        "--config",
        "-c",
        help="Path to worldflux.toml configuration file.",
    ),
    steps: int | None = typer.Option(
        None,
        "--steps",
        help="Override total training steps from config.",
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="Override device from config (cpu, cuda, auto).",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        help="Override output directory from config.",
    ),
    resume_from: str | None = typer.Option(
        None,
        "--resume-from",
        help="Path to checkpoint to resume training from.",
    ),
    cloud: bool = typer.Option(
        False,
        "--cloud",
        help="Submit an experimental cloud training job instead of running local training.",
    ),
    gpu: str | None = typer.Option(
        None,
        "--gpu",
        help="Cloud GPU override (for example: a10g, a100, h100).",
    ),
    override: list[Path] | None = typer.Option(
        None,
        "--override",
        help="Path(s) to YAML/JSON override file(s), applied in order on top of base config.",
    ),
    use_ddp: bool = typer.Option(
        False,
        "--use-ddp",
        help="Enable DistributedDataParallel multi-GPU training.",
    ),
    num_envs: int = typer.Option(
        1,
        "--num-envs",
        help="Number of parallel environments for vectorized data collection.",
    ),
    env_mode: str = typer.Option(
        "auto",
        "--env-mode",
        help="Vectorized env mode: sync, async, or auto.",
    ),
    profile: bool = typer.Option(
        False,
        "--profile",
        help="Enable performance profiling with torch.profiler trace output.",
    ),
) -> None:
    """Train a world model using worldflux.toml configuration.

    [dim]Examples:[/dim]
      worldflux train
      worldflux train --steps 50000 --device cuda
      worldflux train --cloud --gpu a100
      worldflux train --override experiment.yaml
    """
    from worldflux.core.backend_handle import OfficialBackendHandle
    from worldflux.factory import create_world_model
    from worldflux.training import Trainer, TrainingConfig
    from worldflux.training.data import create_random_buffer

    try:
        cfg = load_config(config)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[wf.fail]Configuration error:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    # Apply CLI overrides
    effective_steps = steps if steps is not None else cfg.training.total_steps
    effective_device = device if device is not None else cfg.training.device
    effective_output_dir = output_dir if output_dir is not None else cfg.training.output_dir
    effective_backend_profile = str(
        cfg.training.backend_profile
    ).strip() or canonical_backend_profile(cfg.model, cfg.training.backend)

    if cloud:
        from worldflux.cloud import ModalBackend, WorldFluxCloudClient

        client = WorldFluxCloudClient.from_env()
        if not client.api_key:
            console.print(
                "[wf.fail]Cloud auth missing:[/wf.fail] run `worldflux login --api-key <key>` "
                "or set WORLDFLUX_CLOUD_API_KEY."
            )
            raise typer.Exit(code=1)

        cloud_gpu = gpu if gpu is not None else cfg.cloud.gpu_type
        cloud_region = cfg.cloud.region
        payload = {
            "project_name": cfg.project_name,
            "model": cfg.model,
            "architecture": {
                "obs_shape": list(cfg.architecture.obs_shape),
                "action_dim": cfg.architecture.action_dim,
                "hidden_dim": cfg.architecture.hidden_dim,
            },
            "training": {
                "total_steps": effective_steps,
                "batch_size": cfg.training.batch_size,
                "sequence_length": cfg.training.sequence_length,
                "learning_rate": cfg.training.learning_rate,
                "output_dir": effective_output_dir,
            },
            "cloud": {
                "gpu_type": cloud_gpu,
                "spot": cfg.cloud.spot,
                "region": cloud_region,
                "timeout_hours": cfg.cloud.timeout_hours,
            },
            "flywheel": {
                "opt_in": cfg.flywheel.opt_in,
                "privacy_epsilon": cfg.flywheel.privacy_epsilon,
                "privacy_delta": cfg.flywheel.privacy_delta,
            },
        }
        backend = ModalBackend(client)
        try:
            handle = backend.submit(payload)
        except RuntimeError as exc:
            console.print(f"[wf.fail]Cloud submission failed:[/wf.fail] {exc}")
            raise typer.Exit(code=1) from None

        console.print(
            key_value_panel(
                {
                    "Backend": handle.backend,
                    "Job ID": handle.job_id,
                    "GPU": cloud_gpu,
                    "Region": cloud_region,
                    "Next": f"worldflux logs {handle.job_id}",
                },
                title="Cloud Training Submitted",
                border="wf.border.success",
            )
        )
        return

    # Resolve auto device
    import torch  # late import: avoid 2-5s startup penalty for all CLI commands

    if effective_device == "auto":
        effective_device = "cuda" if torch.cuda.is_available() else "cpu"
    if effective_device == "cuda" and not torch.cuda.is_available():
        console.print("[wf.caution]CUDA is not available. Falling back to CPU.[/wf.caution]")
        effective_device = "cpu"

    console.print(
        key_value_panel(
            {
                "Project": cfg.project_name,
                "Model": cfg.model,
                "Obs shape": str(cfg.architecture.obs_shape),
                "Action dim": str(cfg.architecture.action_dim),
                "Steps": f"{effective_steps:,}",
                "Batch size": str(cfg.training.batch_size),
                "Device": effective_device,
                "Backend": cfg.training.backend,
                "Profile": effective_backend_profile or "-",
                "Output": effective_output_dir,
            },
            title="WorldFlux Train",
            border="wf.border",
        )
    )

    try:
        model = create_world_model(
            model=cfg.model,
            obs_shape=cfg.architecture.obs_shape,
            action_dim=cfg.architecture.action_dim,
            hidden_dim=cfg.architecture.hidden_dim,
            device=effective_device,
            backend=cfg.training.backend,
        )
    except (ValueError, RuntimeError) as exc:
        console.print(f"[wf.fail]Model creation failed:[/wf.fail] {exc}")
        raise typer.Exit(code=1) from None

    training_config = TrainingConfig(
        total_steps=effective_steps,
        batch_size=cfg.training.batch_size,
        sequence_length=cfg.training.sequence_length,
        learning_rate=cfg.training.learning_rate,
        device=effective_device,
        output_dir=effective_output_dir,
        backend=cfg.training.backend,
        backend_profile=effective_backend_profile,
        use_ddp=use_ddp,
        num_envs=num_envs,
        env_mode=env_mode,
        profile=profile,
    )

    if isinstance(model, OfficialBackendHandle):
        backend_model = model.with_metadata(
            env=cfg.verify.env,
            task_filter=_env_to_task_filter(cfg.verify.env),
            profile=effective_backend_profile,
            backend_profile=effective_backend_profile,
        )
        trainer = Trainer(backend_model, training_config)
        try:
            handle = trainer.submit(resume_from=resume_from)
        except (RuntimeError, ValueError) as exc:
            console.print(f"[wf.fail]Delegated training submission failed:[/wf.fail] {exc}")
            raise typer.Exit(code=1) from None

        execution_result = handle.metadata.get("execution_result")
        if isinstance(execution_result, dict):
            status = str(execution_result.get("status", "failed"))
            panel = {
                "Backend": handle.backend,
                "Profile": effective_backend_profile or "-",
                "Run ID": handle.job_id,
                "Status": status,
                "Reason": execution_result.get("reason_code", "-"),
                "Message": execution_result.get("message", "-"),
                "Summary": execution_result.get("summary_path", "-"),
                "Next": execution_result.get("next_action", "-"),
            }
            console.print(
                key_value_panel(
                    panel,
                    title="Delegated Training Result",
                    border="wf.border.success" if status == "succeeded" else "wf.border.info",
                )
            )
            if status == "blocked":
                raise typer.Exit(code=2)
            if status == "incomplete":
                raise typer.Exit(code=3)
            if status != "succeeded":
                raise typer.Exit(code=1)
        else:
            console.print(
                key_value_panel(
                    {
                        "Backend": handle.backend,
                        "Profile": effective_backend_profile or "-",
                        "Run ID": handle.job_id,
                    },
                    title="Delegated Training Submitted",
                    border="wf.border.success",
                )
            )
        return

    project_root = Path(config).resolve().parent
    scaffold_runtime = _load_scaffold_runtime(project_root)
    extra_callbacks: list[Any] = []
    dashboard_context: dict[str, Any] | None = None
    degraded_modes: list[str] = []
    support_surface = _training_support_surface(cfg.model, cfg.training.backend)

    def _noop_cleanup() -> None:
        return None

    cleanup_data_source: Callable[[], None] = _noop_cleanup
    data_mode = "random"

    if scaffold_runtime is not None:
        try:
            dashboard_context, extra_callbacks = _prepare_scaffold_dashboard(
                cfg=cfg,
                effective_steps=effective_steps,
                effective_output_dir=effective_output_dir,
                runtime=scaffold_runtime,
            )
        except Exception as exc:
            console.print(f"[wf.caution]Visualization disabled:[/wf.caution] {exc}")
            dashboard_context = None
            extra_callbacks = []

    # Create data source
    console.print("[wf.info]Preparing training data...[/wf.info]")
    if scaffold_runtime is not None and hasattr(
        scaffold_runtime.dataset_module, "build_training_data"
    ):
        publish_phase = dashboard_context["publish_phase"] if dashboard_context else None
        publish_frame = dashboard_context["publish_frame"] if dashboard_context else None
        if publish_phase is not None and cfg.gameplay.enabled:
            publish_phase("collecting")
        try:
            data, cleanup_data_source, data_mode = (
                scaffold_runtime.dataset_module.build_training_data(
                    model.config,
                    frame_callback=publish_frame if cfg.gameplay.enabled else None,
                    phase_callback=publish_phase,
                )
            )
            if publish_phase is not None:
                publish_phase("training")
            console.print(f"[wf.ok]Training data ready:[/wf.ok] mode={data_mode}")
        except Exception as exc:
            console.print(
                "[wf.caution]Scaffold runtime fallback:[/wf.caution] "
                f"{exc}. Falling back to random replay data."
            )
            degraded_modes.extend(["scaffold_runtime_fallback", "random_replay_fallback"])
            data = create_random_buffer(
                obs_shape=cfg.architecture.obs_shape,
                action_dim=cfg.architecture.action_dim,
            )
            console.print(f"[wf.ok]Training data ready:[/wf.ok] {len(data)} transitions")
    else:
        data = create_random_buffer(
            obs_shape=cfg.architecture.obs_shape,
            action_dim=cfg.architecture.action_dim,
        )
        console.print(f"[wf.ok]Training data ready:[/wf.ok] {len(data)} transitions")

    if dashboard_context is not None:
        phase_events = dashboard_context.get("phase_events", [])
        if isinstance(phase_events, list):
            env_unavailable = any(
                phase == "unavailable"
                and detail is not None
                and "gameplay stream disabled" not in str(detail).lower()
                for phase, detail in phase_events
            )
            if env_unavailable:
                degraded_modes.append("env_collection_unavailable")
                if data_mode in {"offline", "random"}:
                    degraded_modes.append("random_replay_fallback")

    degraded_modes = _normalize_degraded_modes(degraded_modes)
    run_classification = _classify_local_run(
        support_surface=support_surface,
        data_mode=data_mode,
        degraded_modes=degraded_modes,
    )
    if degraded_modes:
        console.print(
            "[wf.caution]Run is degraded:[/wf.caution] "
            + ", ".join(mode.replace("_", "-") for mode in degraded_modes)
        )

    try:
        if use_ddp:
            from worldflux.training.distributed import DDPTrainer

            ddp_trainer = DDPTrainer(model, training_config)
            ddp_trainer._trainer.support_surface = support_surface
            ddp_trainer._trainer.data_mode = data_mode
            ddp_trainer._trainer.degraded_modes = list(degraded_modes)
            ddp_trainer._trainer.run_classification = run_classification
            console.print(
                f"[wf.info]Starting DDP training for {effective_steps:,} steps "
                f"(rank {ddp_trainer.rank}/{ddp_trainer.world_size})...[/wf.info]"
            )
            try:
                ddp_trainer.train(data, resume_from=resume_from)
            except (RuntimeError, KeyboardInterrupt) as exc:
                if isinstance(exc, KeyboardInterrupt):
                    console.print("\n[wf.caution]Training interrupted by user.[/wf.caution]")
                else:
                    console.print(f"[wf.fail]DDP training failed:[/wf.fail] {exc}")
                    raise typer.Exit(code=1) from None
            finally:
                ddp_trainer.cleanup()
            trainer = ddp_trainer._trainer  # For runtime profile access
        else:
            trainer = Trainer(model, training_config, callbacks=extra_callbacks or None)
            trainer.support_surface = support_surface
            trainer.data_mode = data_mode
            trainer.degraded_modes = list(degraded_modes)
            trainer.run_classification = run_classification
            console.print(f"[wf.info]Starting training for {effective_steps:,} steps...[/wf.info]")

            try:
                trainer.train(data, resume_from=resume_from)
            except (RuntimeError, KeyboardInterrupt) as exc:
                if isinstance(exc, KeyboardInterrupt):
                    console.print("\n[wf.caution]Training interrupted by user.[/wf.caution]")
                else:
                    console.print(f"[wf.fail]Training failed:[/wf.fail] {exc}")
                    raise typer.Exit(code=1) from None
    finally:
        try:
            cleanup_data_source()
        except Exception as cleanup_exc:
            console.print(f"[wf.caution]Data source cleanup failed:[/wf.caution] {cleanup_exc}")
        if dashboard_context is not None:
            publish_phase = dashboard_context.get("publish_phase")
            if callable(publish_phase):
                publish_phase("finished")
            dashboard_server = dashboard_context.get("dashboard_server")
            if dashboard_server is not None and hasattr(dashboard_server, "stop"):
                dashboard_server.stop()

    rt_profile = trainer.runtime_profile()
    elapsed = rt_profile.get("elapsed_sec")
    throughput = rt_profile.get("throughput_steps_per_sec")
    final_step = trainer.state.global_step

    summary_lines = [
        f"[wf.label]Run class:[/wf.label]   {run_classification}",
        f"[wf.label]Surface:[/wf.label]     {support_surface}",
        f"[wf.label]Data mode:[/wf.label]   {data_mode}",
        f"[wf.label]Final step:[/wf.label]  {final_step:,}",
        f"[wf.label]Output:[/wf.label]      {Path(effective_output_dir).resolve()}",
    ]
    if degraded_modes:
        summary_lines.append(
            "[wf.label]Warning:[/wf.label]     degraded via "
            + ", ".join(mode.replace("_", "-") for mode in degraded_modes)
        )
        summary_lines.extend(_degraded_guidance_lines(cfg, degraded_modes))
    if elapsed is not None:
        summary_lines.append(f"[wf.label]Elapsed:[/wf.label]     {elapsed:.1f}s")
    if throughput is not None:
        summary_lines.append(f"[wf.label]Throughput:[/wf.label]  {throughput:.1f} steps/s")
    summary_lines.append("")
    verify_next = "worldflux verify --target " + effective_output_dir
    if str(cfg.verify.mode).strip().lower() == "quick":
        verify_next += " --mode quick"
    summary_lines.append("Next: " + verify_next)
    if degraded_modes:
        summary_lines.append(
            "Action: rerun with a real environment-backed dataset before treating this as meaningful local training."
        )

    console.print(
        result_banner(
            passed=True,
            title="Training Complete",
            lines=summary_lines,
        )
    )
