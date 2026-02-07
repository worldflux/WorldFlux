from __future__ import annotations

from pathlib import Path
from typing import Any

from dataset import build_training_data
from local_dashboard import DashboardCallback, GameplayBuffer, MetricBuffer, MetricsDashboardServer

from worldflux import create_world_model
from worldflux.training import Trainer, TrainingConfig

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def load_config(path: str = "worldflux.toml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")
    with config_path.open("rb") as f:
        return tomllib.load(f)


def resolve_model_id(config: dict) -> str:
    model = str(config.get("model", "")).strip()
    if model:
        return model
    model_type = str(config.get("model_type", "dreamer")).strip().lower()
    if model_type.startswith("dreamer"):
        return "dreamer:ci"
    return "tdmpc2:ci"


def resolve_visualization_config(config: dict[str, Any]) -> dict[str, Any]:
    visual = config.get("visualization", {})
    if not isinstance(visual, dict):
        visual = {}

    host = str(visual.get("host", "127.0.0.1")).strip() or "127.0.0.1"
    port = int(visual.get("port", 8765))
    refresh_ms = max(100, int(visual.get("refresh_ms", 1000)))
    history_max_points = max(100, int(visual.get("history_max_points", 2000)))

    return {
        "enabled": bool(visual.get("enabled", True)),
        "host": host,
        "port": max(1, port),
        "refresh_ms": refresh_ms,
        "history_max_points": history_max_points,
        "open_browser": bool(visual.get("open_browser", False)),
    }


def resolve_gameplay_config(config: dict[str, Any]) -> dict[str, Any]:
    gameplay = config.get("gameplay", {})
    if not isinstance(gameplay, dict):
        gameplay = {}

    return {
        "enabled": bool(gameplay.get("enabled", True)),
        "fps": max(1, int(gameplay.get("fps", 8))),
        "max_frames": max(16, int(gameplay.get("max_frames", 512))),
    }


def main() -> None:
    config = load_config()
    architecture = config.get("architecture", {})
    training = config.get("training", {})
    visualization = resolve_visualization_config(config)
    gameplay = resolve_gameplay_config(config)

    obs_shape = tuple(int(dim) for dim in architecture.get("obs_shape", [3, 64, 64]))
    action_dim = int(architecture.get("action_dim", 6))
    hidden_dim = int(architecture.get("hidden_dim", 32))
    model_id = resolve_model_id(config)
    device = str(training.get("device", "cpu"))
    total_steps = int(training.get("total_steps", 100000))
    batch_size = int(training.get("batch_size", 16))
    sequence_length = int(training.get("sequence_length", 50))
    learning_rate = float(training.get("learning_rate", 3e-4))
    output_dir = str(training.get("output_dir", "./outputs"))

    print(f"Initializing model: {model_id} on {device}")
    model = create_world_model(
        model=model_id,
        obs_shape=obs_shape,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        device=device,
    )

    dashboard_buffer: MetricBuffer | None = None
    gameplay_buffer: GameplayBuffer | None = None
    dashboard_server: MetricsDashboardServer | None = None
    dashboard_callback: DashboardCallback | None = None
    extra_callbacks = []
    gameplay_unavailable = not gameplay["enabled"]
    unavailable_detail: str | None = (
        "Gameplay stream disabled in worldflux.toml." if not gameplay["enabled"] else None
    )

    def publish_phase(phase: str, detail: str | None = None) -> None:
        nonlocal gameplay_unavailable, unavailable_detail

        if phase == "unavailable":
            gameplay_unavailable = True
            unavailable_detail = detail or unavailable_detail

        if dashboard_buffer is not None:
            dashboard_buffer.set_phase(phase, detail)
            if phase == "unavailable":
                dashboard_buffer.set_gameplay_available(False)

        if gameplay_buffer is not None:
            gameplay_buffer.set_phase(phase, detail)
            if phase == "unavailable":
                gameplay_buffer.set_status("unavailable")
            elif phase in {"finished", "error"}:
                gameplay_buffer.set_status(phase)
            else:
                gameplay_buffer.set_status("running")

    def publish_frame(
        frame: Any,
        episode: int,
        episode_step: int,
        reward: float,
        done: bool,
    ) -> None:
        if gameplay_buffer is None:
            return
        gameplay_buffer.append_frame(
            frame,
            episode=episode,
            episode_step=episode_step,
            reward=reward,
            done=done,
        )

    if visualization["enabled"]:
        try:
            dashboard_buffer = MetricBuffer(max_points=visualization["history_max_points"])
            if gameplay["enabled"]:
                gameplay_buffer = GameplayBuffer(
                    max_frames=int(gameplay["max_frames"]),
                    fps=int(gameplay["fps"]),
                )
                dashboard_buffer.set_gameplay_available(True)
            else:
                dashboard_buffer.set_gameplay_available(False)
                dashboard_buffer.set_phase("unavailable", unavailable_detail)

            dashboard_server = MetricsDashboardServer(
                metric_buffer=dashboard_buffer,
                gameplay_buffer=gameplay_buffer,
                host=str(visualization["host"]),
                start_port=int(visualization["port"]),
                dashboard_root=Path(__file__).parent / "dashboard",
                refresh_ms=int(visualization["refresh_ms"]),
                max_port_tries=100,
            )
            dashboard_server.start()
            print(f"Dashboard: {dashboard_server.url}")

            if visualization["open_browser"]:
                dashboard_server.open_browser()

            dashboard_callback = DashboardCallback(
                dashboard_buffer,
                Path(output_dir) / "metrics.jsonl",
            )
            extra_callbacks.append(dashboard_callback)
        except Exception as exc:
            dashboard_buffer = None
            gameplay_buffer = None
            dashboard_server = None
            dashboard_callback = None
            print(f"Visualization disabled (startup failed): {exc}")

    print("Preparing training data...")
    if dashboard_buffer is not None and gameplay["enabled"]:
        publish_phase("collecting")

    def _noop_cleanup() -> None:
        return

    cleanup_data_source = _noop_cleanup
    data_mode = "offline"
    data_source, cleanup_data_source, data_mode = build_training_data(
        model.config,
        frame_callback=publish_frame if gameplay["enabled"] else None,
        phase_callback=publish_phase if dashboard_buffer is not None else None,
    )

    if dashboard_buffer is not None:
        if gameplay_unavailable:
            publish_phase("unavailable", unavailable_detail)
        else:
            publish_phase("training")

    print(
        "Training config: "
        f"steps={total_steps}, batch_size={batch_size}, "
        f"sequence_length={sequence_length}, lr={learning_rate}, mode={data_mode}"
    )

    trainer = Trainer(
        model,
        TrainingConfig(
            total_steps=total_steps,
            batch_size=batch_size,
            sequence_length=sequence_length,
            learning_rate=learning_rate,
            output_dir=output_dir,
            device=device,
        ),
        callbacks=extra_callbacks or None,
    )

    print("Starting training...")
    failed_error: Exception | None = None
    try:
        trainer.train(data_source)
    except Exception as exc:
        failed_error = exc
        if dashboard_buffer is not None:
            dashboard_buffer.set_status("error", str(exc))
            dashboard_buffer.set_phase("error", str(exc))
            dashboard_buffer.set_gameplay_available(not gameplay_unavailable)
        if gameplay_buffer is not None:
            gameplay_buffer.set_status("error")
            gameplay_buffer.set_phase("error", str(exc))
        if dashboard_callback is not None:
            dashboard_callback.close()
        raise
    finally:
        try:
            cleanup_data_source()
        except Exception as exc:
            print(f"Data source cleanup failed: {exc}")

        if dashboard_server is not None:
            linger_seconds = 60.0
            if failed_error is None and dashboard_buffer is not None:
                dashboard_buffer.set_status("finished")
                dashboard_buffer.set_phase("finished")
                dashboard_buffer.set_gameplay_available(not gameplay_unavailable)
            if failed_error is None and gameplay_buffer is not None:
                gameplay_buffer.set_status("finished")
                gameplay_buffer.set_phase("finished")
            dashboard_server.schedule_shutdown(linger_seconds)
            print(
                "Dashboard will stay online for " f"{int(linger_seconds)}s: {dashboard_server.url}"
            )
            dashboard_server.wait_for_stop(timeout=linger_seconds + 5.0)

    print("Training complete.")


if __name__ == "__main__":
    main()
