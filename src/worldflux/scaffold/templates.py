"""String templates for ``worldflux init`` project generation."""

from __future__ import annotations

from textwrap import dedent
from typing import Any

from ._asset_dashboard_index import DASHBOARD_INDEX_HTML
from ._asset_dataset import DATASET_PY
from ._asset_local_dashboard import LOCAL_DASHBOARD_PY
from ._asset_train import TRAIN_PY


def _ensure_trailing_newline(content: str) -> str:
    return content if content.endswith("\n") else content + "\n"


def _replace_exact(content: str, old: str, new: str, *, label: str) -> str:
    if old not in content:
        raise RuntimeError(f"Scaffold template drift detected while applying patch: {label}")
    return content.replace(old, new)


def _patch_dataset_template(content: str) -> str:
    content = _replace_exact(
        content,
        "import time\nimport warnings\n",
        "import time\n",
        label="dataset remove warnings import",
    )
    content = _replace_exact(
        content,
        "CleanupFn = Callable[[], None]\n\n\n",
        'CleanupFn = Callable[[], None]\n\n\ndef _info(message: str) -> None:\n    print(f"[dataset] {message}")\n\n\n',
        label="dataset info logger",
    )

    replacements = (
        (
            '        warnings.warn(f"Atari gym collection is unavailable: {exc}")',
            '        _info(f"Atari gym collection is unavailable: {exc}")',
            "dataset atari unavailable log",
        ),
        (
            "        warnings.warn(f\"Failed to create Atari env '{gym_env}': {exc}\")",
            "        _info(f\"Failed to create Atari env '{gym_env}': {exc}\")",
            "dataset atari env failure log",
        ),
        (
            '        warnings.warn(f"MuJoCo gym collection is unavailable: {exc}")',
            '        _info(f"MuJoCo gym collection is unavailable: {exc}")',
            "dataset mujoco unavailable log",
        ),
        (
            "        warnings.warn(f\"Failed to create MuJoCo env '{gym_env}': {exc}\")",
            "        _info(f\"Failed to create MuJoCo env '{gym_env}': {exc}\")",
            "dataset mujoco env failure log",
        ),
        (
            '        warnings.warn("Falling back to random buffer because gym collection failed.")',
            '        _info("Falling back to random buffer because gym collection failed.")',
            "dataset fallback log",
        ),
        (
            '            warnings.warn(f"Online Atari collection is unavailable: {exc}")',
            '            _info(f"Online Atari collection is unavailable: {exc}")',
            "dataset online atari log",
        ),
        (
            '            warnings.warn(f"Online MuJoCo collection is unavailable: {exc}")',
            '            _info(f"Online MuJoCo collection is unavailable: {exc}")',
            "dataset online mujoco log",
        ),
    )
    for old, new, label in replacements:
        content = _replace_exact(content, old, new, label=label)

    return content


def _patch_train_template(content: str) -> str:
    content = _replace_exact(
        content,
        """        if dashboard_buffer is not None:
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
""",
        """        if dashboard_buffer is not None:
            dashboard_buffer.set_phase(phase, detail)
            if phase == "unavailable" or (phase == "training" and gameplay_unavailable):
                dashboard_buffer.set_gameplay_available(False)

        if gameplay_buffer is not None:
            gameplay_buffer.set_phase(phase, detail)
            if phase == "unavailable":
                gameplay_buffer.set_status("unavailable")
            elif phase == "training" and gameplay_unavailable:
                gameplay_buffer.set_status("unavailable")
            elif phase in {"finished", "error"}:
                gameplay_buffer.set_status(phase)
            else:
                gameplay_buffer.set_status("running")
""",
        label="train phase behavior",
    )

    content = _replace_exact(
        content,
        '            dashboard_buffer = MetricBuffer(max_points=visualization["history_max_points"])\n',
        """            dashboard_buffer = MetricBuffer(max_points=visualization["history_max_points"])
            dashboard_buffer.set_target_steps(total_steps)
""",
        label="train set target steps",
    )
    content = _replace_exact(
        content,
        '            print(f"Dashboard: {dashboard_server.url}")\n',
        """            print(f"Dashboard: {dashboard_server.url}")
            print("Open this URL in your browser to monitor live training progress.")
""",
        label="train dashboard url hint",
    )
    content = _replace_exact(
        content,
        """    if dashboard_buffer is not None:
        if gameplay_unavailable:
            publish_phase("unavailable", unavailable_detail)
        else:
            publish_phase("training")
""",
        """    if dashboard_buffer is not None:
        if gameplay_unavailable and unavailable_detail:
            publish_phase(
                "training",
                f"Training is running. Live gameplay unavailable. {unavailable_detail}",
            )
        else:
            publish_phase("training")
""",
        label="train phase summary",
    )
    content = _replace_exact(
        content,
        """            dashboard_server.schedule_shutdown(linger_seconds)
            print(
                "Dashboard will stay online for " f"{int(linger_seconds)}s: {dashboard_server.url}"
            )
            dashboard_server.wait_for_stop(timeout=linger_seconds + 5.0)
""",
        """            dashboard_server.schedule_shutdown(linger_seconds)
            print(
                "Dashboard will stay online for " f"{int(linger_seconds)}s: {dashboard_server.url}"
            )
            try:
                dashboard_server.wait_for_stop(timeout=linger_seconds + 5.0)
            except KeyboardInterrupt:
                print("Interrupted while waiting for dashboard shutdown.")
""",
        label="train dashboard shutdown interrupt handling",
    )
    return content


def _patch_local_dashboard_template(content: str) -> str:
    content = _replace_exact(
        content,
        "        self._latest_speed = 0.0\n",
        """        self._latest_speed = 0.0
        self._target_steps: int | None = None
""",
        label="dashboard target field",
    )
    content = _replace_exact(
        content,
        """    def set_gameplay_available(self, available: bool) -> None:
        with self._lock:
            self._gameplay_available = bool(available)

    def metrics_payload(self, since_step: int = -1) -> dict[str, Any]:
""",
        """    def set_gameplay_available(self, available: bool) -> None:
        with self._lock:
            self._gameplay_available = bool(available)

    def set_target_steps(self, total_steps: int) -> None:
        normalized = max(1, int(total_steps))
        with self._lock:
            self._target_steps = normalized

    def metrics_payload(self, since_step: int = -1) -> dict[str, Any]:
""",
        label="dashboard target setter",
    )
    content = _replace_exact(
        content,
        """    def summary_payload(self, *, host: str, port: int) -> dict[str, Any]:
        with self._lock:
            ended_at = self._ended_at
            now = time.time()
            return {
                "status": self._status,
                "phase": self._phase,
                "phase_message": self._phase_message,
                "gameplay_available": self._gameplay_available,
                "started_at": self._started_at,
                "ended_at": ended_at,
                "elapsed_seconds": (ended_at or now) - self._started_at,
                "latest_step": self._latest_step,
                "latest_metrics": dict(self._latest_metrics),
                "latest_speed": self._latest_speed,
                "error": self._error,
                "host": host,
                "port": int(port),
                "total_points": len(self._points),
            }
""",
        """    def summary_payload(self, *, host: str, port: int) -> dict[str, Any]:
        with self._lock:
            ended_at = self._ended_at
            now = time.time()
            target_steps = self._target_steps
            progress_percent = 0.0
            if target_steps and target_steps > 0:
                progress_percent = min(
                    100.0,
                    max(0.0, (self._latest_step / float(target_steps)) * 100.0),
                )
            return {
                "status": self._status,
                "phase": self._phase,
                "phase_message": self._phase_message,
                "gameplay_available": self._gameplay_available,
                "started_at": self._started_at,
                "ended_at": ended_at,
                "elapsed_seconds": (ended_at or now) - self._started_at,
                "latest_step": self._latest_step,
                "target_steps": target_steps,
                "progress_percent": progress_percent,
                "latest_metrics": dict(self._latest_metrics),
                "latest_speed": self._latest_speed,
                "error": self._error,
                "host": host,
                "port": int(port),
                "total_points": len(self._points),
            }
""",
        label="dashboard summary progress",
    )
    return content


def _patch_dashboard_html_template(content: str) -> str:
    content = _replace_exact(
        content,
        """      .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
      }
""",
        """      .dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
      }

      .progress-track {
        width: 100%;
        height: 14px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: #0e1317;
        overflow: hidden;
      }

      .progress-fill {
        height: 100%;
        width: 0%;
        background: linear-gradient(90deg, #57c58f 0%, #65b1ff 100%);
        transition: width 0.35s ease;
      }

      .progress-meta {
        margin: 10px 0 0;
        color: var(--muted);
        font-size: 12px;
      }
""",
        label="dashboard html progress styles",
    )
    content = _replace_exact(
        content,
        """        <div class="card" data-metric="step">
          <div class="label">Step</div>
          <div id="step" class="value">0</div>
        </div>
        <div class="card" data-metric="elapsed">
""",
        """        <div class="card" data-metric="step">
          <div class="label">Step</div>
          <div id="step" class="value">0</div>
        </div>
        <div class="card" data-metric="progress">
          <div class="label">Progress</div>
          <div id="progress" class="value">0.0%</div>
        </div>
        <div class="card" data-metric="elapsed">
""",
        label="dashboard html progress card",
    )
    content = _replace_exact(
        content,
        """      </section>

      <section class="section">
        <h2>Live Gameplay</h2>
""",
        """      </section>

      <section class="section">
        <h2>Training Progress</h2>
        <div class="progress-track">
          <div id="progress-fill" class="progress-fill"></div>
        </div>
        <p id="progress-meta" class="progress-meta">0 / 0 steps</p>
      </section>

      <section class="section">
        <h2>Live Gameplay</h2>
""",
        label="dashboard html progress section",
    )
    content = _replace_exact(
        content,
        """        step:
          "How many optimizer updates have run. It usually increases by one each training iteration.",
        elapsed:
""",
        """        step:
          "How many optimizer updates have run. It usually increases by one each training iteration.",
        progress:
          "Training completion percentage computed from current step and configured total steps.",
        elapsed:
""",
        label="dashboard html progress tooltip",
    )
    content = _replace_exact(
        content,
        """      const stepEl = document.getElementById("step");
      const elapsedEl = document.getElementById("elapsed");
      const speedEl = document.getElementById("speed");
      const lossEl = document.getElementById("loss");
""",
        """      const stepEl = document.getElementById("step");
      const progressEl = document.getElementById("progress");
      const progressFillEl = document.getElementById("progress-fill");
      const progressMetaEl = document.getElementById("progress-meta");
      const elapsedEl = document.getElementById("elapsed");
      const speedEl = document.getElementById("speed");
      const lossEl = document.getElementById("loss");
""",
        label="dashboard html progress dom refs",
    )
    content = _replace_exact(
        content,
        """      function pickColor(index) {
        return palette[index % palette.length];
      }
""",
        """      function pickColor(index) {
        return palette[index % palette.length];
      }

      function resolveProgressPercent(latest) {
        const summaryPercent = state.summary?.progress_percent;
        if (typeof summaryPercent === "number" && Number.isFinite(summaryPercent)) {
          return Math.min(100, Math.max(0, summaryPercent));
        }

        const targetSteps = state.summary?.target_steps;
        if (
          latest &&
          typeof latest.step === "number" &&
          typeof targetSteps === "number" &&
          Number.isFinite(targetSteps) &&
          targetSteps > 0
        ) {
          return Math.min(100, Math.max(0, (latest.step / targetSteps) * 100));
        }
        return 0;
      }
""",
        label="dashboard html progress helper",
    )
    content = _replace_exact(
        content,
        """        const elapsed = state.summary ? state.summary.elapsed_seconds : 0;
        elapsedEl.textContent = formatElapsed(elapsed);
""",
        """        const elapsed = state.summary ? state.summary.elapsed_seconds : 0;
        elapsedEl.textContent = formatElapsed(elapsed);

        const progressPercent = resolveProgressPercent(latest);
        progressEl.textContent = `${progressPercent.toFixed(1)}%`;
        progressFillEl.style.width = `${progressPercent.toFixed(1)}%`;
        const targetSteps = state.summary?.target_steps;
        if (typeof targetSteps === "number" && targetSteps > 0) {
          const currentStep = latest ? latest.step : 0;
          progressMetaEl.textContent = `${currentStep} / ${targetSteps} steps`;
        } else {
          progressMetaEl.textContent = `${latest ? latest.step : 0} / ? steps`;
        }
""",
        label="dashboard html progress render",
    )
    return content


def _obs_shape_toml(obs_shape: list[int]) -> str:
    return "[" + ", ".join(str(dim) for dim in obs_shape) + "]"


def _default_verify_env(environment: str) -> str:
    if environment == "mujoco":
        return "mujoco/halfcheetah"
    return "atari/pong"


def _default_gym_env(environment: str) -> str:
    if environment == "atari":
        return "ALE/Breakout-v5"
    if environment == "mujoco":
        return "HalfCheetah-v5"
    return ""


def render_worldflux_toml(context: dict[str, Any]) -> str:
    """Render ``worldflux.toml`` content."""
    environment = str(context["environment"]).strip().lower()
    model_type = str(context["model_type"]).strip().lower()
    gym_env = _default_gym_env(environment)
    total_steps = max(1, int(context.get("training_total_steps", 100000)))
    batch_size = max(1, int(context.get("training_batch_size", 16)))

    online_default = (
        environment == "atari" and model_type.startswith("dreamer")
    ) or environment == "mujoco"
    data_source = "gym" if online_default else "random"
    gameplay_enabled = "true" if online_default else "false"
    online_enabled = "true" if online_default else "false"

    verify_env = _default_verify_env(environment)

    return (
        dedent(
            f"""
        project_name = "{context["project_name"]}"
        environment = "{environment}"
        model = "{context["model"]}"
        model_type = "{context["model_type"]}"

        [architecture]
        obs_shape = {_obs_shape_toml(context["obs_shape"])}
        action_dim = {context["action_dim"]}
        hidden_dim = {context["hidden_dim"]}

        [training]
        total_steps = {total_steps}
        batch_size = {batch_size}
        sequence_length = 50
        learning_rate = 3e-4
        device = "{context["device"]}"
        output_dir = "./outputs"

        [data]
        source = "{data_source}"  # "random" or "gym"
        num_episodes = 100
        episode_length = 100
        buffer_capacity = 10000
        gym_env = "{gym_env}"

        [gameplay]
        enabled = {gameplay_enabled}
        fps = 8
        max_frames = 512

        [online_collection]
        enabled = {online_enabled}
        warmup_transitions = 512
        collect_steps_per_update = 64
        max_episode_steps = 100

        [inference]
        horizon = 15
        checkpoint = "./outputs/checkpoint_best.pt"

        [visualization]
        enabled = true
        host = "127.0.0.1"
        port = 8765
        refresh_ms = 1000
        history_max_points = 2000
        open_browser = false

        [verify]
        baseline = "official/dreamerv3"
        env = "{verify_env}"

        [cloud]
        gpu_type = "a100"
        spot = true
        region = "us-east-1"
        timeout_hours = 24

        [flywheel]
        opt_in = false
        privacy_epsilon = 1.0
        privacy_delta = 1e-5
        """
        ).strip()
        + "\n"
    )


def render_train_py(context: dict[str, Any]) -> str:
    """Render ``train.py`` content."""
    del context
    return _ensure_trailing_newline(_patch_train_template(TRAIN_PY))


def render_local_dashboard_py(context: dict[str, Any]) -> str:
    """Render ``local_dashboard.py`` content."""
    del context
    return _ensure_trailing_newline(_patch_local_dashboard_template(LOCAL_DASHBOARD_PY))


def render_dashboard_index_html(context: dict[str, Any]) -> str:
    """Render ``dashboard/index.html`` content."""
    del context
    return _ensure_trailing_newline(_patch_dashboard_html_template(DASHBOARD_INDEX_HTML))


def render_dataset_py(context: dict[str, Any]) -> str:
    """Render ``dataset.py`` content."""
    del context
    return _ensure_trailing_newline(_patch_dataset_template(DATASET_PY))


def render_inference_py(context: dict[str, Any]) -> str:
    """Render ``inference.py`` content."""
    return (
        dedent(
            """
        from __future__ import annotations

        from collections.abc import Mapping
        from pathlib import Path

        import torch
        from worldflux import create_world_model

        try:
            import tomllib  # Python 3.11+
        except ModuleNotFoundError:  # pragma: no cover
            import tomli as tomllib


        def load_config(path: str = "worldflux.toml") -> dict:
            with Path(path).open("rb") as f:
                return tomllib.load(f)


        def resolve_model_id(config: dict) -> str:
            model = str(config.get("model", "")).strip()
            if model:
                return model
            model_type = str(config.get("model_type", "dreamer")).strip().lower()
            if model_type.startswith("dreamer"):
                return "dreamer:ci"
            return "tdmpc2:ci"


        def _unwrap_model_state_dict(payload: object) -> dict[str, torch.Tensor]:
            if isinstance(payload, Mapping):
                nested = payload.get("model_state_dict")
                if isinstance(nested, Mapping):
                    return dict(nested)
                if all(isinstance(key, str) for key in payload.keys()):
                    return dict(payload)
            raise RuntimeError(
                "Unsupported checkpoint format. Expected a raw state_dict or a Trainer checkpoint "
                "containing 'model_state_dict'."
            )


        def try_load_checkpoint(model, checkpoint_path: Path) -> None:
            if not checkpoint_path.exists():
                print(f"No checkpoint found at {checkpoint_path}. Running with fresh weights.")
                return
            checkpoint_payload = torch.load(
                checkpoint_path,
                map_location=model.device,
                weights_only=True,
            )
            state_dict = _unwrap_model_state_dict(checkpoint_payload)
            model.load_state_dict(state_dict)
            print(f"Loaded checkpoint: {checkpoint_path}")


        def main() -> None:
            cfg = load_config()
            arch = cfg.get("architecture", {})
            train_cfg = cfg.get("training", {})
            infer_cfg = cfg.get("inference", {})

            model_id = resolve_model_id(cfg)
            obs_shape = tuple(int(dim) for dim in arch.get("obs_shape", [3, 64, 64]))
            action_dim = int(arch.get("action_dim", 6))
            hidden_dim = int(arch.get("hidden_dim", 32))
            device = str(train_cfg.get("device", "cpu"))

            model = create_world_model(
                model=model_id,
                obs_shape=obs_shape,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                device=device,
            )
            model.eval()

            checkpoint = Path(str(infer_cfg.get("checkpoint", "./outputs/checkpoint_best.pt")))
            try_load_checkpoint(model, checkpoint)

            horizon = max(1, int(infer_cfg.get("horizon", 15)))
            batch_size = 1
            initial_obs = torch.randn(batch_size, *obs_shape, device=model.device)
            action_seq = torch.randn(horizon, batch_size, action_dim, device=model.device)

            with torch.no_grad():
                initial_state = model.encode(initial_obs)
                trajectory = model.rollout(initial_state, action_seq)

            print("Rollout complete.")
            print(f"Horizon: {trajectory.horizon}")
            print(f"States: {len(trajectory.states)}")
            if trajectory.rewards is not None:
                print(
                    "Reward stats: "
                    f"mean={trajectory.rewards.mean().item():.4f}, "
                    f"std={trajectory.rewards.std().item():.4f}"
                )
            else:
                print("This model does not provide reward predictions.")
            if trajectory.continues is not None:
                print(f"Continue mean: {trajectory.continues.mean().item():.4f}")


        if __name__ == "__main__":
            main()
        """
        ).strip()
        + "\n"
    )


def render_readme_md(context: dict[str, Any]) -> str:
    """Render ``README.md`` content."""
    return (
        dedent(
            f"""
        # {context["project_name"]}

        Generated with `worldflux init`.

        ## Quick Start

        ```bash
        # Train your model
        worldflux train

        # Or use the Python script directly
        {context.get("preferred_python_launcher", "uv run python")} train.py
        ```

        When training starts, a local dashboard URL is printed:

        ```text
        Dashboard: http://127.0.0.1:8765
        ```

        If port `8765` is already in use, it automatically falls back to the next available port.

        Verify your trained model:

        ```bash
        worldflux verify --target ./outputs
        ```

        Run inference or imagination checks:

        ```bash
        {context.get("preferred_python_launcher", "uv run python")} inference.py
        ```

        ## Project Files

        - `worldflux.toml`: Main config (model, architecture, training, data).
        - `train.py`: Training entrypoint using `create_world_model` + `Trainer`.
        - `local_dashboard.py`: Local HTTP dashboard server + training callback.
        - `dashboard/index.html`: Browser UI for real-time metrics.
        - `dataset.py`: Demo data provider (random-first, optional gym collection).
        - `inference.py`: Short rollout and prediction stats.

        ## Configuration

        `worldflux.toml` drives this project. Common changes:

        - `training.batch_size`
        - `training.total_steps`
        - `training.learning_rate`
        - `data.source` (`"random"` or `"gym"`)
        - `data.gym_env`
        - `gameplay.enabled`
        - `gameplay.fps`
        - `gameplay.max_frames`
        - `online_collection.enabled`
        - `online_collection.warmup_transitions`
        - `online_collection.collect_steps_per_update`
        - `online_collection.max_episode_steps`
        - `visualization.enabled`
        - `visualization.port`
        - `visualization.refresh_ms`
        - `visualization.open_browser`

        ## Verify Your Model

        Run parity verification against a baseline:

        ```bash
        # Demo mode (instant synthetic results for presentations)
        worldflux verify --target ./outputs/checkpoint_best.pt --demo

        # Real verification (requires parity suite)
        worldflux verify --target ./outputs/checkpoint_best.pt
        ```

        ## Gym Data Collection (Optional)

        This project runs without extra dependencies in random mode.
        For Atari Dreamer projects, online collection is enabled by default so gameplay stays active during training.
        To collect environment data with gym:

        ```bash
        uv pip install "gymnasium>=0.29.0,<2.0.0"
        ```

        For Atari, also install:

        ```bash
        uv pip install "gymnasium[atari]>=0.29.0,<2.0.0" "ale-py>=0.8.0,<1.0.0"
        ```

        For MuJoCo, also install:

        ```bash
        uv pip install "gymnasium[mujoco]>=0.29.0,<2.0.0" "mujoco>=3.0.0,<4.0.0"
        ```
        """
        ).strip()
        + "\n"
    )
