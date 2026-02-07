"""String templates for ``worldflux init`` project generation."""

from __future__ import annotations

from textwrap import dedent
from typing import Any

from ._asset_dashboard_index import DASHBOARD_INDEX_HTML
from ._asset_dataset import DATASET_PY
from ._asset_local_dashboard import LOCAL_DASHBOARD_PY
from ._asset_train import TRAIN_PY


def _obs_shape_toml(obs_shape: list[int]) -> str:
    return "[" + ", ".join(str(dim) for dim in obs_shape) + "]"


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

    online_default = environment == "atari" and model_type.startswith("dreamer")
    data_source = "gym" if online_default else "random"
    gameplay_enabled = "true" if online_default else "false"
    online_enabled = "true" if online_default else "false"

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
        total_steps = 100000
        batch_size = 16
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
        """
        ).strip()
        + "\n"
    )


def render_train_py(context: dict[str, Any]) -> str:
    """Render ``train.py`` content."""
    del context
    return TRAIN_PY if TRAIN_PY.endswith("\n") else TRAIN_PY + "\n"


def render_local_dashboard_py(context: dict[str, Any]) -> str:
    """Render ``local_dashboard.py`` content."""
    del context
    return LOCAL_DASHBOARD_PY if LOCAL_DASHBOARD_PY.endswith("\n") else LOCAL_DASHBOARD_PY + "\n"


def render_dashboard_index_html(context: dict[str, Any]) -> str:
    """Render ``dashboard/index.html`` content."""
    del context
    return (
        DASHBOARD_INDEX_HTML if DASHBOARD_INDEX_HTML.endswith("\n") else DASHBOARD_INDEX_HTML + "\n"
    )


def render_dataset_py(context: dict[str, Any]) -> str:
    """Render ``dataset.py`` content."""
    del context
    return DATASET_PY if DATASET_PY.endswith("\n") else DATASET_PY + "\n"


def render_inference_py(context: dict[str, Any]) -> str:
    """Render ``inference.py`` content."""
    return (
        dedent(
            """
        from __future__ import annotations

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


        def try_load_checkpoint(model, checkpoint_path: Path) -> None:
            if not checkpoint_path.exists():
                print(f"No checkpoint found at {checkpoint_path}. Running with fresh weights.")
                return
            try:
                state_dict = torch.load(
                    checkpoint_path,
                    map_location=model.device,
                    weights_only=True,
                )
            except TypeError:  # pragma: no cover - old torch fallback
                state_dict = torch.load(checkpoint_path, map_location=model.device)
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
        # Preferred launcher:
        # 1) uv run python
        # 2) python
        # 3) python3
        # 4) py
        uv run python train.py
        ```

        When training starts, a local dashboard URL is printed:

        ```text
        Dashboard: http://127.0.0.1:8765
        ```

        If port `8765` is already in use, it automatically falls back to the next available port.

        Run inference or imagination checks:

        ```bash
        uv run python inference.py
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
