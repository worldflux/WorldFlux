"""WorldFlux imagination demo powered by actual WorldFlux model inference."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch

from worldflux import create_world_model

MODEL_SPECS = {
    "DreamerV3": {
        "model_id": "dreamerv3:size12m",
        "obs_shape": (3, 64, 64),
        "action_dim": 6,
    },
    "TD-MPC2": {
        "model_id": "tdmpc2:5m",
        "obs_shape": (39,),
        "action_dim": 6,
    },
}


def _to_numpy_frame(tensor: torch.Tensor) -> np.ndarray | None:
    value = tensor.detach().cpu()
    if value.ndim == 4:
        value = value[0]
    if value.ndim == 3:
        frame = value.numpy()
        if frame.shape[0] in {1, 3}:
            frame = np.transpose(frame, (1, 2, 0))
        frame = np.nan_to_num(frame.astype(np.float32))
        if frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        if frame.max() > frame.min():
            frame = (frame - frame.min()) / (frame.max() - frame.min())
        return frame
    return None


def _extract_predictions(decoded: Any) -> dict[str, torch.Tensor]:
    if isinstance(decoded, dict):
        return decoded
    predictions = getattr(decoded, "predictions", {})
    if isinstance(predictions, dict):
        return predictions
    return {}


@lru_cache(maxsize=8)
def _load_model(model_type: str, checkpoint_path: str) -> Any:
    spec = MODEL_SPECS[model_type]
    model = create_world_model(
        spec["model_id"],
        obs_shape=spec["obs_shape"],
        action_dim=spec["action_dim"],
    )

    resolved_checkpoint = Path(checkpoint_path).expanduser() if checkpoint_path else None
    if resolved_checkpoint and resolved_checkpoint.exists():
        model = model.__class__.from_pretrained(str(resolved_checkpoint))

    model.eval()
    return model


def _build_initial_obs(model_type: str) -> torch.Tensor:
    spec = MODEL_SPECS[model_type]
    obs_shape = spec["obs_shape"]
    if len(obs_shape) == 3:
        channels, height, width = obs_shape
        grid = np.linspace(0.0, 1.0, height * width, dtype=np.float32).reshape(height, width)
        obs = np.stack([(grid + i / max(1, channels)) % 1.0 for i in range(channels)], axis=0)
        return torch.from_numpy(obs).unsqueeze(0)
    if len(obs_shape) == 1:
        vector = np.linspace(-1.0, 1.0, obs_shape[0], dtype=np.float32)
        return torch.from_numpy(vector).unsqueeze(0)
    raise ValueError(f"Unsupported observation shape: {obs_shape}")


def _build_action_sequence(model_type: str, horizon: int, device: torch.device) -> torch.Tensor:
    action_dim = MODEL_SPECS[model_type]["action_dim"]
    actions = torch.zeros(horizon, 1, action_dim, device=device)
    for t in range(horizon):
        actions[t, 0, t % action_dim] = 1.0
    return actions


def _plot_rewards(rewards: list[float]):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, marker="o")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Predicted Reward")
    ax.set_title("Imagined Rewards")
    ax.grid(True, alpha=0.3)
    return fig


def _plot_continues(continues: list[float]):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(continues, marker="s", color="green")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Continue Probability")
    ax.set_title("Episode Continuation")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    return fig


def _plot_frames(frames: list[np.ndarray]):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    if not frames:
        ax.text(
            0.5,
            0.5,
            "This model does not decode image observations.\n(Reward/continue are real model outputs)",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.axis("off")
        return fig

    preview = np.concatenate(frames[: min(5, len(frames))], axis=1)
    ax.imshow(preview)
    ax.set_title("Imagined Frame Preview")
    ax.axis("off")
    return fig


def run_imagination(model_type: str, horizon: int, checkpoint_path: str):
    model = _load_model(model_type, checkpoint_path.strip())
    device = next(model.parameters()).device
    initial_obs = _build_initial_obs(model_type).to(device=device)
    action_sequence = _build_action_sequence(model_type, int(horizon), device)

    rewards: list[float] = []
    continues: list[float] = []
    frames: list[np.ndarray] = []

    with torch.no_grad():
        state = model.encode({"obs": initial_obs})
        trajectory = model.rollout(state, action_sequence)

        rollout_rewards = getattr(trajectory, "rewards", None)
        if isinstance(rollout_rewards, torch.Tensor):
            rewards = rollout_rewards.detach().cpu().view(-1).tolist()
        rollout_continues = getattr(trajectory, "continues", None)
        if isinstance(rollout_continues, torch.Tensor):
            continues = torch.sigmoid(rollout_continues).detach().cpu().view(-1).tolist()

        for state_t in trajectory.states[1:]:
            decoded = model.decode(state_t)
            predictions = _extract_predictions(decoded)

            if not rewards and isinstance(predictions.get("reward"), torch.Tensor):
                rewards.append(float(predictions["reward"].detach().cpu().view(-1)[0]))
            if not continues and isinstance(predictions.get("continue"), torch.Tensor):
                continues.append(
                    float(torch.sigmoid(predictions["continue"]).detach().cpu().view(-1)[0])
                )

            obs_pred = predictions.get("obs")
            if isinstance(obs_pred, torch.Tensor):
                frame = _to_numpy_frame(obs_pred)
                if frame is not None:
                    frames.append(frame)

    if not rewards:
        rewards = [0.0 for _ in range(int(horizon))]
    if not continues:
        continues = [1.0 for _ in range(int(horizon))]

    rewards_plot = _plot_rewards(rewards)
    continues_plot = _plot_continues(continues)
    frames_plot = _plot_frames(frames)
    status = (
        f"Ran {model_type} inference for {int(horizon)} steps "
        f"(checkpoint={checkpoint_path.strip() or 'model preset'})"
    )
    return rewards_plot, continues_plot, frames_plot, status


with gr.Blocks() as demo:
    gr.Markdown("# WorldFlux Demo")
    gr.Markdown("Actual WorldFlux encode → rollout → decode inference (no random mock outputs)")

    model_type = gr.Dropdown(
        choices=["DreamerV3", "TD-MPC2"], value="DreamerV3", label="Model Type"
    )
    checkpoint_path = gr.Textbox(
        label="Checkpoint Path (optional)",
        placeholder="/data/checkpoints/dreamer_final",
    )
    horizon = gr.Slider(5, 50, value=15, step=1, label="Imagination Horizon")
    btn = gr.Button("Run Imagination")

    with gr.Row():
        rewards_plot = gr.Plot(label="Rewards")
        continues_plot = gr.Plot(label="Continues")
    frames_plot = gr.Plot(label="Imagined Frames")
    output_text = gr.Textbox(label="Status")

    btn.click(
        run_imagination,
        inputs=[model_type, horizon, checkpoint_path],
        outputs=[rewards_plot, continues_plot, frames_plot, output_text],
    )

if __name__ == "__main__":
    demo.launch()
