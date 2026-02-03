"""WorldFlux Imagination Rollout Demo - Gradio Interface"""

import gradio as gr
import numpy as np


def run_imagination(model_type, horizon, batch_size):
    """Run imagination rollout visualization"""
    # Placeholder - will be replaced with actual model
    rewards = np.random.randn(horizon).cumsum()
    continues = 1 / (1 + np.exp(-np.random.randn(horizon).cumsum()))

    fig_rewards = plot_rewards(rewards)
    fig_continues = plot_continues(continues)

    return fig_rewards, fig_continues, f"Ran {model_type} for {horizon} steps"


def plot_rewards(rewards):
    """Plot predicted rewards over time"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, marker="o")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Predicted Reward")
    ax.set_title("Imagined Rewards")
    ax.grid(True, alpha=0.3)
    return fig


def plot_continues(continues):
    """Plot continuation probabilities over time"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(continues, marker="s", color="green")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Continue Probability")
    ax.set_title("Episode Continuation")
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    return fig


with gr.Blocks() as demo:
    gr.Markdown("# WorldFlux Demo")
    gr.Markdown("Experience world model imagination rollouts")

    model_type = gr.Dropdown(
        choices=["DreamerV3", "TD-MPC2"], value="DreamerV3", label="Model Type"
    )

    horizon = gr.Slider(5, 50, value=15, label="Imagination Horizon")

    btn = gr.Button("Run Imagination")

    with gr.Row():
        rewards_plot = gr.Plot()
        continues_plot = gr.Plot()

    output_text = gr.Textbox(label="Status")

    btn.click(
        run_imagination,
        inputs=[model_type, horizon, gr.State(1)],
        outputs=[rewards_plot, continues_plot, output_text],
    )

if __name__ == "__main__":
    demo.launch()
