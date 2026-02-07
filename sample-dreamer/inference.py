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
