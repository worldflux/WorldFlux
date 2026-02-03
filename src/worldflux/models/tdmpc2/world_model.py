"""TD-MPC2 World Model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...core.config import TDMPC2Config
from ...core.latent_space import SimNormLatentSpace
from ...core.registry import WorldModelRegistry
from ...core.state import LatentState
from ...core.trajectory import Trajectory
from .dynamics import Dynamics
from .encoder import MLPEncoder
from .heads import PolicyHead, QEnsemble, RewardHead


@WorldModelRegistry.register("tdmpc2", TDMPC2Config)
class TDMPC2WorldModel(nn.Module):
    """
    TD-MPC2 world model implementation.

    Features:
        - No decoder (implicit model)
        - SimNorm latent space
        - Task embedding for multi-task
        - Q-function ensemble
    """

    def __init__(self, config: TDMPC2Config):
        super().__init__()
        self.config = config

        # Latent space
        self.latent_space = SimNormLatentSpace(
            dim=config.latent_dim,
            simnorm_dim=config.simnorm_dim,
        )

        # Compute observation dimension
        obs_dim = (
            config.obs_shape[0]
            if len(config.obs_shape) == 1
            else int(torch.prod(torch.tensor(config.obs_shape)).item())
        )

        # Encoder
        self._encoder = MLPEncoder(
            obs_dim=obs_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
        )

        # Dynamics
        self._dynamics = Dynamics(
            latent_dim=config.latent_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            num_tasks=config.num_tasks,
            task_dim=config.task_dim,
        )

        # Reward prediction
        self._reward_head = RewardHead(
            latent_dim=config.latent_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        )

        # Q-function ensemble
        self._q_ensemble = QEnsemble(
            latent_dim=config.latent_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            num_q_networks=config.num_q_networks,
        )

        # Policy (for MPC warm-start)
        self._policy = PolicyHead(
            latent_dim=config.latent_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
        )

        # Legacy attribute aliases for state_dict compatibility
        # These expose the internal nn.Sequential/ModuleList for loading old checkpoints
        self.encoder = self._encoder.mlp
        self.dynamics = self._dynamics.mlp
        self.task_embedding = self._dynamics.task_embedding
        self.reward_head = self._reward_head.mlp
        q_mlps: list[nn.Module] = [q.mlp for q in self._q_ensemble.q_networks]  # type: ignore[misc]
        self.q_networks = nn.ModuleList(q_mlps)
        self.policy = self._policy.mlp

        self.register_buffer("_device_tracker", torch.empty(0))

    @property
    def device(self) -> torch.device:
        device = self._device_tracker.device
        if not isinstance(device, torch.device):
            raise TypeError(f"Expected torch.device, got {type(device)}")
        return device

    def encode(self, obs: Tensor, deterministic: bool = False) -> LatentState:
        """Encode observation to SimNorm latent space."""
        z = self._encoder(obs)
        z = self.latent_space.sample(z)

        return LatentState(
            deterministic=z,
            latent_type="simnorm",
        )

    def predict(
        self,
        state: LatentState,
        action: Tensor,
        deterministic: bool = False,
        task_id: Tensor | None = None,
    ) -> LatentState:
        """Predict next state."""
        if state.deterministic is None:
            raise ValueError("TD-MPC2 requires deterministic state component")
        z = state.deterministic

        # Residual prediction
        z_delta = self._dynamics(z, action, task_id)
        z_next = z + z_delta
        z_next = self.latent_space.sample(z_next)

        return LatentState(
            deterministic=z_next,
            latent_type="simnorm",
        )

    def observe(self, state: LatentState, action: Tensor, obs: Tensor) -> LatentState:
        """TD-MPC2 directly encodes observations (no posterior like RSSM)."""
        return self.encode(obs)

    def decode(self, state: LatentState) -> dict[str, Tensor]:
        """
        Decode latent state to predictions.

        TD-MPC2 is an implicit model without explicit observation decoding.
        Returns reward prediction and additional model-specific outputs.

        Returns:
            Dictionary with standard keys:
                - reward: Predicted reward [batch, 1]
                - continue: Continue probability (always 1.0 for TD-MPC2) [batch, 1]
            And model-specific keys:
                - q_values: Q-value ensemble predictions [num_q, batch, 1]
                - action: Policy action [batch, action_dim]

        Note:
            TD-MPC2 doesn't decode observations (implicit model), so 'obs' key
            is intentionally omitted. Use isinstance checks or model_type to
            determine if observation decoding is available.
        """
        if state.deterministic is None:
            raise ValueError("TD-MPC2 requires deterministic state component")

        z = state.deterministic

        action = self._policy(z)
        reward = self._reward_head(z, action)
        q_values = self._q_ensemble(z, action)

        return {
            # Standard protocol keys
            "reward": reward,
            "continue": torch.ones_like(reward),  # TD-MPC2 doesn't predict termination
            # Model-specific keys
            "q_values": q_values,
            "action": action,
        }

    def imagine(
        self, initial_state: LatentState, actions: Tensor, deterministic: bool = False
    ) -> Trajectory:
        """Multi-step imagination."""
        horizon = actions.shape[0]
        states = [initial_state]
        rewards = []

        state = initial_state
        for t in range(horizon):
            state = self.predict(state, actions[t], deterministic=deterministic)
            states.append(state)

            if state.deterministic is None:
                raise ValueError(f"Predicted state at step {t} has no deterministic component")
            z = state.deterministic
            reward = self._reward_head(z, actions[t])
            rewards.append(reward)

        return Trajectory(
            states=states,
            actions=actions,
            rewards=torch.stack(rewards, dim=0).squeeze(-1),
        )

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> LatentState:
        """Initial state (uniform SimNorm)."""
        if device is None:
            device = self.device

        z = torch.zeros(batch_size, self.config.latent_dim, device=device)
        z = self.latent_space.sample(z)

        return LatentState(
            deterministic=z,
            latent_type="simnorm",
        )

    def predict_q(self, state: LatentState, action: Tensor) -> Tensor:
        """Predict Q-value ensemble."""
        if state.deterministic is None:
            raise ValueError("TD-MPC2 requires deterministic state component")
        z = state.deterministic
        q_values = self._q_ensemble(z, action)
        return q_values.squeeze(-1)

    def predict_reward(self, state: LatentState, action: Tensor) -> Tensor:
        """Predict reward."""
        if state.deterministic is None:
            raise ValueError("TD-MPC2 requires deterministic state component")
        z = state.deterministic
        return self._reward_head(z, action).squeeze(-1)

    def compute_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        TD-MPC2 loss computation.

        TD learning + latent consistency loss (BYOL-style).

        The encoder receives gradients from:
        - Consistency loss: through z_t -> dynamics -> z_pred path
        - Reward loss: through z_t -> reward_head path
        - TD loss: through z_t -> Q-ensemble path

        Targets are computed with no_grad (BYOL-style self-supervised learning).
        """
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]

        batch_size, seq_len = obs.shape[:2]
        device = obs.device

        # Encode all observations once for efficiency
        # Shape: [batch_size, seq_len, latent_dim]
        obs_flat = obs.view(batch_size * seq_len, *obs.shape[2:])
        z_all = self._encoder(obs_flat)
        z_all = self.latent_space.sample(z_all)
        z_all = z_all.view(batch_size, seq_len, -1)

        # Compute target encodings (no gradient for BYOL-style learning)
        with torch.no_grad():
            z_targets = z_all[:, 1:].detach()  # [batch, seq_len-1, latent_dim]

        losses: dict[str, Tensor] = {}

        # Latent consistency loss
        consistency_loss = torch.tensor(0.0, device=device)
        for t in range(seq_len - 1):
            z_t = z_all[:, t]  # Current encoding (with gradient)
            state_t = LatentState(deterministic=z_t, latent_type="simnorm")

            pred_state = self.predict(state_t, actions[:, t])
            if pred_state.deterministic is None:
                raise ValueError(f"Predicted state at step {t} has no deterministic component")
            z_pred = pred_state.deterministic

            z_target = z_targets[:, t]  # Target (no gradient)
            consistency_loss = consistency_loss + F.mse_loss(z_pred, z_target)

        losses["consistency"] = consistency_loss / max(seq_len - 1, 1)

        # Reward loss
        reward_loss = torch.tensor(0.0, device=device)
        for t in range(seq_len - 1):
            z_t = z_all[:, t]  # With gradient
            state_t = LatentState(deterministic=z_t, latent_type="simnorm")

            pred_reward = self.predict_reward(state_t, actions[:, t])
            reward_loss = reward_loss + F.mse_loss(pred_reward, rewards[:, t + 1])

        losses["reward"] = reward_loss / max(seq_len - 1, 1)

        # TD loss
        td_loss = torch.tensor(0.0, device=device)
        gamma = self.config.gamma
        for t in range(seq_len - 1):
            z_t = z_all[:, t]  # With gradient for Q-network
            state_t = LatentState(deterministic=z_t, latent_type="simnorm")

            q_values = self.predict_q(state_t, actions[:, t])

            with torch.no_grad():
                z_next = z_targets[:, t]  # No gradient for target
                state_next = LatentState(deterministic=z_next, latent_type="simnorm")
                next_action = self._policy(z_next)
                q_next = self.predict_q(state_next, next_action).min(dim=0)[0]
                target = rewards[:, t + 1] + gamma * q_next

            td_loss = td_loss + F.mse_loss(q_values.mean(dim=0), target)

        losses["td"] = td_loss / max(seq_len - 1, 1)

        losses["loss"] = losses["consistency"] + losses["reward"] + losses["td"]

        return losses

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs) -> "TDMPC2WorldModel":
        from ...core.registry import WorldModelRegistry

        model = WorldModelRegistry.from_pretrained(name_or_path, **kwargs)
        if not isinstance(model, cls):
            raise TypeError(
                f"Expected {cls.__name__}, got {type(model).__name__}. "
                f"Check that the model type in the config matches."
            )
        return model

    def save_pretrained(self, path: str) -> None:
        import os

        os.makedirs(path, exist_ok=True)
        self.config.save(os.path.join(path, "config.json"))
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
