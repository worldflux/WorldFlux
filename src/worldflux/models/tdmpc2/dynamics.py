"""TD-MPC2 Dynamics Model."""

import torch
import torch.nn as nn
from torch import Tensor

_DYNAMICS_NAN_FILL = 0.0
_DYNAMICS_INPUT_CLAMP = 1e4
_DYNAMICS_DELTA_MAX_NORM = 10.0
_DYNAMICS_EPS = 1e-6


class Dynamics(nn.Module):
    """Dynamics model with optional task embedding.

    Predicts the next latent state given current state and action using
    a residual connection: z_next = z + delta.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_tasks: int = 1,
        task_dim: int = 96,
    ):
        """Initialize dynamics model.

        Args:
            latent_dim: Latent state dimension.
            action_dim: Action dimension.
            hidden_dim: Hidden layer dimension.
            num_tasks: Number of tasks for multi-task learning.
            task_dim: Task embedding dimension (used when num_tasks > 1).
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_tasks = num_tasks
        self.task_dim = task_dim

        # Compute input dimension
        input_dim = latent_dim + action_dim
        self.task_embedding: nn.Embedding | None = None
        if num_tasks > 1:
            input_dim += task_dim
            self.task_embedding = nn.Embedding(num_tasks, task_dim)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self,
        z: Tensor,
        action: Tensor,
        task_id: Tensor | None = None,
    ) -> Tensor:
        """Predict next latent state (without residual or SimNorm).

        This returns the delta prediction. The caller is responsible for
        adding the residual connection and applying SimNorm.

        Args:
            z: Current latent state of shape [batch, latent_dim].
            action: Action tensor of shape [batch, action_dim].
            task_id: Optional task ID tensor of shape [batch] for multi-task.

        Returns:
            Predicted latent delta of shape [batch, latent_dim].
        """
        z = torch.nan_to_num(
            z, nan=_DYNAMICS_NAN_FILL, posinf=_DYNAMICS_INPUT_CLAMP, neginf=-_DYNAMICS_INPUT_CLAMP
        ).clamp(-_DYNAMICS_INPUT_CLAMP, _DYNAMICS_INPUT_CLAMP)
        action = torch.nan_to_num(
            action,
            nan=_DYNAMICS_NAN_FILL,
            posinf=_DYNAMICS_INPUT_CLAMP,
            neginf=-_DYNAMICS_INPUT_CLAMP,
        ).clamp(-_DYNAMICS_INPUT_CLAMP, _DYNAMICS_INPUT_CLAMP)

        if self.task_embedding is not None:
            if task_id is None:
                task_emb = torch.zeros(
                    z.shape[0],
                    self.task_dim,
                    dtype=z.dtype,
                    device=z.device,
                )
            else:
                bounded_task_id = task_id.long().clamp(min=0, max=self.num_tasks - 1)
                task_emb = self.task_embedding(bounded_task_id).to(dtype=z.dtype)
            dynamics_input = torch.cat([z, action, task_emb], dim=-1)
        else:
            dynamics_input = torch.cat([z, action], dim=-1)

        dynamics_input = torch.nan_to_num(
            dynamics_input,
            nan=_DYNAMICS_NAN_FILL,
            posinf=_DYNAMICS_INPUT_CLAMP,
            neginf=-_DYNAMICS_INPUT_CLAMP,
        ).clamp(-_DYNAMICS_INPUT_CLAMP, _DYNAMICS_INPUT_CLAMP)

        delta = self.mlp(dynamics_input)
        delta = torch.nan_to_num(
            delta,
            nan=_DYNAMICS_NAN_FILL,
            posinf=_DYNAMICS_INPUT_CLAMP,
            neginf=-_DYNAMICS_INPUT_CLAMP,
        )

        # Limit residual update magnitude to avoid exploding latent transitions.
        delta_norm = delta.norm(dim=-1, keepdim=True)
        scale = torch.clamp(
            _DYNAMICS_DELTA_MAX_NORM / (delta_norm + _DYNAMICS_EPS),
            max=1.0,
        )
        return delta * scale
