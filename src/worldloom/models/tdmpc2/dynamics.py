"""TD-MPC2 Dynamics Model."""

import torch
import torch.nn as nn
from torch import Tensor


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
        if self.task_embedding is not None and task_id is not None:
            task_emb = self.task_embedding(task_id)
            dynamics_input = torch.cat([z, action, task_emb], dim=-1)
        else:
            dynamics_input = torch.cat([z, action], dim=-1)

        return self.mlp(dynamics_input)
