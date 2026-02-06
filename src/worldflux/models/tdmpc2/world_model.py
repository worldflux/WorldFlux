"""TD-MPC2 World Model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...core.batch import Batch
from ...core.config import TDMPC2Config
from ...core.interfaces import ActionConditioner, Decoder, DynamicsModel, ObservationEncoder
from ...core.latent_space import SimNormLatentSpace
from ...core.model import WorldModel
from ...core.output import LossOutput, ModelOutput
from ...core.payloads import ActionPayload, ConditionPayload, WorldModelInput
from ...core.registry import WorldModelRegistry
from ...core.spec import (
    ActionSpec,
    Capability,
    ConditionSpec,
    ModalityKind,
    ModalitySpec,
    ModelIOContract,
    ObservationSpec,
    PredictionSpec,
    SequenceLayout,
    StateSpec,
)
from ...core.state import State
from .dynamics import Dynamics
from .encoder import MLPEncoder
from .heads import PolicyHead, QEnsemble, RewardHead


class _TDMPC2ObservationEncoder(ObservationEncoder):
    def __init__(self, model: "TDMPC2WorldModel"):
        self.model = model

    def encode(self, observations: dict[str, Tensor]) -> State:
        obs = observations.get("obs")
        if obs is None:
            raise ValueError("TD-MPC2 encoder requires 'obs'")
        z = self.model._encoder(obs)
        z = self.model.latent_space.sample(z)
        return State(tensors={"latent": z}, meta={"latent_type": "simnorm"})


class _TDMPC2ActionConditioner(ActionConditioner):
    def condition(
        self,
        state: State,
        action: ActionPayload | None,
        conditions: ConditionPayload | None = None,
    ) -> dict[str, Tensor]:
        del state, conditions
        if action is not None and action.tensor is not None:
            return {"action": action.tensor}
        return {}


class _TDMPC2Dynamics(DynamicsModel):
    def __init__(self, model: "TDMPC2WorldModel"):
        self.model = model

    def transition(
        self, state: State, conditioned: dict[str, Tensor], deterministic: bool = False
    ) -> State:
        del deterministic
        z = state.tensors.get("latent")
        if z is None:
            raise ValueError("TD-MPC2 requires 'latent' in State")
        action = conditioned.get("action")
        if action is None:
            action = torch.zeros(z.shape[0], self.model.config.action_dim, device=z.device)
        z_delta = self.model._dynamics(z, action, None)
        z_next = self.model.latent_space.sample(z + z_delta)
        return State(tensors={"latent": z_next}, meta={"latent_type": "simnorm"})


class _TDMPC2Decoder(Decoder):
    def __init__(self, model: "TDMPC2WorldModel"):
        self.model = model

    def decode(self, state: State, conditions: ConditionPayload | None = None) -> dict[str, Tensor]:
        del conditions
        z = state.tensors.get("latent")
        if z is None:
            raise ValueError("TD-MPC2 requires 'latent' in State")
        action = self.model._policy(z)
        reward = self.model._reward_head(z, action)
        q_values = self.model._q_ensemble(z, action)
        return {
            "reward": reward,
            "continue": torch.ones_like(reward),
            "q_values": q_values,
            "action": action,
        }


@WorldModelRegistry.register("tdmpc2", TDMPC2Config)
class TDMPC2WorldModel(WorldModel):
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
        self.capabilities = {
            Capability.LATENT_DYNAMICS,
            Capability.VALUE,
            Capability.POLICY,
            Capability.REWARD_PRED,
            Capability.CONTINUE_PRED,
        }

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

        # Universal component graph (v0.2).
        self.observation_encoder = _TDMPC2ObservationEncoder(self)
        self.action_conditioner = _TDMPC2ActionConditioner()
        self.dynamics_model = _TDMPC2Dynamics(self)
        self.decoder_module = _TDMPC2Decoder(self)

    @property
    def device(self) -> torch.device:
        device = self._device_tracker.device
        if not isinstance(device, torch.device):
            raise TypeError(f"Expected torch.device, got {type(device)}")
        return device

    def io_contract(self) -> ModelIOContract:
        return ModelIOContract(
            observation_spec=ObservationSpec(
                modalities={
                    "obs": ModalitySpec(kind=ModalityKind.VECTOR, shape=self.config.obs_shape)
                }
            ),
            action_spec=ActionSpec(
                kind=self.config.action_type,
                dim=self.config.action_dim,
                discrete=self.config.action_type == "discrete",
                num_actions=self.config.action_dim
                if self.config.action_type == "discrete"
                else None,
            ),
            state_spec=StateSpec(
                tensors={
                    "latent": ModalitySpec(
                        kind=ModalityKind.VECTOR, shape=(self.config.latent_dim,)
                    )
                }
            ),
            prediction_spec=PredictionSpec(
                tensors={
                    "reward": ModalitySpec(kind=ModalityKind.VECTOR, shape=(1,)),
                    "continue": ModalitySpec(kind=ModalityKind.VECTOR, shape=(1,)),
                    "q_values": ModalitySpec(
                        kind=ModalityKind.VECTOR,
                        shape=(self.config.num_q_networks, 1),
                    ),
                    "action": ModalitySpec(
                        kind=ModalityKind.VECTOR,
                        shape=(self.config.action_dim,),
                    ),
                }
            ),
            condition_spec=ConditionSpec(allowed_extra_keys=()),
            sequence_layout=SequenceLayout(
                axes_by_field={
                    "obs": "BT...",
                    "actions": "BT...",
                    "rewards": "BT",
                    "terminations": "BT",
                    "next_obs": "BT...",
                }
            ),
            required_batch_keys=("obs", "actions", "rewards"),
            required_state_keys=("latent",),
        )

    def encode(
        self,
        obs: Tensor | dict[str, Tensor] | WorldModelInput,
        deterministic: bool = False,
    ) -> State:
        """Encode observation to SimNorm latent space."""
        del deterministic
        if isinstance(obs, WorldModelInput):
            return super().encode(obs)
        if isinstance(obs, dict):
            if "obs" in obs:
                obs = obs["obs"]
            else:
                raise ValueError("TD-MPC2 expects observation tensor or dict with 'obs' key")
        z = self._encoder(obs)
        z = self.latent_space.sample(z)

        return State(
            tensors={"latent": z},
            meta={"latent_type": "simnorm"},
        )

    def transition(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
        task_id: Tensor | None = None,
    ) -> State:
        """Predict next state."""
        self._validate_condition_payload(self.coerce_condition_payload(conditions))
        z = state.tensors.get("latent")
        if z is None:
            raise ValueError("TD-MPC2 requires 'latent' in State")

        action_tensor = self.action_tensor_or_none(action)
        if action_tensor is None:
            action_tensor = torch.zeros(z.shape[0], self.config.action_dim, device=z.device)

        # Residual prediction
        z_delta = self._dynamics(z, action_tensor, task_id)
        z_next = z + z_delta
        z_next = self.latent_space.sample(z_next)

        return State(
            tensors={"latent": z_next},
            meta={"latent_type": "simnorm"},
        )

    def update(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        obs: Tensor | dict[str, Tensor] | WorldModelInput,
        conditions: ConditionPayload | None = None,
    ) -> State:
        """TD-MPC2 directly encodes observations (no posterior like RSSM)."""
        del state, action, conditions
        return self.encode(obs)

    def decode(self, state: State, conditions: ConditionPayload | None = None) -> ModelOutput:
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
        del conditions
        return super().decode(state)

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> State:
        """Initial state (uniform SimNorm)."""
        if device is None:
            device = self.device

        z = torch.zeros(batch_size, self.config.latent_dim, device=device)
        z = self.latent_space.sample(z)

        return State(
            tensors={"latent": z},
            meta={"latent_type": "simnorm"},
        )

    def predict_q(self, state: State, action: Tensor) -> Tensor:
        """Predict Q-value ensemble."""
        z = state.tensors.get("latent")
        if z is None:
            raise ValueError("TD-MPC2 requires 'latent' in State")
        q_values = self._q_ensemble(z, action)
        return q_values.squeeze(-1)

    def predict_reward(self, state: State, action: Tensor) -> Tensor:
        """Predict reward."""
        z = state.tensors.get("latent")
        if z is None:
            raise ValueError("TD-MPC2 requires 'latent' in State")
        return self._reward_head(z, action).squeeze(-1)

    def loss(self, batch: Batch) -> LossOutput:
        """
        TD-MPC2 loss computation.

        TD learning + latent consistency loss (BYOL-style).

        The encoder receives gradients from:
        - Consistency loss: through z_t -> dynamics -> z_pred path
        - Reward loss: through z_t -> reward_head path
        - TD loss: through z_t -> Q-ensemble path

        Targets are computed with no_grad (BYOL-style self-supervised learning).
        """
        obs = batch.obs
        obs_tensor: Tensor | None
        if isinstance(obs, dict):
            obs_tensor = obs.get("obs")
        else:
            obs_tensor = obs
        if obs_tensor is None:
            raise ValueError("TD-MPC2 requires obs tensor in batch")
        obs = obs_tensor
        actions = batch.actions
        rewards = batch.rewards
        if actions is None or rewards is None:
            raise ValueError("TD-MPC2 requires actions and rewards in batch")

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

        components: dict[str, Tensor] = {}

        # Latent consistency loss
        consistency_loss = torch.tensor(0.0, device=device)
        for t in range(seq_len - 1):
            z_t = z_all[:, t]  # Current encoding (with gradient)
            state_t = State(tensors={"latent": z_t}, meta={"latent_type": "simnorm"})

            pred_state = self.transition(state_t, actions[:, t])
            z_pred = pred_state.tensors.get("latent")
            if z_pred is None:
                raise ValueError(f"Predicted state at step {t} has no latent")

            z_target = z_targets[:, t]  # Target (no gradient)
            consistency_loss = consistency_loss + F.mse_loss(z_pred, z_target)

        components["consistency"] = consistency_loss / max(seq_len - 1, 1)

        # Reward loss
        reward_loss = torch.tensor(0.0, device=device)
        for t in range(seq_len - 1):
            z_t = z_all[:, t]  # With gradient
            state_t = State(tensors={"latent": z_t}, meta={"latent_type": "simnorm"})

            pred_reward = self.predict_reward(state_t, actions[:, t])
            reward_loss = reward_loss + F.mse_loss(pred_reward, rewards[:, t + 1])

        components["reward"] = reward_loss / max(seq_len - 1, 1)

        # TD loss
        td_loss = torch.tensor(0.0, device=device)
        gamma = self.config.gamma
        for t in range(seq_len - 1):
            z_t = z_all[:, t]  # With gradient for Q-network
            state_t = State(tensors={"latent": z_t}, meta={"latent_type": "simnorm"})

            q_values = self.predict_q(state_t, actions[:, t])

            with torch.no_grad():
                z_next = z_targets[:, t]  # No gradient for target
                state_next = State(tensors={"latent": z_next}, meta={"latent_type": "simnorm"})
                next_action = self._policy(z_next)
                q_next = self.predict_q(state_next, next_action).min(dim=0)[0]
                target = rewards[:, t + 1] + gamma * q_next

            td_loss = td_loss + F.mse_loss(q_values.mean(dim=0), target)

        components["td"] = td_loss / max(seq_len - 1, 1)

        total = components["consistency"] + components["reward"] + components["td"]
        metrics = {k: v.item() for k, v in components.items()}
        return LossOutput(loss=total, components=components, metrics=metrics)

    def save_pretrained(self, path: str) -> None:
        super().save_pretrained(path)
