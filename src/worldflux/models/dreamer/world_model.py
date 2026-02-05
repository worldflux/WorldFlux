"""DreamerV3 World Model implementation."""

import torch
import torch.nn.functional as F
from torch import Tensor

from ...core.batch import Batch
from ...core.config import DreamerV3Config
from ...core.interfaces import ActionConditioner, Decoder, DynamicsModel, ObservationEncoder
from ...core.model import WorldModel
from ...core.output import LossOutput, ModelOutput
from ...core.payloads import ActionPayload, ConditionPayload, WorldModelInput
from ...core.registry import WorldModelRegistry
from ...core.spec import (
    ActionSpec,
    Capability,
    ModalityKind,
    ModalitySpec,
    ModelIOContract,
    ObservationSpec,
    PredictionSpec,
    SequenceLayout,
    StateSpec,
)
from ...core.state import State
from .decoder import CNNDecoder, MLPDecoder
from .encoder import CNNEncoder, MLPEncoder
from .heads import ContinueHead, RewardHead, symlog
from .rssm import RSSM


class _DreamerObservationEncoder(ObservationEncoder):
    def __init__(self, model: "DreamerV3WorldModel"):
        self.model = model

    def encode(self, observations: dict[str, Tensor]) -> State:
        obs = observations.get("obs")
        if obs is None:
            raise ValueError("DreamerV3 encoder requires 'obs'")
        return self.model._encode_tensor(obs)


class _DreamerActionConditioner(ActionConditioner):
    def __init__(self, action_dim: int):
        self.action_dim = action_dim

    def condition(
        self,
        state: State,
        action: ActionPayload | None,
        conditions: ConditionPayload | None = None,
    ) -> dict[str, Tensor]:
        del state, conditions
        if action is None:
            return {}
        if action.tensor is not None:
            return {"action": action.tensor}
        return {}


class _DreamerDynamics(DynamicsModel):
    def __init__(self, model: "DreamerV3WorldModel"):
        self.model = model

    def transition(
        self, state: State, conditioned: dict[str, Tensor], deterministic: bool = False
    ) -> State:
        action = conditioned.get("action")
        if action is None:
            deter = state.tensors.get("deter")
            if deter is None:
                raise ValueError("DreamerV3 state missing 'deter'")
            action = torch.zeros(deter.shape[0], self.model.config.action_dim, device=deter.device)
        return self.model.rssm.prior_step(state, action, deterministic=deterministic)


class _DreamerDecoder(Decoder):
    def __init__(self, model: "DreamerV3WorldModel"):
        self.model = model

    def decode(self, state: State, conditions: ConditionPayload | None = None) -> dict[str, Tensor]:
        del conditions
        features = self.model._features(state)
        return {
            "obs": self.model.decoder(features),
            "reward": self.model.reward_head(features),
            "continue": self.model.continue_head(features),
        }


@WorldModelRegistry.register("dreamer", DreamerV3Config)
class DreamerV3WorldModel(WorldModel):
    """
    DreamerV3 world model implementation.

    Components:
        - Encoder: observation -> embedding
        - RSSM: latent state transitions
        - Decoder: latent state -> observation reconstruction
        - Reward Head: reward prediction
        - Continue Head: continue probability prediction
    """

    def __init__(self, config: DreamerV3Config):
        super().__init__()
        self.config = config
        self.capabilities = {
            Capability.LATENT_DYNAMICS,
            Capability.OBS_DECODER,
            Capability.REPRESENTATION,
            Capability.REWARD_PRED,
            Capability.CONTINUE_PRED,
        }

        # Encoder
        self.encoder: CNNEncoder | MLPEncoder
        if config.encoder_type == "cnn":
            self.encoder = CNNEncoder(
                obs_shape=config.obs_shape,
                depth=config.cnn_depth,
                kernels=config.cnn_kernels,
            )
        else:
            obs_dim = (
                config.obs_shape[0]
                if len(config.obs_shape) == 1
                else int(torch.prod(torch.tensor(config.obs_shape)).item())
            )
            self.encoder = MLPEncoder(
                input_dim=obs_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.deter_dim,
            )

        # RSSM
        self.rssm = RSSM(
            embed_dim=self.encoder.output_dim,
            action_dim=config.action_dim,
            deter_dim=config.deter_dim,
            stoch_discrete=config.stoch_discrete,
            stoch_classes=config.stoch_classes,
            hidden_dim=config.hidden_dim,
        )

        # Decoder
        feature_dim = self.rssm.feature_dim
        self.decoder: CNNDecoder | MLPDecoder
        if config.decoder_type == "cnn":
            self.decoder = CNNDecoder(
                feature_dim=feature_dim,
                obs_shape=config.obs_shape,
                depth=config.cnn_depth,
                kernels=config.cnn_kernels,
            )
        else:
            self.decoder = MLPDecoder(
                feature_dim=feature_dim,
                output_shape=config.obs_shape,
                hidden_dim=config.hidden_dim,
            )

        # Prediction heads
        self.reward_head = RewardHead(
            feature_dim=feature_dim,
            hidden_dim=config.hidden_dim,
            use_symlog=config.use_symlog,
        )
        self.continue_head = ContinueHead(
            feature_dim=feature_dim,
            hidden_dim=config.hidden_dim,
        )

        self.register_buffer("_device_tracker", torch.empty(0))

        # Universal component graph (v0.2).
        self.observation_encoder = _DreamerObservationEncoder(self)
        self.action_conditioner = _DreamerActionConditioner(config.action_dim)
        self.dynamics_model = _DreamerDynamics(self)
        self.decoder_module = _DreamerDecoder(self)

    @property
    def device(self) -> torch.device:
        device = self._device_tracker.device
        if not isinstance(device, torch.device):
            raise TypeError(f"Expected torch.device, got {type(device)}")
        return device

    def io_contract(self) -> ModelIOContract:
        obs_kind = ModalityKind.IMAGE if self.config.encoder_type == "cnn" else ModalityKind.VECTOR
        return ModelIOContract(
            observation_spec=ObservationSpec(
                modalities={"obs": ModalitySpec(kind=obs_kind, shape=self.config.obs_shape)}
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
                    "deter": ModalitySpec(kind=ModalityKind.VECTOR, shape=(self.config.deter_dim,)),
                    "stoch": ModalitySpec(
                        kind=ModalityKind.VECTOR,
                        shape=(self.config.stoch_discrete, self.config.stoch_classes),
                    ),
                }
            ),
            prediction_spec=PredictionSpec(
                tensors={
                    "obs": ModalitySpec(kind=obs_kind, shape=self.config.obs_shape),
                    "reward": ModalitySpec(kind=ModalityKind.VECTOR, shape=(1,)),
                    "continue": ModalitySpec(kind=ModalityKind.VECTOR, shape=(1,)),
                }
            ),
            sequence_layout=SequenceLayout(
                axes_by_field={
                    "obs": "BT...",
                    "actions": "BT...",
                    "rewards": "BT",
                    "terminations": "BT",
                    "next_obs": "BT...",
                }
            ),
            required_batch_keys=("obs", "actions", "rewards", "terminations"),
            required_state_keys=("deter", "stoch"),
        )

    def _encode_tensor(self, obs: Tensor) -> State:
        embed = self.encoder(obs)
        batch_size = obs.shape[0]
        initial = self.rssm.initial_state(batch_size, obs.device)
        zero_action = torch.zeros(batch_size, self.config.action_dim, device=obs.device)
        return self.rssm.posterior_step(initial, zero_action, embed)

    def encode(
        self,
        obs: Tensor | dict[str, Tensor] | WorldModelInput,
        deterministic: bool = False,
    ) -> State:
        """Encode observation to latent state."""
        del deterministic
        if isinstance(obs, WorldModelInput):
            return super().encode(obs)
        if isinstance(obs, dict):
            if "obs" in obs:
                obs = obs["obs"]
            else:
                raise ValueError("DreamerV3 expects observation tensor or dict with 'obs' key")
        return self._encode_tensor(obs)

    def transition(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ) -> State:
        """Predict next state (prior, for imagination)."""
        del conditions
        action_tensor = self.action_tensor_or_none(action)
        if action_tensor is None:
            deter = state.tensors.get("deter")
            if deter is None:
                raise ValueError("DreamerV3 state must contain 'deter' for default action")
            action_tensor = torch.zeros(deter.shape[0], self.config.action_dim, device=deter.device)
        return self.rssm.prior_step(state, action_tensor, deterministic=deterministic)

    def update(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        obs: Tensor | dict[str, Tensor] | WorldModelInput,
        conditions: ConditionPayload | None = None,
    ) -> State:
        """Update state with observation (posterior)."""
        del conditions
        if isinstance(obs, WorldModelInput):
            obs = obs.observations
        if isinstance(obs, dict):
            if "obs" in obs:
                obs = obs["obs"]
            else:
                raise ValueError("DreamerV3 expects observation tensor or dict with 'obs' key")
        action_tensor = self.action_tensor_or_none(action)
        if action_tensor is None:
            action_tensor = torch.zeros(obs.shape[0], self.config.action_dim, device=obs.device)
        embed = self.encoder(obs)
        return self.rssm.posterior_step(state, action_tensor, embed)

    def _features(self, state: State) -> Tensor:
        deter = state.tensors.get("deter")
        stoch = state.tensors.get("stoch")
        if deter is None or stoch is None:
            raise ValueError("DreamerV3 state must contain 'deter' and 'stoch'")
        if stoch.dim() == 3:
            stoch = stoch.flatten(start_dim=1)
        return torch.cat([deter, stoch], dim=-1)

    def decode(self, state: State, conditions: ConditionPayload | None = None) -> ModelOutput:
        """Decode latent state to predictions."""
        del conditions
        return super().decode(state)

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> State:
        """Create initial state."""
        if device is None:
            device = self.device
        return self.rssm.initial_state(batch_size, device)

    def loss(self, batch: Batch) -> LossOutput:
        """Compute training losses."""
        obs = batch.obs
        obs_tensor: Tensor | None
        if isinstance(obs, dict):
            obs_tensor = obs.get("obs")
        else:
            obs_tensor = obs
        if obs_tensor is None:
            raise ValueError("DreamerV3 requires obs tensor in batch")
        obs = obs_tensor
        actions = batch.actions
        rewards = batch.rewards
        terminations = batch.terminations
        if actions is None or rewards is None or terminations is None:
            raise ValueError("DreamerV3 requires actions, rewards, terminations in batch")
        continues = 1.0 - terminations

        batch_size, seq_len = obs.shape[:2]
        device = obs.device

        # Process sequence
        state = self.initial_state(batch_size, device)

        states = []
        for t in range(seq_len):
            if t == 0:
                action = torch.zeros(batch_size, self.config.action_dim, device=device)
            else:
                action = actions[:, t - 1]

            state = self.update(state, action, obs[:, t])
            states.append(state)

        components: dict[str, Tensor] = {}

        # KL loss with KL balancing (DreamerV3 paper section 3.4)
        # kl_balance controls the trade-off between representation learning and dynamics learning
        # kl_balance=1.0: only dynamics learns, kl_balance=0.0: only representation learns
        kl_loss = torch.tensor(0.0, device=device)
        alpha = self.config.kl_balance

        for state in states:
            posterior_logits = state.tensors.get("posterior_logits")
            prior_logits = state.tensors.get("prior_logits")
            if posterior_logits is not None and prior_logits is not None:
                # Dynamics loss: stop gradient on posterior, dynamics learns to predict it
                kl_dynamics = self.rssm.latent_space.kl_divergence(
                    posterior_logits.detach(),  # Stop gradient on representation
                    prior_logits,
                    free_nats=self.config.kl_free,
                )
                # Representation loss: stop gradient on prior, encoder learns to match it
                kl_representation = self.rssm.latent_space.kl_divergence(
                    posterior_logits,
                    prior_logits.detach(),  # Stop gradient on dynamics
                    free_nats=self.config.kl_free,
                )
                # Balanced KL loss
                kl = alpha * kl_dynamics + (1 - alpha) * kl_representation
                kl_loss = kl_loss + kl.mean()
        components["kl"] = kl_loss / seq_len

        # Reconstruction loss
        recon_loss = torch.tensor(0.0, device=device)
        for t, state in enumerate(states):
            decoded = self.decode(state)

            if self.config.use_symlog:
                target = symlog(obs[:, t])
            else:
                target = obs[:, t]
            recon_loss = recon_loss + F.mse_loss(decoded.preds["obs"], target)
        components["reconstruction"] = recon_loss / seq_len

        # Reward loss
        reward_loss = torch.tensor(0.0, device=device)
        for t in range(1, seq_len):
            decoded = self.decode(states[t])
            target = rewards[:, t]
            if self.config.use_symlog:
                target = symlog(target)
            reward_loss = reward_loss + F.mse_loss(decoded.preds["reward"].squeeze(-1), target)
        components["reward"] = reward_loss / max(seq_len - 1, 1)

        # Continue loss
        continue_loss = torch.tensor(0.0, device=device)
        for t in range(1, seq_len):
            decoded = self.decode(states[t])
            target = continues[:, t]
            continue_loss = continue_loss + F.binary_cross_entropy_with_logits(
                decoded.preds["continue"].squeeze(-1), target
            )
        components["continue"] = continue_loss / max(seq_len - 1, 1)

        # Total loss
        total = (
            self.config.loss_scales["reconstruction"] * components["reconstruction"]
            + self.config.loss_scales["kl"] * components["kl"]
            + self.config.loss_scales["reward"] * components["reward"]
            + self.config.loss_scales["continue"] * components["continue"]
        )

        metrics = {k: v.item() for k, v in components.items()}
        return LossOutput(loss=total, components=components, metrics=metrics)

    def save_pretrained(self, path: str) -> None:
        import os

        os.makedirs(path, exist_ok=True)
        self.config.save(os.path.join(path, "config.json"))
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
