"""DreamerV3 World Model implementation."""

from __future__ import annotations

import copy

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
from .decoder import CNNDecoder, MLPDecoder
from .encoder import CNNEncoder, MLPEncoder
from .heads import (
    ContinueHead,
    ContinuousActorHead,
    CriticHead,
    DiscreteActorHead,
    RewardHead,
    compute_td_lambda,
    symlog,
    twohot_encode,
)
from .rssm import RSSM


class _DreamerObservationEncoder(ObservationEncoder):
    def __init__(self, model: DreamerV3WorldModel):
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
    def __init__(self, model: DreamerV3WorldModel):
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
    def __init__(self, model: DreamerV3WorldModel):
        self.model = model

    def decode(self, state: State, conditions: ConditionPayload | None = None) -> dict[str, Tensor]:
        del conditions
        features = self.model._features(state)
        # Return scalar reward for rollout/inference consumers.
        # The loss() method calls reward_head directly for raw logits.
        return {
            "obs": self.model.decoder(features),
            "reward": self.model.reward_head.predict(features).unsqueeze(-1),
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
            use_twohot=config.use_twohot,
            num_bins=config.reward_num_bins,
            bin_min=config.reward_bin_min,
            bin_max=config.reward_bin_max,
        )
        self.continue_head = ContinueHead(
            feature_dim=feature_dim,
            hidden_dim=config.hidden_dim,
        )

        self.register_buffer("_device_tracker", torch.empty(0))

        # Actor-Critic (gated by config.actor_critic)
        if config.actor_critic:
            self.capabilities.add(Capability.POLICY)
            self.capabilities.add(Capability.VALUE)

            if config.action_type == "discrete":
                self.actor_head: DiscreteActorHead | ContinuousActorHead = DiscreteActorHead(
                    feature_dim=feature_dim,
                    action_dim=config.action_dim,
                    hidden_dim=config.hidden_dim,
                    entropy_coef=config.actor_entropy_coef,
                )
            else:
                self.actor_head = ContinuousActorHead(
                    feature_dim=feature_dim,
                    action_dim=config.action_dim,
                    hidden_dim=config.hidden_dim,
                    entropy_coef=config.actor_entropy_coef,
                )

            self.critic_head = CriticHead(
                feature_dim=feature_dim,
                hidden_dim=config.hidden_dim,
                num_bins=config.reward_num_bins,
                bin_min=config.reward_bin_min,
                bin_max=config.reward_bin_max,
            )
            self.slow_critic = copy.deepcopy(self.critic_head)
            for p in self.slow_critic.parameters():
                p.requires_grad = False

            # Running percentile for return normalisation
            self.register_buffer("_return_low", torch.tensor(0.0))
            self.register_buffer("_return_high", torch.tensor(1.0))

        # Universal component graph (v0.2).
        self.observation_encoder = _DreamerObservationEncoder(self)
        self.action_conditioner = _DreamerActionConditioner(config.action_dim)
        self.dynamics_model = _DreamerDynamics(self)
        self.decoder_module = _DreamerDecoder(self)
        self.composable_support = {
            "observation_encoder",
            "action_conditioner",
            "dynamics_model",
            "decoder",
            "rollout_executor",
        }
        if config.actor_critic:
            self.composable_support.update({"actor", "critic"})

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
        return super().transition(
            state,
            action,
            conditions=conditions,
            deterministic=deterministic,
        )

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

    def parameter_groups(self) -> list[dict]:
        """Return parameter groups with separate LRs for actor/critic."""
        if not self.config.actor_critic:
            return [{"params": list(self.parameters())}]
        actor_ids = {id(p) for p in self.actor_head.parameters()}
        critic_ids = {id(p) for p in self.critic_head.parameters()}
        ac_ids = actor_ids | critic_ids
        wm_params = [p for p in self.parameters() if id(p) not in ac_ids]
        return [
            {"params": wm_params, "lr": self.config.learning_rate},
            {"params": list(self.actor_head.parameters()), "lr": self.config.actor_lr},
            {"params": list(self.critic_head.parameters()), "lr": self.config.critic_lr},
        ]

    def decode(self, state: State, conditions: ConditionPayload | None = None) -> ModelOutput:
        """Decode latent state to predictions."""
        return super().decode(state, conditions=conditions)

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

        # KL losses (DreamerV3 paper section 3.4)
        # Dynamics loss (β_dyn=0.5): prior learns to predict posterior
        # Representation loss (β_rep=0.1): encoder learns to match prior
        dyn_loss = torch.tensor(0.0, device=device)
        rep_loss = torch.tensor(0.0, device=device)

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
                dyn_loss = dyn_loss + kl_dynamics.mean()
                rep_loss = rep_loss + kl_representation.mean()
        components["kl_dynamics"] = dyn_loss / seq_len
        components["kl_representation"] = rep_loss / seq_len
        # Combined KL for backward-compatible logging
        components["kl"] = components["kl_dynamics"] + components["kl_representation"]

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
            features = self._features(states[t])
            reward_logits = self.reward_head(features)
            target = rewards[:, t]
            if self.reward_head.use_twohot:
                symlog_target = symlog(target)
                twohot_target = twohot_encode(symlog_target, self.reward_head.bins)
                log_probs = F.log_softmax(reward_logits, dim=-1)
                reward_loss = reward_loss + -(twohot_target * log_probs).sum(dim=-1).mean()
            else:
                if self.config.use_symlog:
                    target = symlog(target)
                reward_loss = reward_loss + F.mse_loss(reward_logits.squeeze(-1), target)
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
            + self.config.loss_scales["kl_dynamics"] * components["kl_dynamics"]
            + self.config.loss_scales["kl_representation"] * components["kl_representation"]
            + self.config.loss_scales["reward"] * components["reward"]
            + self.config.loss_scales["continue"] * components["continue"]
        )

        # Actor-Critic losses (imagination)
        if self.config.actor_critic:
            ac = self._imagine_and_compute_ac_loss(states)
            components.update(ac)
            self._update_slow_critic()
            total = total + (
                self.config.loss_scales.get("actor", 1.0) * components["actor"]
                + self.config.loss_scales.get("critic", 1.0) * components["critic"]
            )

        metrics = {k: v.item() for k, v in components.items()}
        return LossOutput(loss=total, components=components, metrics=metrics)

    # ------------------------------------------------------------------
    # Actor-Critic helpers
    # ------------------------------------------------------------------

    def _imagine_and_compute_ac_loss(self, posterior_states: list[State]) -> dict[str, Tensor]:
        """Run imagination from posterior states and compute AC losses."""
        device = self.device
        horizon = self.config.imagination_horizon

        # Flatten all posterior states into a single batch for imagination
        all_deter = torch.cat([s.tensors["deter"] for s in posterior_states], dim=0)
        all_stoch = torch.cat([s.tensors["stoch"] for s in posterior_states], dim=0)
        imag_state = State(
            tensors={"deter": all_deter.detach(), "stoch": all_stoch.detach()},
            meta={"latent_type": "categorical"},
        )

        # Collect imagination trajectory
        imag_features_list: list[Tensor] = []
        log_probs_list: list[Tensor] = []

        for _ in range(horizon):
            features = self._features(imag_state)
            imag_features_list.append(features)
            action, log_prob = self.actor_head.sample(features)
            log_probs_list.append(log_prob)
            imag_state = self.rssm.prior_step(imag_state, action)

        # Final features for bootstrap value
        final_features = self._features(imag_state)
        imag_features_list.append(final_features)

        # Stack: (horizon+1, N, F) for features, (horizon, N) for log_probs
        imag_features = torch.stack(imag_features_list, dim=0)
        log_probs = torch.stack(log_probs_list, dim=0)

        # Compute targets with no_grad
        with torch.no_grad():
            rewards = torch.stack(
                [self.reward_head.predict(imag_features[t + 1]) for t in range(horizon)], dim=0
            )
            continues = torch.stack(
                [self.continue_head.predict(imag_features[t + 1]) for t in range(horizon)], dim=0
            )
            slow_values = torch.stack(
                [self.slow_critic.predict(imag_features[t]) for t in range(horizon + 1)], dim=0
            )
            returns = compute_td_lambda(
                rewards,
                slow_values,
                continues,
                gamma=self.config.gamma,
                lambda_=self.config.lambda_,
            )

            # Normalise returns
            if self.config.return_normalization:
                returns = self._normalize_returns(returns)

        # Critic loss: twohot CE per step
        critic_loss = torch.tensor(0.0, device=device)
        for t in range(horizon):
            critic_logits = self.critic_head(imag_features[t].detach())
            target_symlog = symlog(returns[t])
            target_twohot = twohot_encode(target_symlog, self.critic_head.bins)
            log_probs_critic = F.log_softmax(critic_logits, dim=-1)
            critic_loss = critic_loss - (target_twohot * log_probs_critic).sum(dim=-1).mean()
        critic_loss = critic_loss / horizon

        # Actor loss: REINFORCE + entropy
        with torch.no_grad():
            baseline = torch.stack(
                [self.critic_head.predict(imag_features[t]) for t in range(horizon)], dim=0
            )
            advantages = returns - baseline

        reinforce = -(log_probs * advantages).mean()
        entropy = torch.stack(
            [self.actor_head.entropy(imag_features[t]) for t in range(horizon)], dim=0
        ).mean()
        actor_loss = reinforce - self.actor_head.entropy_coef * entropy

        return {"actor": actor_loss, "critic": critic_loss}

    def _update_slow_critic(self) -> None:
        """EMA update for slow critic target network."""
        frac = self.config.slow_critic_fraction
        for slow_p, fast_p in zip(self.slow_critic.parameters(), self.critic_head.parameters()):
            slow_p.data.mul_(1 - frac).add_(fast_p.data, alpha=frac)

    def _normalize_returns(self, returns: Tensor) -> Tensor:
        """Normalise returns to [0, 1] using running percentile (5th/95th)."""
        momentum = 0.99
        flat = returns.detach().flatten()
        low = torch.quantile(flat, 0.05)
        high = torch.quantile(flat, 0.95)
        ret_low: Tensor = self._return_low  # type: ignore[assignment]
        ret_high: Tensor = self._return_high  # type: ignore[assignment]
        ret_low.mul_(momentum).add_(low, alpha=1 - momentum)
        ret_high.mul_(momentum).add_(high, alpha=1 - momentum)
        span = (ret_high - ret_low).clamp(min=1.0)
        return (returns - ret_low) / span

    def save_pretrained(self, path: str) -> None:
        super().save_pretrained(path)
