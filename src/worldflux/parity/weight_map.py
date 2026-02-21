"""Weight mapping between official checkpoints and WorldFlux implementations."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class WeightMapEntry:
    """Single mapping between an official weight key and a WorldFlux key."""

    official_key: str
    worldflux_key: str
    transpose: bool = False
    reshape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class WeightMap:
    """Mapping table for a model family."""

    family: str
    entries: tuple[WeightMapEntry, ...]

    def official_to_worldflux(self, official_state: dict[str, Tensor]) -> dict[str, Tensor]:
        """Convert official state dict to WorldFlux state dict."""
        result: dict[str, Tensor] = {}
        for entry in self.entries:
            if entry.official_key not in official_state:
                continue
            tensor = official_state[entry.official_key]
            if entry.transpose:
                tensor = tensor.t()
            if entry.reshape is not None:
                tensor = tensor.reshape(entry.reshape)
            result[entry.worldflux_key] = tensor
        return result

    def validate_coverage(
        self,
        official_keys: set[str],
        worldflux_keys: set[str],
    ) -> tuple[set[str], set[str]]:
        """Return (unmapped_official, unmapped_worldflux) key sets."""
        mapped_official = {e.official_key for e in self.entries}
        mapped_worldflux = {e.worldflux_key for e in self.entries}
        return official_keys - mapped_official, worldflux_keys - mapped_worldflux


def dreamerv3_weight_map() -> WeightMap:
    """Create weight mapping for DreamerV3 (JAX/Haiku -> PyTorch).

    JAX/Haiku uses row-major (features_in, features_out) for Dense layers,
    while PyTorch uses (features_out, features_in). Linear weight tensors
    must be transposed.

    RSSM structure (from rssm.py):
      gru: GRUCell(stoch_dim + action_dim, deter_dim)
      prior_net: Sequential(
          0: Linear(deter_dim, hidden_dim),
          1: LayerNorm(hidden_dim),
          2: SiLU(),
          3: Linear(hidden_dim, stoch_dim),
      )
      posterior_net: Sequential(
          0: Linear(deter_dim + embed_dim, hidden_dim),
          1: LayerNorm(hidden_dim),
          2: SiLU(),
          3: Linear(hidden_dim, stoch_dim),
      )

    Encoder (MLPEncoder):
      mlp: Sequential(
          0: Linear(input_dim, hidden_dim),
          1: LayerNorm(hidden_dim),
          2: SiLU(),
          3: Linear(hidden_dim, output_dim),
      )

    Decoder (MLPDecoder):
      mlp: Sequential(
          0: Linear(feature_dim, hidden_dim),
          1: LayerNorm(hidden_dim),
          2: SiLU(),
          3: Linear(hidden_dim, output_dim),
      )

    RewardHead:
      mlp: Sequential(
          0: Linear, 1: LN, 2: SiLU,
          3: Linear, 4: LN, 5: SiLU,
          6: Linear(hidden_dim, 1),
      )

    ContinueHead: same structure as RewardHead.
    """
    entries: list[WeightMapEntry] = []

    # --- Encoder (MLP) ---
    entries.extend(
        [
            WeightMapEntry("encoder/mlp/linear_0/w", "encoder.mlp.0.weight", transpose=True),
            WeightMapEntry("encoder/mlp/linear_0/b", "encoder.mlp.0.bias"),
            WeightMapEntry("encoder/mlp/layer_norm_0/scale", "encoder.mlp.1.weight"),
            WeightMapEntry("encoder/mlp/layer_norm_0/offset", "encoder.mlp.1.bias"),
            WeightMapEntry("encoder/mlp/linear_1/w", "encoder.mlp.3.weight", transpose=True),
            WeightMapEntry("encoder/mlp/linear_1/b", "encoder.mlp.3.bias"),
        ]
    )

    # --- RSSM GRU ---
    # GRUCell has weight_ih (3*hidden, input), weight_hh (3*hidden, hidden), bias_ih, bias_hh
    entries.extend(
        [
            WeightMapEntry("rssm/gru/input/w", "rssm.gru.weight_ih", transpose=True),
            WeightMapEntry("rssm/gru/input/b", "rssm.gru.bias_ih"),
            WeightMapEntry("rssm/gru/hidden/w", "rssm.gru.weight_hh", transpose=True),
            WeightMapEntry("rssm/gru/hidden/b", "rssm.gru.bias_hh"),
        ]
    )

    # --- RSSM Prior Net ---
    entries.extend(
        [
            WeightMapEntry("rssm/prior/linear/w", "rssm.prior_net.0.weight", transpose=True),
            WeightMapEntry("rssm/prior/linear/b", "rssm.prior_net.0.bias"),
            WeightMapEntry("rssm/prior/layer_norm/scale", "rssm.prior_net.1.weight"),
            WeightMapEntry("rssm/prior/layer_norm/offset", "rssm.prior_net.1.bias"),
            WeightMapEntry("rssm/prior/out/w", "rssm.prior_net.3.weight", transpose=True),
            WeightMapEntry("rssm/prior/out/b", "rssm.prior_net.3.bias"),
        ]
    )

    # --- RSSM Posterior Net ---
    entries.extend(
        [
            WeightMapEntry(
                "rssm/posterior/linear/w", "rssm.posterior_net.0.weight", transpose=True
            ),
            WeightMapEntry("rssm/posterior/linear/b", "rssm.posterior_net.0.bias"),
            WeightMapEntry("rssm/posterior/layer_norm/scale", "rssm.posterior_net.1.weight"),
            WeightMapEntry("rssm/posterior/layer_norm/offset", "rssm.posterior_net.1.bias"),
            WeightMapEntry("rssm/posterior/out/w", "rssm.posterior_net.3.weight", transpose=True),
            WeightMapEntry("rssm/posterior/out/b", "rssm.posterior_net.3.bias"),
        ]
    )

    # --- Decoder (MLP) ---
    entries.extend(
        [
            WeightMapEntry("decoder/mlp/linear_0/w", "decoder.mlp.0.weight", transpose=True),
            WeightMapEntry("decoder/mlp/linear_0/b", "decoder.mlp.0.bias"),
            WeightMapEntry("decoder/mlp/layer_norm_0/scale", "decoder.mlp.1.weight"),
            WeightMapEntry("decoder/mlp/layer_norm_0/offset", "decoder.mlp.1.bias"),
            WeightMapEntry("decoder/mlp/linear_1/w", "decoder.mlp.3.weight", transpose=True),
            WeightMapEntry("decoder/mlp/linear_1/b", "decoder.mlp.3.bias"),
        ]
    )

    # --- Reward Head ---
    entries.extend(
        [
            WeightMapEntry("reward_head/linear_0/w", "reward_head.mlp.0.weight", transpose=True),
            WeightMapEntry("reward_head/linear_0/b", "reward_head.mlp.0.bias"),
            WeightMapEntry("reward_head/layer_norm_0/scale", "reward_head.mlp.1.weight"),
            WeightMapEntry("reward_head/layer_norm_0/offset", "reward_head.mlp.1.bias"),
            WeightMapEntry("reward_head/linear_1/w", "reward_head.mlp.3.weight", transpose=True),
            WeightMapEntry("reward_head/linear_1/b", "reward_head.mlp.3.bias"),
            WeightMapEntry("reward_head/layer_norm_1/scale", "reward_head.mlp.4.weight"),
            WeightMapEntry("reward_head/layer_norm_1/offset", "reward_head.mlp.4.bias"),
            WeightMapEntry("reward_head/out/w", "reward_head.mlp.6.weight", transpose=True),
            WeightMapEntry("reward_head/out/b", "reward_head.mlp.6.bias"),
        ]
    )

    # --- Continue Head ---
    entries.extend(
        [
            WeightMapEntry(
                "continue_head/linear_0/w", "continue_head.mlp.0.weight", transpose=True
            ),
            WeightMapEntry("continue_head/linear_0/b", "continue_head.mlp.0.bias"),
            WeightMapEntry("continue_head/layer_norm_0/scale", "continue_head.mlp.1.weight"),
            WeightMapEntry("continue_head/layer_norm_0/offset", "continue_head.mlp.1.bias"),
            WeightMapEntry(
                "continue_head/linear_1/w", "continue_head.mlp.3.weight", transpose=True
            ),
            WeightMapEntry("continue_head/linear_1/b", "continue_head.mlp.3.bias"),
            WeightMapEntry("continue_head/layer_norm_1/scale", "continue_head.mlp.4.weight"),
            WeightMapEntry("continue_head/layer_norm_1/offset", "continue_head.mlp.4.bias"),
            WeightMapEntry("continue_head/out/w", "continue_head.mlp.6.weight", transpose=True),
            WeightMapEntry("continue_head/out/b", "continue_head.mlp.6.bias"),
        ]
    )

    return WeightMap(family="dreamerv3", entries=tuple(entries))


def tdmpc2_weight_map() -> WeightMap:
    """Create weight mapping for TD-MPC2 (PyTorch -> WorldFlux PyTorch).

    Official TD-MPC2 uses slightly different naming conventions.
    No transpose needed since both use PyTorch.

    WorldFlux TDMPC2WorldModel exposes legacy aliases (world_model.py:175-181):
      self.encoder = self._encoder.mlp
      self.dynamics = self._dynamics.mlp
      self.task_embedding = self._dynamics.task_embedding
      self.reward_head = self._reward_head.mlp
      self.q_networks = ModuleList([q.mlp for q in ...])
      self.policy = self._policy.mlp

    Official TD-MPC2 naming:
      _encoder.0.weight -> encoder.0.weight
      _dynamics.0.weight -> dynamics.0.weight
      _Qs.0.mlp.0.weight -> q_networks.0.0.weight
      _pi.0.weight -> policy.0.weight
      _reward.0.weight -> reward_head.0.weight

    Encoder (MLPEncoder): mlp layers [0-6] = L,LN,Mish,L,LN,Mish,L
    Dynamics: mlp layers [0-6] = L,LN,Mish,L,LN,Mish,L
    RewardHead: mlp layers [0-3] = L,LN,Mish,L
    QNetwork: mlp layers [0-6] = L,LN,Mish,L,LN,Mish,L
    PolicyHead: mlp layers [0-4] = L,LN,Mish,L,Tanh
    """
    entries: list[WeightMapEntry] = []

    # Helper for MLP layers
    def _mlp_entries(
        official_prefix: str,
        worldflux_prefix: str,
        layer_indices: list[int],
    ) -> None:
        for idx in layer_indices:
            entries.append(
                WeightMapEntry(
                    f"{official_prefix}.{idx}.weight",
                    f"{worldflux_prefix}.{idx}.weight",
                )
            )
            entries.append(
                WeightMapEntry(
                    f"{official_prefix}.{idx}.bias",
                    f"{worldflux_prefix}.{idx}.bias",
                )
            )

    # --- Encoder ---
    # Linear layers at 0, 3, 6; LayerNorm at 1, 4
    _mlp_entries("_encoder", "encoder", [0, 1, 3, 4, 6])

    # --- Dynamics ---
    _mlp_entries("_dynamics", "dynamics", [0, 1, 3, 4, 6])

    # --- Reward Head ---
    # Linear at 0; LayerNorm at 1; Linear at 3
    _mlp_entries("_reward", "reward_head", [0, 1, 3])

    # --- Q Ensemble (5 networks by default) ---
    for qi in range(5):
        for layer_idx in [0, 1, 3, 4, 6]:
            entries.append(
                WeightMapEntry(
                    f"_Qs.{qi}.mlp.{layer_idx}.weight",
                    f"q_networks.{qi}.{layer_idx}.weight",
                )
            )
            entries.append(
                WeightMapEntry(
                    f"_Qs.{qi}.mlp.{layer_idx}.bias",
                    f"q_networks.{qi}.{layer_idx}.bias",
                )
            )

    # --- Policy ---
    # Linear at 0; LayerNorm at 1; Linear at 3 (Tanh at 4 has no params)
    _mlp_entries("_pi", "policy", [0, 1, 3])

    return WeightMap(family="tdmpc2", entries=tuple(entries))
