"""Tests for capability helpers."""

from worldflux.core.config import DreamerV3Config, JEPABaseConfig, TDMPC2Config
from worldflux.core.spec import Capability
from worldflux.models.dreamer import DreamerV3WorldModel
from worldflux.models.jepa import JEPABaseWorldModel
from worldflux.models.tdmpc2 import TDMPC2WorldModel


def test_supports_reward_continue_flags():
    dreamer = DreamerV3WorldModel(
        DreamerV3Config.from_size(
            "ci",
            obs_shape=(4,),
            action_dim=2,
            encoder_type="mlp",
            decoder_type="mlp",
            hidden_dim=32,
            deter_dim=32,
            stoch_dim=4,
            stoch_classes=4,
            stoch_discrete=True,
        )
    )
    tdmpc2 = TDMPC2WorldModel(TDMPC2Config.from_size("ci", obs_shape=(4,), action_dim=2))
    jepa = JEPABaseWorldModel(
        JEPABaseConfig(obs_shape=(4,), action_dim=2, encoder_dim=32, predictor_dim=32)
    )

    assert dreamer.supports_reward
    assert dreamer.supports_continue
    assert Capability.REWARD_PRED in dreamer.capabilities

    assert tdmpc2.supports_reward
    assert tdmpc2.supports_continue
    assert Capability.REWARD_PRED in tdmpc2.capabilities

    assert not jepa.supports_reward
    assert not jepa.supports_continue
