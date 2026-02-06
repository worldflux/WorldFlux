"""Smoke tests for universal-architecture model families."""

from __future__ import annotations

import pytest
import torch

from worldflux import create_world_model
from worldflux.core.batch import Batch
from worldflux.core.exceptions import ConfigurationError, ValidationError
from worldflux.core.interfaces import ComponentSpec
from worldflux.core.payloads import ConditionPayload
from worldflux.core.registry import WorldModelRegistry
from worldflux.training import Trainer, TrainingConfig


class _SmokeProvider:
    def __init__(self, obs_dim: int, action_dim: int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def sample(
        self, batch_size: int, seq_len: int | None = None, device: str | torch.device = "cpu"
    ) -> Batch:
        t = seq_len or 3
        obs = torch.randn(batch_size, t, self.obs_dim, device=device)
        next_obs = torch.randn(batch_size, t, self.obs_dim, device=device)
        actions = torch.randn(batch_size, t, self.action_dim, device=device)
        rewards = torch.randn(batch_size, t, device=device)
        terminations = torch.zeros(batch_size, t, device=device)
        return Batch(
            obs=obs,
            next_obs=next_obs,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            context=obs,
            target=next_obs,
            layouts={
                "obs": "BT...",
                "next_obs": "BT...",
                "actions": "BT...",
                "rewards": "BT",
                "terminations": "BT",
                "context": "BT...",
                "target": "BT...",
            },
            strict_layout=True,
        )


@pytest.mark.parametrize(
    "model_id",
    [
        "vjepa2:base",
        "token:base",
        "diffusion:base",
        "dit:base",
        "ssm:base",
        "renderer3d:base",
        "physics:base",
        "gan:base",
    ],
)
def test_model_family_smoke_train_one_step(model_id: str, tmp_path):
    with pytest.warns(DeprecationWarning):
        model = create_world_model(model_id, obs_shape=(4,), action_dim=2, api_version="v0.2")

    provider = _SmokeProvider(obs_dim=4, action_dim=2)
    cfg = TrainingConfig(
        total_steps=1,
        batch_size=3,
        sequence_length=3,
        output_dir=str(tmp_path / model_id.replace(":", "-")),
        device="cpu",
    )
    trainer = Trainer(model, cfg)
    trained = trainer.train(provider, num_steps=1)
    assert trained is model


@pytest.mark.parametrize(
    "model_id,kwargs",
    [
        (
            "dreamerv3:size12m",
            {
                "obs_shape": (4,),
                "action_dim": 2,
                "encoder_type": "mlp",
                "decoder_type": "mlp",
                "hidden_dim": 64,
                "deter_dim": 64,
                "stoch_discrete": 4,
                "stoch_classes": 4,
            },
        ),
        ("tdmpc2:5m", {"obs_shape": (4,), "action_dim": 2, "hidden_dim": 64, "latent_dim": 32}),
        ("jepa:base", {"obs_shape": (4,), "action_dim": 2}),
        ("vjepa2:base", {"obs_shape": (4,), "action_dim": 2}),
        ("token:base", {"obs_shape": (4,), "action_dim": 2}),
        ("diffusion:base", {"obs_shape": (4,), "action_dim": 2}),
    ],
)
def test_v3_transition_rejects_undeclared_condition_extras(
    model_id: str, kwargs: dict[str, object]
):
    model = create_world_model(model_id, api_version="v3", **kwargs)
    if model_id == "token:base":
        obs = torch.randint(0, 8, (2, 4))
    else:
        obs = torch.randn(2, 4)
    action = torch.randn(2, 2)
    state = model.encode(obs)

    with pytest.raises(ValidationError, match="undeclared keys"):
        model.transition(
            state,
            action,
            conditions=ConditionPayload(extras={"wf.custom.condition": torch.ones(1)}),
        )


@pytest.mark.parametrize(
    "model_id,kwargs",
    [
        (
            "dreamerv3:size12m",
            {"obs_shape": (4,), "action_dim": 2, "encoder_type": "mlp", "decoder_type": "mlp"},
        ),
        ("tdmpc2:5m", {"obs_shape": (4,), "action_dim": 2}),
        ("jepa:base", {"obs_shape": (4,), "action_dim": 2}),
        ("vjepa2:base", {"obs_shape": (4,), "action_dim": 2}),
        ("token:base", {"obs_shape": (4,), "action_dim": 2}),
        ("diffusion:base", {"obs_shape": (4,), "action_dim": 2}),
    ],
)
def test_model_contracts_expose_condition_spec(model_id: str, kwargs: dict[str, object]):
    model = create_world_model(model_id, api_version="v3", **kwargs)
    contract = model.io_contract()
    contract.validate()
    assert contract.condition_spec is not None
    assert isinstance(contract.condition_spec.allowed_extra_keys, tuple)


def test_factory_component_overrides_support_one_step_training(tmp_path):
    class _ZeroActionConditioner:
        def condition(self, state, action, conditions=None):
            del state, conditions
            if action is None or action.tensor is None:
                return {}
            return {"action": torch.zeros_like(action.tensor)}

    component_id = "tests.override.zero_action_conditioner"
    WorldModelRegistry.register_component(
        component_id,
        _ZeroActionConditioner,
        ComponentSpec(name="Zero Action Conditioner", component_type="action_conditioner"),
    )
    try:
        model = create_world_model(
            "tdmpc2:ci",
            obs_shape=(4,),
            action_dim=2,
            api_version="v3",
            component_overrides={"action_conditioner": component_id},
        )
        assert model.action_conditioner is not None
        assert type(model.action_conditioner).__name__ == "_ZeroActionConditioner"

        provider = _SmokeProvider(obs_dim=4, action_dim=2)
        cfg = TrainingConfig(
            total_steps=1,
            batch_size=3,
            sequence_length=3,
            output_dir=str(tmp_path / "override-smoke"),
            device="cpu",
        )
        trainer = Trainer(model, cfg)
        trained = trainer.train(provider, num_steps=1)
        assert trained is model
    finally:
        WorldModelRegistry.unregister_component(component_id)


@pytest.mark.parametrize(
    "model_id,kwargs,state_key",
    [
        (
            "dreamerv3:size12m",
            {
                "obs_shape": (4,),
                "action_dim": 2,
                "encoder_type": "mlp",
                "decoder_type": "mlp",
                "hidden_dim": 64,
                "deter_dim": 64,
                "stoch_discrete": 4,
                "stoch_classes": 4,
            },
            "deter",
        ),
        (
            "tdmpc2:5m",
            {"obs_shape": (4,), "action_dim": 2, "hidden_dim": 64, "latent_dim": 32},
            "latent",
        ),
    ],
)
def test_component_overrides_change_transition_dynamics(
    model_id: str,
    kwargs: dict[str, object],
    state_key: str,
):
    class _ZeroActionConditioner:
        def condition(self, state, action, conditions=None):
            del state, conditions
            if action is None or action.tensor is None:
                return {}
            return {"action": torch.zeros_like(action.tensor)}

    component_id = "tests.override.transition.zero_action_conditioner"
    WorldModelRegistry.register_component(
        component_id,
        _ZeroActionConditioner,
        ComponentSpec(name="Zero Action Conditioner", component_type="action_conditioner"),
    )

    try:
        torch.manual_seed(0)
        base_model = create_world_model(model_id, api_version="v3", **kwargs)
        override_model = create_world_model(
            model_id,
            api_version="v3",
            component_overrides={"action_conditioner": component_id},
            **kwargs,
        )
        override_model.load_state_dict(base_model.state_dict())

        obs = torch.randn(2, 4)
        action = torch.randn(2, 2)
        base_state = base_model.encode(obs)
        override_state = override_model.encode(obs)

        base_next = base_model.transition(base_state, action, deterministic=True)
        override_next = override_model.transition(override_state, action, deterministic=True)

        assert state_key in base_next.tensors
        assert state_key in override_next.tensors
        assert not torch.allclose(base_next.tensors[state_key], override_next.tensors[state_key])
    finally:
        WorldModelRegistry.unregister_component(component_id)


def test_component_overrides_fail_for_non_composable_family():
    class _NoOpConditioner:
        def condition(self, state, action, conditions=None):
            del state, action, conditions
            return {}

    component_id = "tests.override.non_composable.noop"
    WorldModelRegistry.register_component(
        component_id,
        _NoOpConditioner,
        ComponentSpec(name="No-op Conditioner", component_type="action_conditioner"),
    )
    try:
        with pytest.raises(ConfigurationError, match="not supported by model"):
            create_world_model(
                "jepa:base",
                obs_shape=(4,),
                action_dim=2,
                api_version="v3",
                component_overrides={"action_conditioner": component_id},
            )
    finally:
        WorldModelRegistry.unregister_component(component_id)
