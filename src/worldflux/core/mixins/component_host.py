"""Component host mixin for the 5-layer pluggable architecture."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..interfaces import (
        ActionConditioner,
        Decoder,
        DynamicsModel,
        ObservationEncoder,
        RolloutExecutor,
    )


class ComponentHostMixin:
    """Mixin that provides the 5-layer pluggable component slots.

    Slots:
        1. observation_encoder — encodes raw observations into latent State.
        2. action_conditioner — fuses action + condition info for dynamics.
        3. dynamics_model — predicts next latent state.
        4. decoder_module — maps latent state to observable predictions.
        5. rollout_executor — executes multi-step open-loop rollouts.

    Models that want pluggable component composition should inherit from
    this mixin and set the slots in ``__init__``.

    Example::

        class MyModel(WorldModel, ComponentHostMixin):
            def __init__(self, config):
                super().__init__()
                self.init_component_slots()
                self.observation_encoder = MyEncoder(config)
    """

    observation_encoder: ObservationEncoder | None
    action_conditioner: ActionConditioner | None
    dynamics_model: DynamicsModel | None
    decoder_module: Decoder | None
    rollout_executor: RolloutExecutor | None
    composable_support: set[str]

    def init_component_slots(self) -> None:
        """Initialize all component slots to ``None``."""
        self.observation_encoder = None
        self.action_conditioner = None
        self.dynamics_model = None
        self.decoder_module = None
        self.rollout_executor = None
        self.composable_support = set()

    def swap_component(self, slot: str, component: Any) -> Any:
        """Replace a component slot and return the previous occupant.

        Args:
            slot: Name of the slot (e.g. ``"observation_encoder"``).
            component: New component instance, or ``None`` to clear.

        Returns:
            The previous component that occupied the slot.

        Raises:
            ValueError: If ``slot`` is not a recognized component slot.
        """
        valid_slots = {
            "observation_encoder",
            "action_conditioner",
            "dynamics_model",
            "decoder_module",
            "rollout_executor",
        }
        if slot not in valid_slots:
            raise ValueError(
                f"Unknown component slot: {slot!r}. " f"Valid slots: {sorted(valid_slots)}"
            )
        previous = getattr(self, slot, None)
        setattr(self, slot, component)
        if component is not None:
            self.composable_support.add(slot)
        else:
            self.composable_support.discard(slot)
        return previous
