"""Official DreamerV3 Atari100k recipe constants used by parity runtime."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DreamerOfficialAtari100kRecipe:
    """Reference recipe taken from official dreamerv3/configs.yaml atari100k."""

    recipe_id: str = "official_dreamerv3_atari100k"
    recipe_source: str = "dreamerv3/configs.yaml#atari100k"
    steps: int = 110_000
    envs: int = 1
    train_ratio: float = 256.0
    batch_size: int = 16
    batch_length: int = 64
    report_length: int = 32
    replay_size: int = 5_000_000
    replay_chunksize: int = 1024
    log_every: int = 120
    report_every: int = 300
    save_every: int = 900
    eval_eps: int = 1
    learning_rate: float = 4e-5
    model_profile: str = "official_xl"
    action_type: str = "discrete"
    max_episode_steps: int = 27_000

    def to_metadata(self) -> dict[str, Any]:
        return asdict(self)


OFFICIAL_DREAMER_ATARI100K_RECIPE = DreamerOfficialAtari100kRecipe()


__all__ = [
    "DreamerOfficialAtari100kRecipe",
    "OFFICIAL_DREAMER_ATARI100K_RECIPE",
]
