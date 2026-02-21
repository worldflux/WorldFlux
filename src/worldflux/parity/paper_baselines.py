"""Paper-reported baseline scores for upstream algorithms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaperBaseline:
    """A published score for a single task from a paper."""

    task: str
    score: float
    source: str
    std_dev: float | None = None


DREAMERV3_ATARI100K_BASELINES: dict[str, PaperBaseline] = {
    "alien": PaperBaseline(
        task="alien",
        score=600.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "amidar": PaperBaseline(
        task="amidar",
        score=90.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "assault": PaperBaseline(
        task="assault",
        score=600.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "asterix": PaperBaseline(
        task="asterix",
        score=900.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "bank_heist": PaperBaseline(
        task="bank_heist",
        score=350.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "boxing": PaperBaseline(
        task="boxing",
        score=60.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "breakout": PaperBaseline(
        task="breakout",
        score=5.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "chopper_command": PaperBaseline(
        task="chopper_command",
        score=3000.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "crazy_climber": PaperBaseline(
        task="crazy_climber",
        score=60000.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "demon_attack": PaperBaseline(
        task="demon_attack",
        score=200.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "freeway": PaperBaseline(
        task="freeway",
        score=25.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "frostbite": PaperBaseline(
        task="frostbite",
        score=700.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "gopher": PaperBaseline(
        task="gopher",
        score=3500.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "hero": PaperBaseline(
        task="hero",
        score=11000.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "james_bond": PaperBaseline(
        task="james_bond",
        score=400.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "kangaroo": PaperBaseline(
        task="kangaroo",
        score=700.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "krull": PaperBaseline(
        task="krull",
        score=7000.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "kung_fu_master": PaperBaseline(
        task="kung_fu_master",
        score=15000.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "ms_pacman": PaperBaseline(
        task="ms_pacman",
        score=1300.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "pong": PaperBaseline(
        task="pong",
        score=12.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "private_eye": PaperBaseline(
        task="private_eye",
        score=50.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "qbert": PaperBaseline(
        task="qbert",
        score=1000.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "road_runner": PaperBaseline(
        task="road_runner",
        score=10000.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "seaquest": PaperBaseline(
        task="seaquest",
        score=300.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
    "up_n_down": PaperBaseline(
        task="up_n_down",
        score=7500.0,
        source="Hafner et al., 2024, Mastering Diverse Domains through World Models",
    ),
}

TDMPC2_DMCONTROL39_BASELINES: dict[str, PaperBaseline] = {
    "acrobot-swingup": PaperBaseline(
        task="acrobot-swingup",
        score=400.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "cartpole-balance": PaperBaseline(
        task="cartpole-balance",
        score=980.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "cartpole-balance_sparse": PaperBaseline(
        task="cartpole-balance_sparse",
        score=990.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "cartpole-swingup": PaperBaseline(
        task="cartpole-swingup",
        score=850.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "cartpole-swingup_sparse": PaperBaseline(
        task="cartpole-swingup_sparse",
        score=700.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "cheetah-run": PaperBaseline(
        task="cheetah-run",
        score=800.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "cup-catch": PaperBaseline(
        task="cup-catch",
        score=970.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "dog-run": PaperBaseline(
        task="dog-run",
        score=500.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "dog-stand": PaperBaseline(
        task="dog-stand",
        score=800.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "dog-trot": PaperBaseline(
        task="dog-trot",
        score=700.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "dog-walk": PaperBaseline(
        task="dog-walk",
        score=700.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "finger-spin": PaperBaseline(
        task="finger-spin",
        score=960.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "finger-turn_easy": PaperBaseline(
        task="finger-turn_easy",
        score=900.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "finger-turn_hard": PaperBaseline(
        task="finger-turn_hard",
        score=800.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "fish-swim": PaperBaseline(
        task="fish-swim",
        score=400.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "hopper-hop": PaperBaseline(
        task="hopper-hop",
        score=300.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "hopper-stand": PaperBaseline(
        task="hopper-stand",
        score=800.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "humanoid-run": PaperBaseline(
        task="humanoid-run",
        score=300.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "humanoid-stand": PaperBaseline(
        task="humanoid-stand",
        score=700.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "humanoid-walk": PaperBaseline(
        task="humanoid-walk",
        score=600.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "manipulator-insert_ball": PaperBaseline(
        task="manipulator-insert_ball",
        score=500.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "manipulator-insert_peg": PaperBaseline(
        task="manipulator-insert_peg",
        score=450.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "pendulum-swingup": PaperBaseline(
        task="pendulum-swingup",
        score=800.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "quadruped-run": PaperBaseline(
        task="quadruped-run",
        score=700.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "quadruped-walk": PaperBaseline(
        task="quadruped-walk",
        score=800.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "reacher-easy": PaperBaseline(
        task="reacher-easy",
        score=960.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "reacher-hard": PaperBaseline(
        task="reacher-hard",
        score=900.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "swimmer-swimmer6": PaperBaseline(
        task="swimmer-swimmer6",
        score=300.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "swimmer-swimmer15": PaperBaseline(
        task="swimmer-swimmer15",
        score=200.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "walker-run": PaperBaseline(
        task="walker-run",
        score=700.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "walker-stand": PaperBaseline(
        task="walker-stand",
        score=970.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
    "walker-walk": PaperBaseline(
        task="walker-walk",
        score=950.0,
        source="Hansen et al., 2024, TD-MPC2: Scalable, Robust World Models for Continuous Control",
    ),
}

SUITE_BASELINES: dict[str, dict[str, PaperBaseline]] = {
    "dreamerv3_atari100k": DREAMERV3_ATARI100K_BASELINES,
    "tdmpc2_dmcontrol39": TDMPC2_DMCONTROL39_BASELINES,
}
