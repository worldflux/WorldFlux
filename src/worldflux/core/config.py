"""Configuration classes for world models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .exceptions import ConfigurationError


class LatentType(Enum):
    """Type of latent space representation."""

    DETERMINISTIC = "deterministic"
    GAUSSIAN = "gaussian"
    CATEGORICAL = "categorical"
    VQ = "vq"
    SIMNORM = "simnorm"


class DynamicsType(Enum):
    """Type of dynamics model architecture."""

    RSSM = "rssm"
    MLP = "mlp"
    TRANSFORMER = "transformer"
    SSM = "ssm"


@dataclass
class WorldModelConfig:
    """
    Base configuration for all world models.

    This class defines the common configuration parameters shared across
    all world model implementations. Subclasses (DreamerV3Config, TDMPC2Config)
    extend this with model-specific parameters.

    Attributes:
        model_type: Identifier for the model type ("dreamer", "tdmpc2", etc.).
        model_name: Human-readable name or size preset name.
        obs_shape: Shape of observations (e.g., (3, 64, 64) for images).
        action_dim: Dimension of the action space.
        action_type: Type of actions ("continuous" or "discrete").
        latent_type: Type of latent space representation.
        latent_dim: Dimension of the primary latent space.
        deter_dim: Dimension of deterministic state (RSSM models).
        stoch_dim: Dimension of stochastic state (RSSM models).
        dynamics_type: Type of dynamics model architecture.
        hidden_dim: Hidden dimension for MLPs and other layers.
        learning_rate: Default learning rate for training.
        grad_clip: Gradient clipping threshold.
        device: Target device ("cuda", "cpu", "auto").
        dtype: Data type ("float32", "float16", "bfloat16").

    Example:
        >>> config = WorldModelConfig(obs_shape=(84, 84, 3), action_dim=4)
        >>> config.save("config.json")
        >>> loaded = WorldModelConfig.load("config.json")
    """

    model_type: str = "base"
    model_name: str = "unnamed"

    # Observation/action space
    obs_shape: tuple[int, ...] = (3, 64, 64)
    action_dim: int = 6
    action_type: str = "continuous"

    # Latent space
    latent_type: LatentType = LatentType.DETERMINISTIC
    latent_dim: int = 256
    deter_dim: int = 256
    stoch_dim: int = 32

    # Dynamics
    dynamics_type: DynamicsType = DynamicsType.MLP
    hidden_dim: int = 512

    # Training
    learning_rate: float = 3e-4
    grad_clip: float = 100.0

    # Device
    device: str = "cuda"
    dtype: str = "float32"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        if not self.obs_shape or len(self.obs_shape) == 0:
            raise ConfigurationError(
                "obs_shape must be a non-empty tuple", config_name=self.model_name
            )

        if any(d <= 0 for d in self.obs_shape):
            raise ConfigurationError(
                f"obs_shape dimensions must be positive, got {self.obs_shape}",
                config_name=self.model_name,
            )

        if self.action_dim <= 0:
            raise ConfigurationError(
                f"action_dim must be positive, got {self.action_dim}",
                config_name=self.model_name,
            )

        if self.hidden_dim <= 0:
            raise ConfigurationError(
                f"hidden_dim must be positive, got {self.hidden_dim}",
                config_name=self.model_name,
            )

        if self.learning_rate <= 0:
            raise ConfigurationError(
                f"learning_rate must be positive, got {self.learning_rate}",
                config_name=self.model_name,
            )

        if self.grad_clip < 0:
            raise ConfigurationError(
                f"grad_clip must be non-negative, got {self.grad_clip}",
                config_name=self.model_name,
            )

        if self.action_type not in ("continuous", "discrete"):
            raise ConfigurationError(
                f"action_type must be 'continuous' or 'discrete', got {self.action_type!r}",
                config_name=self.model_name,
            )

        if self.dtype not in ("float32", "float16", "bfloat16"):
            raise ConfigurationError(
                f"dtype must be 'float32', 'float16', or 'bfloat16', got {self.dtype!r}",
                config_name=self.model_name,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        d = asdict(self)
        d["latent_type"] = self.latent_type.value
        d["dynamics_type"] = self.dynamics_type.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorldModelConfig:
        """
        Create configuration from dictionary.

        Args:
            d: Dictionary with configuration parameters.

        Returns:
            Configuration instance.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        d = d.copy()
        if "latent_type" in d and isinstance(d["latent_type"], str):
            d["latent_type"] = LatentType(d["latent_type"])
        if "dynamics_type" in d and isinstance(d["dynamics_type"], str):
            d["dynamics_type"] = DynamicsType(d["dynamics_type"])
        if "obs_shape" in d and isinstance(d["obs_shape"], list):
            d["obs_shape"] = tuple(d["obs_shape"])
        return cls(**d)

    def save(self, path: str | Path) -> None:
        """
        Save configuration to JSON file.

        Args:
            path: Path to save the configuration.
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> WorldModelConfig:
        """
        Load configuration from JSON file.

        Args:
            path: Path to the configuration file.

        Returns:
            Configuration instance (DreamerV3Config, TDMPC2Config, or base).

        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file is not valid JSON.
            ConfigurationError: If the configuration is invalid.
        """
        from .registry import ConfigRegistry

        return ConfigRegistry.from_pretrained(str(path))

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name!r}, "
            f"obs_shape={self.obs_shape}, "
            f"action_dim={self.action_dim}, "
            f"hidden_dim={self.hidden_dim})"
        )


@dataclass
class DreamerV3Config(WorldModelConfig):
    """
    DreamerV3 world model configuration.

    DreamerV3 uses an RSSM (Recurrent State-Space Model) with categorical
    latent variables. This configuration supports multiple size presets
    matching the original paper.

    Size Presets:
        - size12m:  12M params - deter=2048, stoch=16x16, hidden=256
        - size25m:  25M params - deter=4096, stoch=32x16, hidden=512
        - size50m:  50M params - deter=4096, stoch=32x32, hidden=640
        - size100m: 100M params - deter=8192, stoch=32x32, hidden=768
        - size200m: 200M params - deter=8192, stoch=32x32, hidden=1024

    Attributes:
        stoch_discrete: Number of categorical distributions.
        stoch_classes: Number of classes per categorical distribution.
        encoder_type: Type of encoder ("cnn" or "mlp").
        decoder_type: Type of decoder ("cnn" or "mlp").
        cnn_depth: Base depth multiplier for CNN encoder/decoder.
        cnn_kernels: Kernel sizes for CNN layers.
        kl_free: Free nats for KL divergence (prevents posterior collapse).
        kl_balance: Balance between prior and posterior in KL loss.
        loss_scales: Weights for each loss component.
        use_symlog: Whether to use symlog transformation for predictions.

    Example:
        >>> # Create from size preset
        >>> config = DreamerV3Config.from_size("size12m")

        >>> # Create with custom parameters
        >>> config = DreamerV3Config(
        ...     obs_shape=(3, 64, 64),
        ...     action_dim=4,
        ...     deter_dim=1024,
        ...     stoch_discrete=16,
        ...     stoch_classes=16,
        ... )
    """

    model_type: str = "dreamer"

    # RSSM
    latent_type: LatentType = LatentType.CATEGORICAL
    dynamics_type: DynamicsType = DynamicsType.RSSM

    stoch_discrete: int = 32
    stoch_classes: int = 32

    # Encoder/Decoder
    encoder_type: str = "cnn"
    decoder_type: str = "cnn"
    cnn_depth: int = 48
    cnn_kernels: tuple[int, ...] = (4, 4, 4, 4)

    # Loss weights
    kl_free: float = 1.0
    kl_balance: float = 0.8
    loss_scales: dict[str, float] = field(
        default_factory=lambda: {
            "reconstruction": 1.0,
            "kl": 0.1,
            "reward": 1.0,
            "continue": 1.0,
        }
    )

    use_symlog: bool = True

    def __post_init__(self) -> None:
        """Initialize derived values and validate."""
        self.stoch_dim = self.stoch_discrete * self.stoch_classes
        self._validate()

    def _validate(self) -> None:
        """Validate DreamerV3-specific configuration."""
        super()._validate()

        if self.stoch_discrete <= 0:
            raise ConfigurationError(
                f"stoch_discrete must be positive, got {self.stoch_discrete}",
                config_name=self.model_name,
            )

        if self.stoch_classes <= 0:
            raise ConfigurationError(
                f"stoch_classes must be positive, got {self.stoch_classes}",
                config_name=self.model_name,
            )

        if self.deter_dim <= 0:
            raise ConfigurationError(
                f"deter_dim must be positive, got {self.deter_dim}",
                config_name=self.model_name,
            )

        if self.encoder_type not in ("cnn", "mlp"):
            raise ConfigurationError(
                f"encoder_type must be 'cnn' or 'mlp', got {self.encoder_type!r}",
                config_name=self.model_name,
            )

        if self.decoder_type not in ("cnn", "mlp"):
            raise ConfigurationError(
                f"decoder_type must be 'cnn' or 'mlp', got {self.decoder_type!r}",
                config_name=self.model_name,
            )

        if self.encoder_type == "cnn" and len(self.obs_shape) != 3:
            raise ConfigurationError(
                f"CNN encoder requires 3D obs_shape (C, H, W), got {self.obs_shape}",
                config_name=self.model_name,
            )

        if self.kl_free < 0:
            raise ConfigurationError(
                f"kl_free must be non-negative, got {self.kl_free}",
                config_name=self.model_name,
            )

        if not (0 <= self.kl_balance <= 1):
            raise ConfigurationError(
                f"kl_balance must be in [0, 1], got {self.kl_balance}",
                config_name=self.model_name,
            )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DreamerV3Config:
        """Create configuration from dictionary."""
        d = d.copy()
        if "latent_type" in d and isinstance(d["latent_type"], str):
            d["latent_type"] = LatentType(d["latent_type"])
        if "dynamics_type" in d and isinstance(d["dynamics_type"], str):
            d["dynamics_type"] = DynamicsType(d["dynamics_type"])
        if "obs_shape" in d and isinstance(d["obs_shape"], list):
            d["obs_shape"] = tuple(d["obs_shape"])
        if "cnn_kernels" in d and isinstance(d["cnn_kernels"], list):
            d["cnn_kernels"] = tuple(d["cnn_kernels"])
        return cls(**d)

    @classmethod
    def from_size(cls, size: str, **kwargs: Any) -> DreamerV3Config:
        """
        Create configuration from a size preset.

        Args:
            size: Size preset name (size12m, size25m, size50m, size100m, size200m).
            **kwargs: Override any preset parameters.

        Returns:
            DreamerV3Config with the specified size preset.

        Raises:
            ValueError: If the size preset is not recognized.
        """
        presets: dict[str, dict[str, Any]] = {
            "ci": {
                "deter_dim": 64,
                "stoch_discrete": 4,
                "stoch_classes": 4,
                "hidden_dim": 32,
                "cnn_depth": 8,
            },
            "size12m": {
                "deter_dim": 2048,
                "stoch_discrete": 16,
                "stoch_classes": 16,
                "hidden_dim": 256,
            },
            "size25m": {
                "deter_dim": 4096,
                "stoch_discrete": 32,
                "stoch_classes": 16,
                "hidden_dim": 512,
            },
            "size50m": {
                "deter_dim": 4096,
                "stoch_discrete": 32,
                "stoch_classes": 32,
                "hidden_dim": 640,
            },
            "size100m": {
                "deter_dim": 8192,
                "stoch_discrete": 32,
                "stoch_classes": 32,
                "hidden_dim": 768,
            },
            "size200m": {
                "deter_dim": 8192,
                "stoch_discrete": 32,
                "stoch_classes": 32,
                "hidden_dim": 1024,
            },
        }
        if size not in presets:
            raise ValueError(f"Unknown size: {size}. Available: {list(presets.keys())}")

        preset = presets[size]
        preset.update(kwargs)
        return cls(model_name=size, **preset)


@dataclass
class TDMPC2Config(WorldModelConfig):
    """
    TD-MPC2 world model configuration.

    TD-MPC2 is an implicit world model that uses SimNorm latent space
    and learns value functions for planning. It's particularly effective
    for continuous control tasks.

    Size Presets:
        - 5m:   5M params - latent=256, hidden=256
        - 19m:  19M params - latent=512, hidden=512
        - 48m:  48M params - latent=512, hidden=1024
        - 317m: 317M params - latent=1024, hidden=2048

    Attributes:
        simnorm_dim: Dimension for SimNorm grouping (latent_dim must be divisible).
        num_hidden_layers: Number of hidden layers in MLPs.
        task_dim: Dimension of task embedding for multi-task learning.
        num_tasks: Number of tasks for multi-task learning.
        num_q_networks: Number of Q-networks in the ensemble.
        horizon: Planning horizon for MPC.
        num_samples: Number of action samples for planning.
        num_elites: Number of elite samples for CEM planning.
        temperature: Temperature for action sampling.
        momentum: Momentum for CEM mean update.
        use_decoder: Whether to use a decoder (TD-MPC2 is typically implicit).

    Example:
        >>> # Create from size preset
        >>> config = TDMPC2Config.from_size("5m", obs_shape=(39,), action_dim=6)

        >>> # Create with custom parameters
        >>> config = TDMPC2Config(
        ...     obs_shape=(39,),
        ...     action_dim=6,
        ...     latent_dim=256,
        ...     hidden_dim=256,
        ... )
    """

    model_type: str = "tdmpc2"

    latent_type: LatentType = LatentType.SIMNORM
    dynamics_type: DynamicsType = DynamicsType.MLP

    # SimNorm
    simnorm_dim: int = 8

    # MLP dynamics
    num_hidden_layers: int = 2

    # Multi-task
    task_dim: int = 96
    num_tasks: int = 1

    # Value ensemble
    num_q_networks: int = 5

    # Planning
    horizon: int = 5
    num_samples: int = 512
    num_elites: int = 64
    temperature: float = 0.5
    momentum: float = 0.1

    # TD learning
    gamma: float = 0.99

    # TD-MPC2 is implicit by default
    use_decoder: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        self._validate()

    def _validate(self) -> None:
        """Validate TD-MPC2-specific configuration."""
        super()._validate()

        if self.latent_dim <= 0:
            raise ConfigurationError(
                f"latent_dim must be positive, got {self.latent_dim}",
                config_name=self.model_name,
            )

        if self.simnorm_dim <= 0:
            raise ConfigurationError(
                f"simnorm_dim must be positive, got {self.simnorm_dim}",
                config_name=self.model_name,
            )

        if self.latent_dim % self.simnorm_dim != 0:
            raise ConfigurationError(
                f"latent_dim ({self.latent_dim}) must be divisible by "
                f"simnorm_dim ({self.simnorm_dim})",
                config_name=self.model_name,
            )

        if self.num_hidden_layers < 1:
            raise ConfigurationError(
                f"num_hidden_layers must be at least 1, got {self.num_hidden_layers}",
                config_name=self.model_name,
            )

        if self.num_q_networks < 1:
            raise ConfigurationError(
                f"num_q_networks must be at least 1, got {self.num_q_networks}",
                config_name=self.model_name,
            )

        if self.horizon < 1:
            raise ConfigurationError(
                f"horizon must be at least 1, got {self.horizon}",
                config_name=self.model_name,
            )

        if self.num_elites > self.num_samples:
            raise ConfigurationError(
                f"num_elites ({self.num_elites}) cannot exceed num_samples ({self.num_samples})",
                config_name=self.model_name,
            )

        if not (0 < self.temperature):
            raise ConfigurationError(
                f"temperature must be positive, got {self.temperature}",
                config_name=self.model_name,
            )

        if not (0 <= self.momentum <= 1):
            raise ConfigurationError(
                f"momentum must be in [0, 1], got {self.momentum}",
                config_name=self.model_name,
            )

        if not (0 < self.gamma <= 1):
            raise ConfigurationError(
                f"gamma must be in (0, 1], got {self.gamma}",
                config_name=self.model_name,
            )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TDMPC2Config:
        """Create configuration from dictionary."""
        d = d.copy()
        if "latent_type" in d and isinstance(d["latent_type"], str):
            d["latent_type"] = LatentType(d["latent_type"])
        if "dynamics_type" in d and isinstance(d["dynamics_type"], str):
            d["dynamics_type"] = DynamicsType(d["dynamics_type"])
        if "obs_shape" in d and isinstance(d["obs_shape"], list):
            d["obs_shape"] = tuple(d["obs_shape"])
        return cls(**d)

    @classmethod
    def from_size(cls, size: str, **kwargs: Any) -> TDMPC2Config:
        """
        Create configuration from a size preset.

        Args:
            size: Size preset name (5m, 19m, 48m, 317m).
            **kwargs: Override any preset parameters.

        Returns:
            TDMPC2Config with the specified size preset.

        Raises:
            ValueError: If the size preset is not recognized.
        """
        presets: dict[str, dict[str, Any]] = {
            "ci": {"latent_dim": 32, "hidden_dim": 32, "num_q_networks": 2},
            "5m": {"latent_dim": 256, "hidden_dim": 256},
            "19m": {"latent_dim": 512, "hidden_dim": 512},
            "48m": {"latent_dim": 512, "hidden_dim": 1024},
            "317m": {"latent_dim": 1024, "hidden_dim": 2048},
        }
        if size not in presets:
            raise ValueError(f"Unknown size: {size}. Available: {list(presets.keys())}")

        preset = presets[size]
        preset.update(kwargs)
        return cls(model_name=size, **preset)


@dataclass
class JEPABaseConfig(WorldModelConfig):
    """Base configuration for JEPA-style models."""

    model_type: str = "jepa"
    latent_type: LatentType = LatentType.DETERMINISTIC
    dynamics_type: DynamicsType = DynamicsType.TRANSFORMER

    encoder_dim: int = 512
    predictor_dim: int = 512
    projection_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.0

    def _validate(self) -> None:
        super()._validate()
        if self.encoder_dim <= 0:
            raise ConfigurationError(
                f"encoder_dim must be positive, got {self.encoder_dim}",
                config_name=self.model_name,
            )
        if self.predictor_dim <= 0:
            raise ConfigurationError(
                f"predictor_dim must be positive, got {self.predictor_dim}",
                config_name=self.model_name,
            )
        if self.projection_dim <= 0:
            raise ConfigurationError(
                f"projection_dim must be positive, got {self.projection_dim}",
                config_name=self.model_name,
            )
        if self.num_layers <= 0:
            raise ConfigurationError(
                f"num_layers must be positive, got {self.num_layers}",
                config_name=self.model_name,
            )
        if self.num_heads <= 0:
            raise ConfigurationError(
                f"num_heads must be positive, got {self.num_heads}",
                config_name=self.model_name,
            )


@dataclass
class TokenWorldModelConfig(WorldModelConfig):
    """Configuration for token-based world models."""

    model_type: str = "token"
    latent_type: LatentType = LatentType.VQ
    dynamics_type: DynamicsType = DynamicsType.TRANSFORMER

    vocab_size: int = 1024
    token_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1

    def _validate(self) -> None:
        super()._validate()
        if self.vocab_size <= 0:
            raise ConfigurationError(
                f"vocab_size must be positive, got {self.vocab_size}",
                config_name=self.model_name,
            )
        if self.token_dim <= 0:
            raise ConfigurationError(
                f"token_dim must be positive, got {self.token_dim}",
                config_name=self.model_name,
            )
        if self.num_layers <= 0:
            raise ConfigurationError(
                f"num_layers must be positive, got {self.num_layers}",
                config_name=self.model_name,
            )
        if self.num_heads <= 0:
            raise ConfigurationError(
                f"num_heads must be positive, got {self.num_heads}",
                config_name=self.model_name,
            )

    @classmethod
    def from_size(cls, size: str, **kwargs: Any) -> TokenWorldModelConfig:
        presets: dict[str, dict[str, Any]] = {
            "ci": {"token_dim": 32, "num_layers": 1, "num_heads": 1, "vocab_size": 64},
            "tiny": {"token_dim": 64, "num_layers": 2, "num_heads": 2, "vocab_size": 128},
            "base": {"token_dim": 256, "num_layers": 4, "num_heads": 4, "vocab_size": 1024},
        }
        if size not in presets:
            raise ValueError(f"Unknown size: {size}. Available: {list(presets.keys())}")
        preset = presets[size]
        preset.update(kwargs)
        return cls(model_name=size, **preset)


@dataclass
class DiffusionWorldModelConfig(WorldModelConfig):
    """Configuration for diffusion-style world models."""

    model_type: str = "diffusion"
    latent_type: LatentType = LatentType.GAUSSIAN
    dynamics_type: DynamicsType = DynamicsType.MLP

    diffusion_steps: int = 4
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def _validate(self) -> None:
        super()._validate()
        if self.diffusion_steps <= 0:
            raise ConfigurationError(
                f"diffusion_steps must be positive, got {self.diffusion_steps}",
                config_name=self.model_name,
            )
        if self.beta_start <= 0 or self.beta_end <= 0:
            raise ConfigurationError(
                f"beta values must be positive, got {self.beta_start}, {self.beta_end}",
                config_name=self.model_name,
            )
        if self.beta_end <= self.beta_start:
            raise ConfigurationError(
                "beta_end must be greater than beta_start",
                config_name=self.model_name,
            )

    @classmethod
    def from_size(cls, size: str, **kwargs: Any) -> DiffusionWorldModelConfig:
        presets: dict[str, dict[str, Any]] = {
            "ci": {"hidden_dim": 32, "diffusion_steps": 1},
            "tiny": {"hidden_dim": 64, "diffusion_steps": 2},
            "base": {"hidden_dim": 256, "diffusion_steps": 4},
        }
        if size not in presets:
            raise ValueError(f"Unknown size: {size}. Available: {list(presets.keys())}")
        preset = presets[size]
        preset.update(kwargs)
        return cls(model_name=size, **preset)
