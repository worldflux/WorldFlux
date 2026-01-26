"""Custom exceptions for World Models SDK."""


class WorldModelsError(Exception):
    """Base exception for all World Models errors."""

    pass


class ConfigurationError(WorldModelsError):
    """Raised when model configuration is invalid."""

    def __init__(self, message: str, config_name: str | None = None):
        self.config_name = config_name
        if config_name:
            message = f"[{config_name}] {message}"
        super().__init__(message)


class ShapeMismatchError(WorldModelsError):
    """Raised when tensor shapes don't match expected dimensions."""

    def __init__(
        self,
        message: str,
        expected: tuple[int, ...] | None = None,
        got: tuple[int, ...] | None = None,
    ):
        self.expected = expected
        self.got = got
        if expected is not None and got is not None:
            message = f"{message} (expected {expected}, got {got})"
        super().__init__(message)


class StateError(WorldModelsError):
    """Raised when LatentState is in an invalid state."""

    pass


class ModelNotFoundError(WorldModelsError):
    """Raised when a requested model is not found in the registry."""

    def __init__(self, model_name: str, available: list[str] | None = None):
        self.model_name = model_name
        self.available = available
        message = f"Model '{model_name}' not found"
        if available:
            message += f". Available models: {available}"
        super().__init__(message)


class CheckpointError(WorldModelsError):
    """Raised when checkpoint loading/saving fails."""

    pass


class TrainingError(WorldModelsError):
    """Raised when training encounters an error."""

    pass


class BufferError(WorldModelsError):
    """Raised when replay buffer operations fail."""

    pass
