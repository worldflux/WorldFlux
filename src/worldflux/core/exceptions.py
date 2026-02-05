"""Custom exceptions for WorldFlux."""


class WorldFluxError(Exception):
    """Base exception for all WorldFlux errors."""

    pass


class ConfigurationError(WorldFluxError):
    """Raised when model configuration is invalid."""

    def __init__(self, message: str, config_name: str | None = None):
        self.config_name = config_name
        if config_name:
            message = f"[{config_name}] {message}"
        super().__init__(message)


class ShapeMismatchError(WorldFluxError):
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


class StateError(WorldFluxError):
    """Raised when State is in an invalid state."""

    pass


class ValidationError(WorldFluxError):
    """Raised when runtime validation fails."""

    pass


class ContractValidationError(ValidationError):
    """Raised when model I/O contract validation fails."""

    pass


class CapabilityError(WorldFluxError):
    """Raised when a required model capability is missing."""

    pass


class ModelNotFoundError(WorldFluxError):
    """Raised when a requested model is not found in the registry."""

    def __init__(self, model_name: str, available: list[str] | None = None):
        self.model_name = model_name
        self.available = available
        message = f"Model '{model_name}' not found"
        if available:
            message += f". Available models: {available}"
        super().__init__(message)


class CheckpointError(WorldFluxError):
    """Raised when checkpoint loading/saving fails."""

    pass


class TrainingError(WorldFluxError):
    """Raised when training encounters an error."""

    pass


class BufferError(WorldFluxError):
    """Raised when replay buffer operations fail."""

    pass
