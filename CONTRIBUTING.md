# Contributing to WorldFlux

Thank you for your interest in contributing to WorldFlux! This document provides guidelines and information for contributors.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/worldflux/WorldFlux.git
   cd worldflux
   ```

2. **Install `uv` (if needed)**
   ```bash
   # macOS (Homebrew)
   brew install uv

   # Or: https://docs.astral.sh/uv/getting-started/installation/
   ```

3. **Install development dependencies**
   ```bash
   uv sync --extra dev
   ```

## Code Quality

We maintain high code quality standards. Before submitting a PR, ensure your code passes all checks:

```bash
# Run tests
uv run pytest tests/ -v

# Type checking
uv run mypy src/worldflux/

# Linting
uv run ruff check src/

# Fix lint issues automatically
uv run ruff check --fix src/
```

## Coding Standards

### Style Guide

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints for all public functions and methods
- Maximum line length: 100 characters
- Use double quotes for strings

### Documentation

- Write docstrings for all public classes, methods, and functions
- Use Google-style docstrings
- Include examples in docstrings where appropriate
- Keep comments concise and meaningful

### Example Docstring

```python
def encode(self, obs: Tensor) -> LatentState:
    """
    Encode observation to latent state.

    Args:
        obs: Observation tensor of shape [batch_size, *obs_shape].

    Returns:
        LatentState containing the encoded representation.

    Raises:
        ShapeMismatchError: If observation shape doesn't match config.

    Example:
        >>> state = model.encode(torch.randn(32, 3, 64, 64))
        >>> state.features.shape
        torch.Size([32, 1024])
    """
```

## Testing

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_training.py -v

# Run with coverage
uv run pytest tests/ --cov=worldflux --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Mirror the source structure (e.g., `tests/test_core/test_state.py`)
- Use descriptive test names: `test_encode_returns_correct_shape`
- Test edge cases and error conditions
- Use fixtures for common setup

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, focused commits
   - Include tests for new functionality
   - Update documentation as needed

3. **Ensure all checks pass**
   ```bash
   uv run python scripts/run_local_ci_gate.py
   ```
   - `pre-commit` now enforces the same local CI gate on every commit.
   - Install hooks once per clone:
     ```bash
     uv run --with pre-commit pre-commit install --hook-type pre-commit --hook-type pre-push
     ```

4. **Submit a pull request**
   - Provide a clear description of the changes
   - Reference any related issues
   - Request review from maintainers

## Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Adding or updating tests
- `refactor`: Code refactoring
- `style`: Code style changes (formatting, etc.)
- `chore`: Maintenance tasks

Examples:
```
feat(training): add early stopping callback
fix(dreamer): correct KL divergence computation
docs(readme): add training examples
```

## Adding New Models

To add a new world model implementation:

1. **Create configuration class** in `src/worldflux/core/config.py`
   ```python
   @dataclass
   class NewModelConfig(WorldModelConfig):
       model_type: str = "newmodel"
       # Add model-specific parameters
   ```

2. **Implement the model** in `src/worldflux/models/newmodel/`
   - Create `__init__.py` with exports
   - Create `world_model.py` implementing the `WorldModel` protocol
   - Register with `@WorldModelRegistry.register("newmodel", NewModelConfig)`
   - If component overrides should be effective, declare `self.composable_support` slots explicitly

3. **Add tests** in `tests/test_models/test_newmodel.py`
   - Test all protocol methods
   - Test configuration validation
   - Test save/load functionality

4. **Update factory** in `src/worldflux/factory.py`
   - Add size presets to `MODEL_CATALOG`
   - Add aliases to `MODEL_ALIASES`

5. **Update documentation**
   - Add to README.md

### External Plugin Note

Entry-point plugins are currently treated as experimental API. External plugins should
register a compatibility manifest (`plugin_api_version`, `worldflux_version_range`,
`capabilities`) via `WorldModelRegistry.register_plugin_manifest(...)`.

## Reporting Issues

When reporting issues, please include:

1. **Description**: Clear description of the problem
2. **Reproduction steps**: Minimal code to reproduce the issue
3. **Expected behavior**: What you expected to happen
4. **Actual behavior**: What actually happened
5. **Environment**: Python version, PyTorch version, OS

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas

Thank you for contributing!
