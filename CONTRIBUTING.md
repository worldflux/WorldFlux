# Contributing to World Models SDK

Thank you for your interest in contributing to World Models SDK! This document provides guidelines and information for contributors.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/worldmodels/worldmodels-sdk.git
   cd worldmodels-sdk
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

## Code Quality

We maintain high code quality standards. Before submitting a PR, ensure your code passes all checks:

```bash
# Run tests
pytest tests/ -v

# Type checking
mypy src/worldmodels/

# Linting
ruff check src/

# Fix lint issues automatically
ruff check --fix src/
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
pytest tests/ -v

# Run specific test file
pytest tests/test_training.py -v

# Run with coverage
pytest tests/ --cov=worldmodels --cov-report=html
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
   pytest tests/ -v
   mypy src/worldmodels/
   ruff check src/
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

1. **Create configuration class** in `src/worldmodels/core/config.py`
   ```python
   @dataclass
   class NewModelConfig(WorldModelConfig):
       model_type: str = "newmodel"
       # Add model-specific parameters
   ```

2. **Implement the model** in `src/worldmodels/models/newmodel/`
   - Create `__init__.py` with exports
   - Create `world_model.py` implementing the `WorldModel` protocol
   - Register with `@WorldModelRegistry.register("newmodel", NewModelConfig)`

3. **Add tests** in `tests/test_models/test_newmodel.py`
   - Test all protocol methods
   - Test configuration validation
   - Test save/load functionality

4. **Update factory** in `src/worldmodels/factory.py`
   - Add size presets to `MODEL_CATALOG`
   - Add aliases to `MODEL_ALIASES`

5. **Update documentation**
   - Add to README.md
   - Add to CLAUDE.md

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
