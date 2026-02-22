.PHONY: help install dev test lint typecheck format docs docs-serve clean security quickstart bench pre-commit-install pre-commit-run

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package
	uv sync

dev: ## Install with all dev dependencies
	uv sync --extra dev --extra all

test: ## Run tests
	uv run pytest tests/ -v

test-quick: ## Run tests (fast subset)
	uv run pytest tests/ -v -x --timeout=60

test-cov: ## Run tests with coverage report
	uv run pytest tests/ --cov=src/worldflux --cov-report=term-missing

lint: ## Run linter
	uv run ruff check src/ tests/ examples/

lint-fix: ## Run linter with auto-fix
	uv run ruff check --fix src/ tests/ examples/

format: ## Format code
	uv run ruff format src/ tests/ examples/

format-check: ## Check formatting without changes
	uv run ruff format --check src/ tests/ examples/

typecheck: ## Run type checker
	uv run mypy src/worldflux/

security: ## Run security checks
	uv run bandit -r src/worldflux/ -ll
	uv run pip-audit

docs: ## Build documentation
	uv run mkdocs build --strict

docs-serve: ## Serve documentation locally
	uv run mkdocs serve

quickstart: ## Run CPU quickstart
	uv run python examples/quickstart_cpu_success.py --quick

clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	rm -rf htmlcov/ .coverage*
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

bench: ## Run benchmarks
	uv run pytest tests/ -v -k bench --timeout=120

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

pre-commit-run: ## Run pre-commit on all files
	uv run pre-commit run --all-files

ci: lint format-check typecheck test bench ## Run full CI gate locally
