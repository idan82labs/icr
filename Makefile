.PHONY: install dev test lint format clean build publish help

# Default target
help:
	@echo "ICR Development Commands"
	@echo "========================"
	@echo ""
	@echo "Setup:"
	@echo "  make install    Install package in editable mode"
	@echo "  make dev        Install with development dependencies"
	@echo ""
	@echo "Quality:"
	@echo "  make test       Run tests with pytest"
	@echo "  make lint       Run linting (ruff + mypy)"
	@echo "  make format     Format code with ruff"
	@echo ""
	@echo "Build:"
	@echo "  make build      Build distribution packages"
	@echo "  make clean      Remove build artifacts"
	@echo ""
	@echo "Other:"
	@echo "  make help       Show this help message"

# Installation targets
install:
	pip install -e .

dev:
	pip install -e ".[dev]"

full:
	pip install -e ".[full]"

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src/icr --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow"

# Linting and formatting
lint:
	ruff check src/ tests/
	mypy src/

lint-fix:
	ruff check --fix src/ tests/

format:
	ruff format src/ tests/

format-check:
	ruff format --check src/ tests/

# Build targets
build: clean
	python -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Publishing (requires twine)
publish: build
	twine upload dist/*

publish-test: build
	twine upload --repository testpypi dist/*

# Development helpers
init-icr:
	icr init

doctor:
	icr doctor

# Type stubs generation
stubs:
	stubgen -p icr -o stubs/

# Pre-commit hooks
pre-commit-install:
	pre-commit install

pre-commit-run:
	pre-commit run --all-files
