# SVOD Development Makefile
# Common development tasks for Smart Video Orientation Detector

.PHONY: help install format lint test test-cov test-ci clean build docs setup check deploy release docker-build docker-run

# Default target
help:
	@echo "SVOD Development Commands:"
	@echo "  install     - Install development dependencies"
	@echo "  format      - Format code with Black"
	@echo "  lint        - Run linting with Flake8"
	@echo "  test        - Run pytest"
	@echo "  test-cov    - Run pytest with a coverage report"
	@echo "  test-ci     - Run pytest with the CI coverage gate"
	@echo "  clean       - Clean up temporary files and caches"
	@echo "  build       - Build distribution packages"
	@echo "  docs        - Generate/update documentation"
	@echo "  setup       - Initial project setup (install + format)"
	@echo "  check       - Run all quality checks (format + lint + test)"
	@echo "  deploy      - Full deployment pipeline"
	@echo "  release     - Create release archive"

# Install development dependencies
install:
	pip install -r requirements.txt
	pip install black flake8 pre-commit
	pre-commit install

# Format code
format:
	python -m black .

# Lint code (uses .flake8 config file for all settings)
lint:
	python -m flake8 .

# Run basic tests
test:
	python -m pytest tests/ -v --tb=short

# Run tests with coverage
test-cov:
	python -m pytest tests/ --cov=video_orientation_detector --cov=svod_detector --cov=svod_bootstrap --cov=svod_cli --cov=svod_models --cov=svod_statistics --cov=svod_state --cov-report=html --cov-report=term-missing

# Run tests with coverage and fail if below threshold
test-ci:
	python -m pytest tests/ --cov=video_orientation_detector --cov=svod_detector --cov=svod_bootstrap --cov=svod_cli --cov=svod_models --cov=svod_statistics --cov=svod_state --cov-report=term-missing --cov-fail-under=30

# Clean up
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".tox" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/

# Build distribution
build:
	python -m build

# Generate documentation
docs:
	@echo "Documentation is maintained in README.md"

# Initial setup
setup: install format
	pre-commit run --all-files

# Run all quality checks
check: format lint test

# Full deployment pipeline
deploy: clean check build
	@echo "Deployment package ready in dist/"

docker-build:
	docker build -t svod .

docker-run:
	docker run --rm -v $(PWD)/data:/data svod --help

# Create release archive
release: clean check
	@echo "Creating release archive..."
	@VERSION=$$(grep "version" pyproject.toml | head -1 | cut -d'"' -f2); \
	mkdir -p svod-$$VERSION; \
	cp video_orientation_detector.py svod_detector.py svod_bootstrap.py svod_state.py svod_cli.py svod_models.py svod_statistics.py __init__.py inspect_rotation.py requirements.txt pyproject.toml README.md LICENSE Makefile svod-$$VERSION/; \
	cp -r tests svod-$$VERSION/; \
	cp *.yaml *.caffemodel *.prototxt *.xml *.bin *.names *.cfg *.weights *.pt *.csv svod-$$VERSION/ 2>/dev/null || true; \
	tar -czf svod-$$VERSION.tar.gz svod-$$VERSION; \
	rm -rf svod-$$VERSION; \
	echo "Release archive created: svod-$$VERSION.tar.gz"