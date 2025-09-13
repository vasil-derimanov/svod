# SVOD Development Makefile
# Common development tasks for Smart Video Orientation Detector

.PHONY: help install format lint test clean build docs

# Default target
help:
	@echo "SVOD Development Commands:"
	@echo "  install     - Install development dependencies"
	@echo "  format      - Format code with Black"
	@echo "  lint        - Run linting with Flake8"
	@echo "  test        - Run basic functionality tests"
	@echo "  clean       - Clean up temporary files and caches"
	@echo "  build       - Build distribution packages"
	@echo "  docs        - Generate/update documentation"
	@echo "  setup       - Initial project setup (install + format)"
	@echo "  check       - Run all quality checks (format + lint + test)"
	@echo "  deploy      - Full deployment pipeline"
	@echo "  docker      - Build and run Docker container"
	@echo "  docker-build- Build Docker image"
	@echo "  docker-run  - Run Docker container"
	@echo "  release     - Create release archive"

# Install development dependencies
install:
	pip install -r requirements.txt
	pip install black flake8 pre-commit
	pre-commit install

# Format code
format:
	python -m black .

# Lint code
lint:
	python -m flake8 . --max-line-length=100 --extend-ignore=E203,W503

# Run basic tests
test:
	python -m pytest tests/ -v --tb=short

# Run tests with coverage
test-cov:
	python -m pytest tests/ --cov=video_orientation_detector --cov-report=html --cov-report=term-missing

# Run tests with coverage and fail if below threshold
test-ci:
	python -m pytest tests/ --cov=video_orientation_detector --cov-report=term-missing --cov-fail-under=80

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

# Docker commands
docker: docker-build docker-run

docker-build:
	docker build -t svod .

docker-run:
	docker run --rm -v $(PWD)/data:/data svod python video_orientation_detector.py --help

# Create release archive
release: clean check
	@echo "Creating release archive..."
	@VERSION=$$(grep "version" pyproject.toml | head -1 | cut -d'"' -f2); \
	mkdir -p svod-$$VERSION; \
	cp -r video_orientation_detector.py tests/ requirements.txt pyproject.toml README.md LICENSE Makefile svod-$$VERSION/; \
	cp *.yaml *.caffemodel *.prototxt *.xml *.bin *.names *.cfg *.weights *.pt *.csv svod-$$VERSION/ 2>/dev/null || true; \
	tar -czf svod-$$VERSION.tar.gz svod-$$VERSION; \
	rm -rf svod-$$VERSION; \
	echo "Release archive created: svod-$$VERSION.tar.gz"