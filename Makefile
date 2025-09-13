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
	python video_orientation_detector.py --version
	python video_orientation_detector.py test_video.mp4 --no-display --time-limit 1

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
	python setup.py sdist bdist_wheel

# Generate documentation
docs:
	@echo "Documentation is maintained in README.md"

# Initial setup
setup: install format
	pre-commit run --all-files

# Run all quality checks
check: format lint test