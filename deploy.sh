#!/bin/bash
# SVOD Deployment Script
# This script handles building, testing, and deploying SVOD

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PACKAGE_NAME="svod"
VERSION=$(grep "version" pyproject.toml | head -1 | cut -d'"' -f2)
DIST_DIR="dist"
BUILD_DIR="build"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    log_info "Checking dependencies..."

    # Check Python version
    if ! python --version | grep -q "Python 3.11\|Python 3.12"; then
        log_error "Python 3.11 or 3.12 required"
        exit 1
    fi

    # Check required tools
    for tool in pip black flake8 pytest; do
        if ! command -v $tool &> /dev/null; then
            log_error "$tool is required but not installed"
            exit 1
        fi
    done

    log_success "Dependencies check passed"
}

run_tests() {
    log_info "Running test suite..."

    if [ "$SKIP_TESTS" = "true" ]; then
        log_warning "Skipping tests as requested"
        return
    fi

    # Run tests with coverage
    if ! python -m pytest tests/ -v --cov=video_orientation_detector --cov-report=term-missing; then
        log_error "Tests failed"
        exit 1
    fi

    log_success "All tests passed"
}

run_linting() {
    log_info "Running code quality checks..."

    # Black formatting check
    if ! black --check --diff .; then
        log_error "Code formatting issues found. Run 'black .' to fix"
        exit 1
    fi

    # Flake8 linting
    if ! flake8 video_orientation_detector.py tests/; then
        log_error "Linting issues found"
        exit 1
    fi

    log_success "Code quality checks passed"
}

build_package() {
    log_info "Building package..."

    # Clean previous builds
    rm -rf $DIST_DIR $BUILD_DIR

    # Build package
    if ! python -m build; then
        log_error "Package build failed"
        exit 1
    fi

    # Verify package contents
    if ! tar -tf dist/*.tar.gz | head -20; then
        log_error "Package verification failed"
        exit 1
    fi

    log_success "Package built successfully"
}

install_package() {
    log_info "Installing package locally..."

    if ! pip install -e .; then
        log_error "Package installation failed"
        exit 1
    fi

    # Test installation
    if ! python -c "import video_orientation_detector; print('Import successful')"; then
        log_error "Package import test failed"
        exit 1
    fi

    log_success "Package installed successfully"
}

create_release_archive() {
    log_info "Creating release archive..."

    RELEASE_DIR="${PACKAGE_NAME}-${VERSION}"
    RELEASE_FILE="${PACKAGE_NAME}-${VERSION}.tar.gz"

    # Create release directory
    mkdir -p "$RELEASE_DIR"

    # Copy necessary files
    cp -r video_orientation_detector.py tests/ requirements.txt pyproject.toml README.md LICENSE "$RELEASE_DIR/"

    # Copy model files
    cp *.yaml *.caffemodel *.prototxt *.xml *.bin *.names *.cfg *.weights *.pt *.csv "$RELEASE_DIR/" 2>/dev/null || true

    # Create archive
    tar -czf "$RELEASE_FILE" "$RELEASE_DIR"

    # Cleanup
    rm -rf "$RELEASE_DIR"

    log_success "Release archive created: $RELEASE_FILE"
}

show_usage() {
    cat << EOF
SVOD Deployment Script

Usage: $0 [OPTIONS] [COMMAND]

Commands:
    all         Run full deployment pipeline (default)
    test        Run tests only
    lint        Run linting only
    build       Build package only
    install     Install package locally
    release     Create release archive

Options:
    --skip-tests    Skip running tests
    --help, -h      Show this help message

Examples:
    $0              # Full deployment
    $0 test         # Run tests only
    $0 --skip-tests build  # Build without tests
    $0 release      # Create release archive

EOF
}

# Main script
main() {
    local command="all"
    SKIP_TESTS="false"

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-tests)
                SKIP_TESTS="true"
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            test|lint|build|install|release|all)
                command="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    log_info "SVOD Deployment Script v$VERSION"
    log_info "Command: $command"

    case $command in
        all)
            check_dependencies
            run_linting
            run_tests
            build_package
            install_package
            create_release_archive
            log_success "Full deployment completed successfully!"
            ;;
        test)
            check_dependencies
            run_tests
            ;;
        lint)
            check_dependencies
            run_linting
            ;;
        build)
            check_dependencies
            run_linting
            run_tests
            build_package
            ;;
        install)
            check_dependencies
            run_linting
            run_tests
            build_package
            install_package
            ;;
        release)
            check_dependencies
            run_linting
            run_tests
            create_release_archive
            ;;
    esac
}

# Run main function with all arguments
main "$@"