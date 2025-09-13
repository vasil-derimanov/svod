# SVOD Deployment Script for Windows
# This script handles building, testing, and deploying SVOD on Windows

param(
    [string]$Command = "all",
    [switch]$SkipTests,
    [switch]$Help
)

# Configuration
$PACKAGE_NAME = "svod"
$VERSION = (Get-Content pyproject.toml | Select-String "version" | Select-Object -First 1).ToString().Split('"')[1]
$DIST_DIR = "dist"
$BUILD_DIR = "build"

# Colors for output (PowerShell)
$RED = "Red"
$GREEN = "Green"
$YELLOW = "Yellow"
$BLUE = "Cyan"
$NC = "White"

function Write-ColorOutput {
    param([string]$Color, [string]$Message)
    Write-Host "[$Color] $Message" -ForegroundColor $Color
}

function Write-Info {
    param([string]$Message)
    Write-ColorOutput $BLUE "INFO" $Message
}

function Write-Success {
    param([string]$Message)
    Write-ColorOutput $GREEN "SUCCESS" $Message
}

function Write-Warning {
    param([string]$Message)
    Write-ColorOutput $YELLOW "WARNING" $Message
}

function Write-Error {
    param([string]$Message)
    Write-ColorOutput $RED "ERROR" $Message
}

function Check-Dependencies {
    Write-Info "Checking dependencies..."

    # Check Python version
    $pythonVersion = python --version 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Python not found"
        exit 1
    }

    if ($pythonVersion -notmatch "Python 3\.11|Python 3\.12") {
        Write-Error "Python 3.11 or 3.12 required"
        exit 1
    }

    # Check required tools
    $requiredTools = @("pip", "black", "flake8", "pytest")
    foreach ($tool in $requiredTools) {
        try {
            $null = Get-Command $tool -ErrorAction Stop
        }
        catch {
            Write-Error "$tool is required but not installed"
            exit 1
        }
    }

    Write-Success "Dependencies check passed"
}

function Run-Tests {
    Write-Info "Running test suite..."

    if ($SkipTests) {
        Write-Warning "Skipping tests as requested"
        return
    }

    # Run tests with coverage
    $testResult = python -m pytest tests/ -v --cov=video_orientation_detector --cov-report=term-missing
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Tests failed"
        exit 1
    }

    Write-Success "All tests passed"
}

function Run-Linting {
    Write-Info "Running code quality checks..."

    # Black formatting check
    $blackResult = black --check --diff .
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Code formatting issues found. Run 'black .' to fix"
        exit 1
    }

    # Flake8 linting
    $flakeResult = flake8 video_orientation_detector.py tests/
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Linting issues found"
        exit 1
    }

    Write-Success "Code quality checks passed"
}

function Build-Package {
    Write-Info "Building package..."

    # Clean previous builds
    if (Test-Path $DIST_DIR) {
        Remove-Item -Recurse -Force $DIST_DIR
    }
    if (Test-Path $BUILD_DIR) {
        Remove-Item -Recurse -Force $BUILD_DIR
    }

    # Build package
    $buildResult = python -m build
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Package build failed"
        exit 1
    }

    # Verify package contents (basic check)
    $packageFiles = Get-ChildItem "$DIST_DIR\*.tar.gz"
    if ($packageFiles.Count -eq 0) {
        Write-Error "No package files found in dist directory"
        exit 1
    }

    Write-Success "Package built successfully"
}

function Install-Package {
    Write-Info "Installing package locally..."

    $installResult = pip install -e .
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Package installation failed"
        exit 1
    }

    # Test installation
    $importResult = python -c "import video_orientation_detector; print('Import successful')"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Package import test failed"
        exit 1
    }

    Write-Success "Package installed successfully"
}

function Create-ReleaseArchive {
    Write-Info "Creating release archive..."

    $RELEASE_DIR = "$PACKAGE_NAME-$VERSION"
    $RELEASE_FILE = "$PACKAGE_NAME-$VERSION.zip"

    # Create release directory
    if (Test-Path $RELEASE_DIR) {
        Remove-Item -Recurse -Force $RELEASE_DIR
    }
    New-Item -ItemType Directory -Path $RELEASE_DIR | Out-Null

    # Copy necessary files
    Copy-Item "video_orientation_detector.py" $RELEASE_DIR
    Copy-Item "tests" $RELEASE_DIR -Recurse
    Copy-Item "requirements.txt" $RELEASE_DIR
    Copy-Item "pyproject.toml" $RELEASE_DIR
    Copy-Item "README.md" $RELEASE_DIR
    Copy-Item "LICENSE" $RELEASE_DIR

    # Copy model files
    $modelExtensions = @("*.yaml", "*.caffemodel", "*.prototxt", "*.xml", "*.bin", "*.names", "*.cfg", "*.weights", "*.pt", "*.csv")
    foreach ($ext in $modelExtensions) {
        $files = Get-ChildItem $ext -ErrorAction SilentlyContinue
        foreach ($file in $files) {
            Copy-Item $file.FullName $RELEASE_DIR
        }
    }

    # Create archive
    Compress-Archive -Path $RELEASE_DIR -DestinationPath $RELEASE_FILE -Force

    # Cleanup
    Remove-Item -Recurse -Force $RELEASE_DIR

    Write-Success "Release archive created: $RELEASE_FILE"
}

function Show-Usage {
    Write-Host @"
SVOD Deployment Script for Windows

Usage: .\deploy.ps1 [OPTIONS] [COMMAND]

Commands:
    all         Run full deployment pipeline (default)
    test        Run tests only
    lint        Run linting only
    build       Build package only
    install     Install package locally
    release     Create release archive

Options:
    -SkipTests      Skip running tests
    -Help           Show this help message

Examples:
    .\deploy.ps1                    # Full deployment
    .\deploy.ps1 test               # Run tests only
    .\deploy.ps1 -SkipTests build   # Build without tests
    .\deploy.ps1 release            # Create release archive

"@
}

# Main script
function Main {
    if ($Help) {
        Show-Usage
        exit 0
    }

    Write-Info "SVOD Deployment Script v$VERSION"
    Write-Info "Command: $Command"

    switch ($Command) {
        "all" {
            Check-Dependencies
            Run-Linting
            Run-Tests
            Build-Package
            Install-Package
            Create-ReleaseArchive
            Write-Success "Full deployment completed successfully!"
        }
        "test" {
            Check-Dependencies
            Run-Tests
        }
        "lint" {
            Check-Dependencies
            Run-Linting
        }
        "build" {
            Check-Dependencies
            Run-Linting
            Run-Tests
            Build-Package
        }
        "install" {
            Check-Dependencies
            Run-Linting
            Run-Tests
            Build-Package
            Install-Package
        }
        "release" {
            Check-Dependencies
            Run-Linting
            Run-Tests
            Create-ReleaseArchive
        }
        default {
            Write-Error "Unknown command: $Command"
            Show-Usage
            exit 1
        }
    }
}

# Run main function
Main