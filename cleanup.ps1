# SVOD Project Cleanup Script
# Version: 1.0.0
# Last Updated: 2025-09-15
# Follows rules from copilot-instructions.md for safe project cleanup

Write-Host "SVOD Cleanup Script v1.0.0 - Safe project cleanup following copilot-instructions.md" -ForegroundColor Cyan
Write-Host "WARNING: This script will only remove truly unnecessary files and folders" -ForegroundColor Yellow
Write-Host "OK: All critical files and folders will be preserved" -ForegroundColor Green

# Critical files and folders that MUST NOT be deleted (from copilot-instructions.md)
$protectedFiles = @(
    "video_orientation_detector.py",
    "video_orientation_detector_old.py",
    "test_batch.py",
    "test_single.py",
    "test_comparison.py",
    "test_improved_detection.py",
    "test_logic_improvements.py",
    "test_p2170127_advanced.py",
    "test_p2170127_improvements.py",
    "test_p2170127_quick.py",
    "test_practical_improvements.py",
    "test_real_p2170127.py",
    "test_real_videos.py",
    "test_simple.py",
    "debug_p2170127.py",
    "performance_comparison.py",
    "TEST_README.md",
    "reference_orientations.csv",
    "pyproject.toml",
    "requirements.txt",
    "Makefile",
    "cleanup.ps1",
    "cleanup.py",
    ".pre-commit-config.yaml"
)

$protectedFolders = @(
    "tests",
    ".vscode",
    "performance_baselines"
)

# Function to check if path is protected
function Test-ProtectedPath {
    param([string]$path)

    # Check exact file matches
    foreach ($file in $protectedFiles) {
        if ($path -eq $file) {
            return $true
        }
    }

    # Check if path is inside protected folders
    foreach ($folder in $protectedFolders) {
        if ($path.StartsWith("$folder\") -or $path.StartsWith("$folder/") -or $path -eq $folder) {
            return $true
        }
    }

    return $false
}

# Remove unnecessary model files (keeping only essential ones)
$unnecessaryModelFiles = @(
    "coco.names",
    "mobilenet-v2.bin",
    "mobilenet-v2.xml",
    "yolov4.cfg",
    "yolov4.weights"
)

Write-Host "`nRemoving unnecessary model files..." -ForegroundColor Yellow
foreach ($file in $unnecessaryModelFiles) {
    if (Test-Path $file) {
        if (Test-ProtectedPath $file) {
            Write-Host "Protected: $file" -ForegroundColor Blue
        } else {
            Remove-Item -Force $file
            Write-Host "Removed: $file" -ForegroundColor Green
        }
    } else {
        Write-Host "Not found: $file" -ForegroundColor Gray
    }
}

# Remove old test virtual environments (keeping current test_env)
$oldTestEnvs = @(
    ".venv-clean",
    ".venv-test",
    ".venv-wsl-clean",
    ".venv-test-linux",
    ".venv-test-linux-clean",
    ".venv-test-v492",
    ".venv-wsl-test-v492",
    ".venv-test-py313",
    ".venv-test-py311",
    ".venv-test-v410",
    ".venv-test-v410-windows",
    ".venv-test-v410-wsl",
    ".venv-final-test",
    ".venv-comprehensive-test",
    ".venv-accuracy-test",
    ".venv-rotation-test"
)

Write-Host "`nRemoving old test virtual environments..." -ForegroundColor Yellow
foreach ($env in $oldTestEnvs) {
    if (Test-Path $env) {
        if (Test-ProtectedPath $env) {
            Write-Host "Protected: $env" -ForegroundColor Blue
        } else {
            Remove-Item -Recurse -Force $env
            Write-Host "Removed: $env" -ForegroundColor Green
        }
    } else {
        Write-Host "Not found: $env" -ForegroundColor Gray
    }
}

# Remove temporary and cache files (but preserve protected folders)
Write-Host "`nRemoving temporary and cache files..." -ForegroundColor Yellow

# Remove __pycache__ directories (but not inside protected folders)
$pyCacheDirs = Get-ChildItem -Path "." -Directory -Name "__pycache__" -Recurse -ErrorAction SilentlyContinue
foreach ($dir in $pyCacheDirs) {
    $fullPath = Resolve-Path $dir
    if (-not (Test-ProtectedPath $fullPath.Path)) {
        Remove-Item -Recurse -Force $fullPath.Path
        Write-Host "Removed: $fullPath" -ForegroundColor Green
    } else {
        Write-Host "Protected: $fullPath" -ForegroundColor Blue
    }
}

# Remove .pytest_cache directories
$pytestCacheDirs = Get-ChildItem -Path "." -Directory -Name ".pytest_cache" -Recurse -ErrorAction SilentlyContinue
foreach ($dir in $pytestCacheDirs) {
    $fullPath = Resolve-Path $dir
    if (-not (Test-ProtectedPath $fullPath.Path)) {
        Remove-Item -Recurse -Force $fullPath.Path
        Write-Host "Removed: $fullPath" -ForegroundColor Green
    } else {
        Write-Host "Protected: $fullPath" -ForegroundColor Blue
    }
}

# Remove coverage files
$coverageFiles = @("htmlcov", ".coverage", "coverage.xml", ".coverage.*")
foreach ($file in $coverageFiles) {
    if (Test-Path $file) {
        if (Test-ProtectedPath $file) {
            Write-Host "Protected: $file" -ForegroundColor Blue
        } else {
            Remove-Item -Recurse -Force $file
            Write-Host "Removed: $file" -ForegroundColor Green
        }
    }
}

# Remove other temporary files
$tempPatterns = @("*.tmp", "*.temp", "*.log", ".project_status*")
foreach ($pattern in $tempPatterns) {
    $tempFiles = Get-ChildItem -Path "." -File -Name $pattern -Recurse -ErrorAction SilentlyContinue
    foreach ($file in $tempFiles) {
        $fullPath = Resolve-Path $file
        if (-not (Test-ProtectedPath $fullPath.Path)) {
            Remove-Item -Force $fullPath.Path
            Write-Host "Removed: $fullPath" -ForegroundColor Green
        } else {
            Write-Host "Protected: $fullPath" -ForegroundColor Blue
        }
    }
}

# Check deployment files (don't auto-remove)
$unnecessaryDeployFiles = @(
    "Dockerfile",
    "docker-compose.yml",
    "deploy.sh",
    "deploy.prototxt"
)

Write-Host "`nChecking deployment files..." -ForegroundColor Yellow
foreach ($file in $unnecessaryDeployFiles) {
    if (Test-Path $file) {
        Write-Host "Review needed: $file (marked for potential removal)" -ForegroundColor Yellow
    }
}

# Check for duplicate test files
Write-Host "`nChecking for duplicate test files..." -ForegroundColor Yellow
$duplicateTestFiles = @("test_single.py", "test_comparison.py")
foreach ($file in $duplicateTestFiles) {
    if (Test-Path $file) {
        Write-Host "Duplicate test file found: $file (consider moving to tests/ folder)" -ForegroundColor Yellow
    }
}

Write-Host "`nSafe cleanup completed!" -ForegroundColor Green
Write-Host "All critical files and folders have been preserved" -ForegroundColor Blue
Write-Host "Protected items:" -ForegroundColor Cyan
foreach ($file in $protectedFiles) {
    Write-Host "   * $file" -ForegroundColor White
}
foreach ($folder in $protectedFolders) {
    Write-Host "   * $folder/ (and all contents)" -ForegroundColor White
}

Write-Host "`nCurrent project structure:" -ForegroundColor Cyan
Get-ChildItem -Name | Sort-Object | ForEach-Object {
    if ($protectedFiles -contains $_ -or $protectedFolders -contains $_) {
        Write-Host "   * $_ [PROTECTED]" -ForegroundColor Green
    } else {
        Write-Host "   * $_" -ForegroundColor White
    }
}