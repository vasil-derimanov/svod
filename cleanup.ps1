# SVOD Project Cleanup Script
# Version: 1.0.0
# Last Updated: 2025-09-14
# Follows rules from copilot-instructions.md for safe project cleanup

Write-Host "🧹 SVOD Cleanup Script v1.0.0 - Safe project cleanup following copilot-instructions.md" -ForegroundColor Cyan
Write-Host "⚠️  This script will only remove truly unnecessary files and folders" -ForegroundColor Yellow
Write-Host "✅ All critical files and folders will be preserved" -ForegroundColor Green

# Critical files and folders that MUST NOT be deleted (from copilot-instructions.md)
$protectedFiles = @(
    "video_orientation_detector.py",
    "video_orientation_detector_old.py",
    "test_batch.py",
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
function Is-Protected {
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
    "coco.names",           # Not needed for current YOLOv8 setup
    "mobilenet-v2.bin",     # Old OpenVINO model
    "mobilenet-v2.xml",     # Old OpenVINO model
    "yolov4.cfg",           # Old YOLOv4 config
    "yolov4.weights"        # Old YOLOv4 weights
)

Write-Host "`n📂 Removing unnecessary model files..." -ForegroundColor Yellow
foreach ($file in $unnecessaryModelFiles) {
    if (Test-Path $file) {
        if (Is-Protected $file) {
            Write-Host "🛡️  Protected: $file" -ForegroundColor Blue
        } else {
            Remove-Item -Force $file
            Write-Host "✅ Removed: $file" -ForegroundColor Green
        }
    } else {
        Write-Host "⚪ Not found: $file" -ForegroundColor Gray
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

Write-Host "`n🗂️ Removing old test virtual environments..." -ForegroundColor Yellow
foreach ($env in $oldTestEnvs) {
    if (Test-Path $env) {
        if (Is-Protected $env) {
            Write-Host "🛡️  Protected: $env" -ForegroundColor Blue
        } else {
            Remove-Item -Recurse -Force $env
            Write-Host "✅ Removed: $env" -ForegroundColor Green
        }
    } else {
        Write-Host "⚪ Not found: $env" -ForegroundColor Gray
    }
}

# Remove temporary and cache files (but preserve protected folders)
Write-Host "`n🗑️ Removing temporary and cache files..." -ForegroundColor Yellow

# Remove __pycache__ directories (but not inside protected folders)
$pyCacheDirs = Get-ChildItem -Path "." -Directory -Name "__pycache__" -Recurse -ErrorAction SilentlyContinue
foreach ($dir in $pyCacheDirs) {
    $fullPath = Resolve-Path $dir
    if (-not (Is-Protected $fullPath.Path)) {
        Remove-Item -Recurse -Force $fullPath.Path
        Write-Host "✅ Removed: $fullPath" -ForegroundColor Green
    } else {
        Write-Host "🛡️  Protected: $fullPath" -ForegroundColor Blue
    }
}

# Remove .pytest_cache directories
$pytestCacheDirs = Get-ChildItem -Path "." -Directory -Name ".pytest_cache" -Recurse -ErrorAction SilentlyContinue
foreach ($dir in $pytestCacheDirs) {
    $fullPath = Resolve-Path $dir
    if (-not (Is-Protected $fullPath.Path)) {
        Remove-Item -Recurse -Force $fullPath.Path
        Write-Host "✅ Removed: $fullPath" -ForegroundColor Green
    } else {
        Write-Host "🛡️  Protected: $fullPath" -ForegroundColor Blue
    }
}

# Remove coverage files
$coverageFiles = @("htmlcov", ".coverage", "coverage.xml", ".coverage.*")
foreach ($file in $coverageFiles) {
    if (Test-Path $file) {
        if (Is-Protected $file) {
            Write-Host "�️  Protected: $file" -ForegroundColor Blue
        } else {
            Remove-Item -Recurse -Force $file
            Write-Host "✅ Removed: $file" -ForegroundColor Green
        }
    }
}

# Remove other temporary files
$tempPatterns = @("*.tmp", "*.temp", "*.log", ".project_status*")
foreach ($pattern in $tempPatterns) {
    $tempFiles = Get-ChildItem -Path "." -File -Name $pattern -Recurse -ErrorAction SilentlyContinue
    foreach ($file in $tempFiles) {
        $fullPath = Resolve-Path $file
        if (-not (Is-Protected $fullPath.Path)) {
            Remove-Item -Force $fullPath.Path
            Write-Host "✅ Removed: $fullPath" -ForegroundColor Green
        } else {
            Write-Host "🛡️  Protected: $fullPath" -ForegroundColor Blue
        }
    }
}

# Remove unnecessary deployment files (keeping essential ones)
$unnecessaryDeployFiles = @(
    "Dockerfile",           # Old Docker setup
    "docker-compose.yml",   # Old Docker setup
    "deploy.sh",            # Keep for reference but mark as unnecessary
    "deploy.prototxt"       # Keep for reference but mark as unnecessary
)

Write-Host "`n🚀 Checking deployment files..." -ForegroundColor Yellow
foreach ($file in $unnecessaryDeployFiles) {
    if (Test-Path $file) {
        Write-Host "⚠️  Review needed: $file (marked for potential removal)" -ForegroundColor Yellow
        # Don't auto-remove deployment files - require manual review
    }
}

# Remove duplicate test files outside tests/ folder
Write-Host "`n🧪 Checking for duplicate test files..." -ForegroundColor Yellow
$duplicateTestFiles = @("test_single.py", "test_comparison.py")
foreach ($file in $duplicateTestFiles) {
    if (Test-Path $file) {
        Write-Host "⚠️  Duplicate test file found: $file (consider moving to tests/ folder)" -ForegroundColor Yellow
        # Don't auto-remove - require manual review
    }
}

Write-Host "`n✨ Safe cleanup completed!" -ForegroundColor Green
Write-Host "🛡️  All critical files and folders have been preserved" -ForegroundColor Blue
Write-Host "📋 Protected items:" -ForegroundColor Cyan
foreach ($file in $protectedFiles) {
    Write-Host "   • $file" -ForegroundColor White
}
foreach ($folder in $protectedFolders) {
    Write-Host "   • $folder/ (and all contents)" -ForegroundColor White
}

Write-Host "`n📂 Current project structure:" -ForegroundColor Cyan
Get-ChildItem -Name | Sort-Object | ForEach-Object {
    if ($protectedFiles -contains $_ -or $protectedFolders -contains $_) {
        Write-Host "   • $_ 🛡️" -ForegroundColor Green
    } else {
        Write-Host "   • $_" -ForegroundColor White
    }
}