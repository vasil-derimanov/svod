# SVOD Project Cleanup Script
# Removes all model files and test environments for vanilla testing

Write-Host "🧹 SVOD Cleanup Script - Preparing for vanilla testing" -ForegroundColor Cyan

# Remove all model files
$modelFiles = @(
    "coco.names",
    "deploy.prototxt", 
    "lbfmodel.yaml",
    "mobilenet-v2.bin",
    "mobilenet-v2.xml",
    "res10_300x300_ssd_iter_140000.caffemodel",
    "yolov4.cfg",
    "yolov4.weights"
)

Write-Host "`n📂 Removing model files..." -ForegroundColor Yellow
foreach ($file in $modelFiles) {
    if (Test-Path $file) {
        Remove-Item -Force $file
        Write-Host "✅ Removed: $file" -ForegroundColor Green
    } else {
        Write-Host "⚪ Not found: $file" -ForegroundColor Gray
    }
}

# Remove test virtual environments
$testEnvs = @(
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

Write-Host "`n🗂️ Removing test virtual environments..." -ForegroundColor Yellow
foreach ($env in $testEnvs) {
    if (Test-Path $env) {
        Remove-Item -Recurse -Force $env
        Write-Host "✅ Removed: $env" -ForegroundColor Green
    } else {
        Write-Host "⚪ Not found: $env" -ForegroundColor Gray
    }
}

# Remove temp/cache files
$tempFiles = @(
    "__pycache__",
    "*.pyc",
    "*.tmp",
    "*.temp",
    ".project_status",
    "models"
)

Write-Host "`n🗑️ Removing temporary files..." -ForegroundColor Yellow
foreach ($pattern in $tempFiles) {
    $found = Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue
    if ($found) {
        Remove-Item -Recurse -Force $pattern
        Write-Host "✅ Removed: $pattern" -ForegroundColor Green
    } else {
        Write-Host "⚪ Not found: $pattern" -ForegroundColor Gray
    }
}

Write-Host "`n✨ Cleanup completed! Ready for vanilla testing." -ForegroundColor Green
Write-Host "📋 Project now contains only:" -ForegroundColor Cyan
Get-ChildItem -Name | Sort-Object | ForEach-Object { Write-Host "   • $_" -ForegroundColor White }