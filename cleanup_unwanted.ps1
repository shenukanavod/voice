# Cleanup Script - Remove Unwanted Files
# Keeps: train_voice_model.py, dataset folder, main application files

Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("="*79) -ForegroundColor Cyan
Write-Host "  CLEANUP SCRIPT - Removing Unwanted Files" -ForegroundColor Yellow
Write-Host ("="*80) -ForegroundColor Cyan
Write-Host ""

$deletedCount = 0
$savedSpace = 0

# Function to safely delete file
function Remove-SafeFile {
    param($path)
    if (Test-Path $path) {
        $size = (Get-Item $path).Length
        Remove-Item $path -Force
        $script:deletedCount++
        $script:savedSpace += $size
        Write-Host "  [DELETED] $path" -ForegroundColor Green
        return $true
    }
    return $false
}

# Function to safely delete folder
function Remove-SafeFolder {
    param($path)
    if (Test-Path $path) {
        $size = (Get-ChildItem $path -Recurse | Measure-Object -Property Length -Sum).Sum
        Remove-Item $path -Recurse -Force
        $script:deletedCount++
        $script:savedSpace += $size
        Write-Host "  [DELETED] $path (folder)" -ForegroundColor Green
        return $true
    }
    return $false
}

Write-Host "Step 1: Removing Test & Debug Files..." -ForegroundColor Cyan
Write-Host ""
Remove-SafeFile "test_dataset_loading.py"
Remove-SafeFile "test_embeddings.py"
Remove-SafeFile "test_full_pipeline.py"
Remove-SafeFile "test_model_initialization.py"
Remove-SafeFile "test_training_loop.py"
Remove-SafeFile "verify_all_code.py"
Remove-SafeFile "debug_model.py"
Remove-SafeFile "check_dataset_quality.py"
Remove-SafeFile "analyze_dataset.py"
Remove-SafeFile "show_model_details.py"

Write-Host ""
Write-Host "Step 2: Removing Old Training Scripts..." -ForegroundColor Cyan
Write-Host ""
Remove-SafeFile "start_training.py"
Remove-SafeFile "START_FAST_TRAINING.py"
Remove-SafeFile "monitor_training_progress.py"

Write-Host ""
Write-Host "Step 3: Removing Documentation Files..." -ForegroundColor Cyan
Write-Host ""
Remove-SafeFile "ARCHITECTURE_EXPLANATION.md"
Remove-SafeFile "TRAINING_GUIDE.md"
Remove-SafeFile "TROUBLESHOOTING_TRAINING.md"
Remove-SafeFile "VERIFICATION_REPORT.md"

Write-Host ""
Write-Host "Step 4: Removing Old/Redundant Model Files..." -ForegroundColor Cyan
Write-Host "  (Keeping: speaker_embedding_model.pth & training_history.json)" -ForegroundColor Yellow
Write-Host ""
Remove-SafeFile "models/fast_half_dataset_model.pth"
Remove-SafeFile "models/cnn_lstm_mfcc_model.pth"
Remove-SafeFile "models/high_accuracy_siamese_triplet.pth"
Remove-SafeFile "models/speaker_embedding_cpu.pth"
Remove-SafeFile "models/trained_30hour_600spk.pth"
Remove-SafeFile "models/fast_training_history.json"

Write-Host ""
Write-Host "Step 5: Removing Python Cache Files..." -ForegroundColor Cyan
Write-Host ""
if (Test-Path "__pycache__") {
    Remove-SafeFolder "__pycache__"
}
if (Test-Path "app/__pycache__") {
    Remove-SafeFolder "app/__pycache__"
}
if (Test-Path "auth_system/__pycache__") {
    Remove-SafeFolder "auth_system/__pycache__"
}
Get-ChildItem -Path . -Filter "__pycache__" -Recurse -Directory | ForEach-Object {
    Remove-SafeFolder $_.FullName
}

Write-Host ""
Write-Host "Step 6: Removing Optional Standalone Files..." -ForegroundColor Cyan
Write-Host ""
Remove-SafeFile "monitoring_window.py"
Remove-SafeFile "voice_security.py"

Write-Host ""
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("="*79) -ForegroundColor Cyan
Write-Host "  CLEANUP COMPLETE" -ForegroundColor Green
Write-Host ("="*80) -ForegroundColor Cyan
Write-Host ""
Write-Host "  Files/Folders Deleted: $deletedCount" -ForegroundColor Yellow
$savedSpaceMB = [math]::Round($savedSpace / 1MB, 2)
Write-Host "  Space Saved: $savedSpaceMB MB" -ForegroundColor Yellow
Write-Host ""
Write-Host "KEPT FILES:" -ForegroundColor Green
Write-Host "  - train_voice_model.py (main training script)" -ForegroundColor White
Write-Host "  - dataset/ folder (training data)" -ForegroundColor White
Write-Host "  - desktop_app.py (main application)" -ForegroundColor White
Write-Host "  - app/ folder (core modules)" -ForegroundColor White
Write-Host "  - auth_system/ folder (authentication)" -ForegroundColor White
Write-Host "  - models/speaker_embedding_model.pth (86.53% accuracy model)" -ForegroundColor White
Write-Host "  - models/training_history.json (training metrics)" -ForegroundColor White
Write-Host "  - All core service files" -ForegroundColor White
Write-Host ""
Write-Host "Your voice system is ready to use! Run: python desktop_app.py" -ForegroundColor Cyan
Write-Host ""
