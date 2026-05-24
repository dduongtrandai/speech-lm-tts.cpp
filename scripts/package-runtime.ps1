param (
    [string]$BuildDir = "build",
    [string]$OutputDir = "dist",
    [string]$OnnxRuntimeRoot = $env:ONNXRUNTIME_ROOT
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

Write-Host "========================================="
Write-Host "Packaging SpeechLM TTS Runtime Release..."
Write-Host "========================================="

# Create output dir
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$ZipPath = Join-Path $OutputDir "speech-lm-tts-win-cpu.zip"
if (Test-Path $ZipPath) {
    Remove-Item $ZipPath
}

# Temporary staging dir
$StagingDir = Join-Path $OutputDir "stage"
if (Test-Path $StagingDir) {
    Remove-Item -Recurse -Force $StagingDir
}
New-Item -ItemType Directory -Path $StagingDir | Out-Null

function Resolve-BuildArtifactPath {
    param(
        [string]$BaseDir,
        [string]$Filename
    )

    $candidates = @(
        (Join-Path $BaseDir $Filename),
        (Join-Path (Join-Path $BaseDir "Release") $Filename),
        (Join-Path (Join-Path $BaseDir "RelWithDebInfo") $Filename)
    )

    foreach ($p in $candidates) {
        if (Test-Path $p) {
            return $p
        }
    }
    return ""
}

function Resolve-FirstExistingPath {
    param(
        [string[]]$Candidates
    )

    foreach ($p in $Candidates) {
        if ($p -and (Test-Path $p)) {
            return $p
        }
    }
    return ""
}

# Verify build artifacts
$DllPath = Resolve-BuildArtifactPath -BaseDir $BuildDir -Filename "speechlm-tts.dll"
$CliPath = Resolve-BuildArtifactPath -BaseDir $BuildDir -Filename "speechlm-tts-cli.exe"

if (-not $DllPath) {
    Write-Error "Could not find build library speechlm-tts.dll under '$BuildDir'. Build the project first."
}
if (-not $CliPath) {
    Write-Error "Could not find build CLI tool speechlm-tts-cli.exe under '$BuildDir'. Build the project first."
}

# Copy DLL and CLI
Copy-Item $DllPath $StagingDir
Copy-Item $CliPath $StagingDir

# Copy onnxruntime.dll
$OrtCandidates = @()
if ($OnnxRuntimeRoot) {
    $OrtCandidates += (Join-Path $OnnxRuntimeRoot "bin\onnxruntime.dll")
    $OrtCandidates += (Join-Path $OnnxRuntimeRoot "lib\onnxruntime.dll")
    $OrtCandidates += (Join-Path $OnnxRuntimeRoot "onnxruntime.dll")
}
$OrtCandidates += (Join-Path $BuildDir "onnxruntime.dll")
$OrtCandidates += (Join-Path (Join-Path $BuildDir "Release") "onnxruntime.dll")
$OrtCandidates += (Join-Path (Join-Path $BuildDir "RelWithDebInfo") "onnxruntime.dll")

$OrtDllSrc = Resolve-FirstExistingPath -Candidates $OrtCandidates
if (-not $OrtDllSrc) {
    $OrtFallback = Resolve-BuildArtifactPath -BaseDir $BuildDir -Filename "onnxruntime.dll"
    if ($OrtFallback) {
        $OrtDllSrc = $OrtFallback
    }
}

if (-not $OrtDllSrc) {
    $diag = ($OrtCandidates | ForEach-Object { " - $_" }) -join "`n"
    Write-Error "onnxruntime.dll is required but was not found. Checked:`n$diag"
}

Write-Host "Copying ONNX Runtime DLL from: $OrtDllSrc"
Copy-Item $OrtDllSrc $StagingDir

# Copy optional llama.cpp shared DLLs (like llama.dll, ggml.dll if built as shared)
foreach ($dll in @("llama.dll", "ggml.dll")) {
    $src = Join-Path $BuildDir $dll
    if (Test-Path $src) {
        Write-Host "Copying llama.cpp DLL: $dll"
        Copy-Item $src $StagingDir
    }
}

# Copy LICENSE files if they exist
$LicensesDir = Join-Path $StagingDir "LICENSES"
New-Item -ItemType Directory -Path $LicensesDir | Out-Null
$LicenseFiles = @(
    (Join-Path $ProjectRoot "LICENSE"),
    (Join-Path $ProjectRoot "README.md"),
    (Join-Path $ProjectRoot "llama.cpp\LICENSE")
)
foreach ($lf in $LicenseFiles) {
    if (Test-Path $lf) {
        Copy-Item $lf $LicensesDir
    }
}

# Write backend-manifest.json
$ManifestPath = Join-Path $StagingDir "backend-manifest.json"
$ManifestContent = @{
    id = "speech-lm-tts-win-x86_64-cpu"
    name = "CPU SpeechLM TTS Pipeline (x64)"
    version = "v0.1.0"
    type = "tts"
    engineFamily = "speech-lm-tts"
    variant = "win-x86_64-cpu"
    library = "speechlm-tts.dll"
    metadata = @{
        backend = "speech-lm-tts"
        components = @{
            "llama.cpp" = "b4600" # fallback/placeholder commit
            "onnxruntime" = "1.17.1"
        }
        profiles = @("neutts-air-v1", "vieneu-v2-turbo")
    }
}

$ManifestContent | ConvertTo-Json -Depth 5 | Out-File -FilePath $ManifestPath -Encoding utf8

# Compress to zip file
Write-Host "Compressing stage files to: $ZipPath..."
Compress-Archive -Path (Join-Path $StagingDir "*") -DestinationPath $ZipPath

# Cleanup staging dir
Remove-Item -Recurse -Force $StagingDir

Write-Host "Successfully packaged: $ZipPath"
