[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$PayloadRoot,
    [Parameter(Mandatory)][string]$InstallRoot,
    [Parameter(Mandatory)][string]$CoordinatorUrl,
    [Parameter(Mandatory)][string]$ActivationCodeFile
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$ProgressPreference = 'SilentlyContinue'
$SetupLog = Join-Path $InstallRoot 'setup.log'
New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
Start-Transcript -LiteralPath $SetupLog -Force | Out-Null
trap {
    Write-Error ($_ | Out-String)
    try { Stop-Transcript | Out-Null } catch {}
    exit 1
}
$Host.UI.RawUI.WindowTitle = 'MARP Inference Worker Setup'
Write-Host '============================================================' -ForegroundColor Cyan
Write-Host '              MARP Inference Worker Setup' -ForegroundColor Cyan
Write-Host ' Marine Applied Research GPU volunteer worker installation' -ForegroundColor Cyan
Write-Host '============================================================' -ForegroundColor Cyan
$StateRoot = Join-Path $InstallRoot 'state'
$DownloadRoot = Join-Path $StateRoot 'setup-downloads'

function Write-Stage([int]$Number, [string]$Message) {
    Write-Host ''
    Write-Host "[$Number/8] $Message" -ForegroundColor Cyan
}

function Read-Lock([string]$Name) {
    Get-Content -LiteralPath (Join-Path $PayloadRoot $Name) -Raw | ConvertFrom-Json
}

function Get-VerifiedArchive($Lock, [string]$Destination) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    Invoke-WebRequest -Uri $Lock.url -OutFile $Destination -UseBasicParsing
    $File = Get-Item -LiteralPath $Destination
    if ($File.Length -ne [int64]$Lock.size_bytes) { throw "$($Lock.name) download has the wrong size." }
    $Hash = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($Hash -ne $Lock.sha256) { throw "$($Lock.name) download failed SHA-256 verification." }
}

$BootstrapManifest = Read-Lock 'bootstrap-manifest.json'
$ReleaseKey = "$($BootstrapManifest.version)-$($BootstrapManifest.compute_runtime)"
$VersionRoot = Join-Path $InstallRoot "versions\$ReleaseKey"

Write-Stage 1 'Checking the NVIDIA graphics driver...'
if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
    throw 'No NVIDIA driver was found. Install a supported NVIDIA driver, then run MARP setup again.'
}

if (Test-Path -LiteralPath $VersionRoot) {
    Remove-Item -LiteralPath $VersionRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $VersionRoot, $StateRoot, $DownloadRoot | Out-Null
Write-Stage 2 'Downloading the verified setup tools...'
$UvLock = Read-Lock 'uv-windows-x64.lock.json'
$UvArchive = Join-Path $DownloadRoot 'uv.zip'
Get-VerifiedArchive $UvLock $UvArchive
$UvRoot = Join-Path $DownloadRoot 'uv'
if (Test-Path -LiteralPath $UvRoot) { Remove-Item -LiteralPath $UvRoot -Recurse -Force }
Expand-Archive -LiteralPath $UvArchive -DestinationPath $UvRoot
$Uv = Join-Path $UvRoot 'uv.exe'

$env:UV_PYTHON_INSTALL_DIR = Join-Path $InstallRoot 'python'
$env:UV_CACHE_DIR = Join-Path $StateRoot 'uv-cache'
Write-Stage 3 'Installing the managed Python 3.12 runtime...'
$PreviousErrorPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $Uv python install 3.12 2>$null | Out-Null
$ErrorActionPreference = $PreviousErrorPreference
$ManagedPython = Get-ChildItem -LiteralPath $env:UV_PYTHON_INSTALL_DIR -Directory |
    Where-Object { $_.Name -match '^cpython-3\.12\.\d+-windows-x86_64-none$' } |
    Sort-Object Name -Descending |
    ForEach-Object { Join-Path $_.FullName 'python.exe' } |
    Where-Object { Test-Path -LiteralPath $_ } |
    Select-Object -First 1
if (-not $ManagedPython) { throw 'Python 3.12 installation failed.' }
$Runtime = Join-Path $VersionRoot 'runtime'
Write-Stage 4 'Creating an isolated MARP worker environment...'
& $ManagedPython -m venv --without-pip $Runtime
if ($LASTEXITCODE -ne 0) { throw 'Worker runtime creation failed.' }
$Python = Join-Path $Runtime 'Scripts\python.exe'

Write-Stage 5 'Installing CUDA 12.6 inference requirements. This is the longest step...'
$Wheel = Get-ChildItem -LiteralPath (Join-Path $PayloadRoot 'wheels') -Filter 'cython_bbox-*.whl' | Select-Object -First 1
if (-not $Wheel) { throw 'The prebuilt ByteTrack dependency is missing.' }
& $Uv pip install --python $Python --no-deps $Wheel.FullName
if ($LASTEXITCODE -ne 0) { throw 'ByteTrack dependency installation failed.' }
& $Uv pip sync --python $Python (Join-Path $PayloadRoot 'requirements-windows-cu126.lock.txt') --torch-backend cu126
if ($LASTEXITCODE -ne 0) { throw 'Worker dependency installation failed.' }

$SourceRoot = Join-Path $VersionRoot 'source'
Write-Stage 6 'Installing the MARP worker and video display...'
Expand-Archive -LiteralPath (Join-Path $PayloadRoot 'worker-source.zip') -DestinationPath $SourceRoot
& $Uv pip install --python $Python --no-deps --editable $SourceRoot
if ($LASTEXITCODE -ne 0) { throw 'MARP worker installation failed.' }

Copy-Item -LiteralPath (Join-Path $PayloadRoot 'player') -Destination $VersionRoot -Recurse -Force
$ChromeLock = Read-Lock 'chromium-windows-x64.lock.json'
Write-Host 'Downloading the packaged video display...'
$ChromeArchive = Join-Path $DownloadRoot 'chromium.zip'
Get-VerifiedArchive $ChromeLock $ChromeArchive
$ChromeExtract = Join-Path $DownloadRoot 'chromium'
if (Test-Path -LiteralPath $ChromeExtract) { Remove-Item -LiteralPath $ChromeExtract -Recurse -Force }
Expand-Archive -LiteralPath $ChromeArchive -DestinationPath $ChromeExtract
Move-Item -LiteralPath (Join-Path $ChromeExtract 'chrome-win') -Destination (Join-Path $VersionRoot 'chromium')

@{
    version = $BootstrapManifest.version
    platform = 'windows'
    architecture = 'x86_64'
    compute_runtime = $BootstrapManifest.compute_runtime
    entrypoint = 'runtime/Scripts/marp-worker.exe'
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $VersionRoot 'manifest.json') -Encoding utf8

Copy-Item -LiteralPath (Join-Path $PayloadRoot 'launcher-windows.ps1') -Destination (Join-Path $InstallRoot 'launcher.ps1') -Force
@{ coordinator_url = $CoordinatorUrl.TrimEnd('/'); screen = 'window'; api_port = 8010 } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $InstallRoot 'config.json') -Encoding utf8
@{ release = $ReleaseKey } | ConvertTo-Json |
    Set-Content -LiteralPath (Join-Path $InstallRoot 'active.json') -Encoding utf8

$env:MARP_COORDINATOR_URL = $CoordinatorUrl
$env:MARP_WORKER_STATE_DIR = $StateRoot
Write-Stage 7 'Connecting this computer to MARP...'
& $Python -m marp_inference_worker.worker_main --activate-code-file $ActivationCodeFile `
    --coordinator-url $CoordinatorUrl --state-dir $StateRoot
if ($LASTEXITCODE -ne 0) { throw 'Worker activation failed.' }

$Launcher = Join-Path $InstallRoot 'launcher.ps1'
$LauncherArguments = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Launcher`" -Screen window"
$WorkerProcess = Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -ArgumentList $LauncherArguments `
    -PassThru `
    -WindowStyle Hidden
$Healthy = $false
Write-Stage 8 'Starting the worker and checking its GPU...'
for ($Attempt = 0; $Attempt -lt 60; $Attempt += 1) {
    try {
        $Health = Invoke-RestMethod -Uri 'http://127.0.0.1:8010/health' -TimeoutSec 2
        $Status = Invoke-RestMethod -Uri 'http://127.0.0.1:8010/status' -TimeoutSec 2
        if ($Health.status -eq 'ok' -and $Status.capabilities -and $Status.enrolled) {
            $Healthy = $true
            break
        }
    } catch { }
    Start-Sleep -Seconds 1
}
if (-not $Healthy) {
    & taskkill.exe /PID $WorkerProcess.Id /T /F 2>$null | Out-Null
    throw 'The worker installed but did not report healthy enrollment and GPU discovery.'
}

Remove-Item -LiteralPath $UvArchive, $ChromeArchive -Force -ErrorAction SilentlyContinue
Write-Host ''
Write-Host 'MARP Inference Worker is installed and ready.' -ForegroundColor Green
Stop-Transcript | Out-Null
