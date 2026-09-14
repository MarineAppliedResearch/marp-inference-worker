[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$PayloadRoot,
    [Parameter(Mandatory)][string]$InstallRoot,
    [Parameter(Mandatory)][string]$CoordinatorUrl,
    [Parameter(Mandatory)][string]$ActivationCodeFile
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$StateRoot = Join-Path $InstallRoot 'state'
$DownloadRoot = Join-Path $StateRoot 'setup-downloads'

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

if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
    throw 'No NVIDIA driver was found. Install a supported NVIDIA driver, then run MARP setup again.'
}

New-Item -ItemType Directory -Force -Path $VersionRoot, $StateRoot, $DownloadRoot | Out-Null
$UvLock = Read-Lock 'uv-windows-x64.lock.json'
$UvArchive = Join-Path $DownloadRoot 'uv.zip'
Get-VerifiedArchive $UvLock $UvArchive
$UvRoot = Join-Path $DownloadRoot 'uv'
if (Test-Path -LiteralPath $UvRoot) { Remove-Item -LiteralPath $UvRoot -Recurse -Force }
Expand-Archive -LiteralPath $UvArchive -DestinationPath $UvRoot
$Uv = Join-Path $UvRoot 'uv.exe'

$env:UV_PYTHON_INSTALL_DIR = Join-Path $InstallRoot 'python'
$env:UV_CACHE_DIR = Join-Path $StateRoot 'uv-cache'
& $Uv python install 3.12
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 installation failed.' }
$Runtime = Join-Path $VersionRoot 'runtime'
& $Uv venv $Runtime --python 3.12 --python-preference only-managed
if ($LASTEXITCODE -ne 0) { throw 'Worker runtime creation failed.' }
$Python = Join-Path $Runtime 'Scripts\python.exe'

$Wheel = Get-ChildItem -LiteralPath (Join-Path $PayloadRoot 'wheels') -Filter 'cython_bbox-*.whl' | Select-Object -First 1
if (-not $Wheel) { throw 'The prebuilt ByteTrack dependency is missing.' }
& $Uv pip install --python $Python $Wheel.FullName
if ($LASTEXITCODE -ne 0) { throw 'ByteTrack dependency installation failed.' }
& $Uv pip sync --python $Python (Join-Path $PayloadRoot 'requirements-windows-cu126.lock.txt') --torch-backend cu126
if ($LASTEXITCODE -ne 0) { throw 'Worker dependency installation failed.' }

$SourceRoot = Join-Path $VersionRoot 'source'
Expand-Archive -LiteralPath (Join-Path $PayloadRoot 'worker-source.zip') -DestinationPath $SourceRoot
& $Uv pip install --python $Python --no-deps --editable $SourceRoot
if ($LASTEXITCODE -ne 0) { throw 'MARP worker installation failed.' }

Copy-Item -LiteralPath (Join-Path $PayloadRoot 'player') -Destination $VersionRoot -Recurse -Force
$ChromeLock = Read-Lock 'chromium-windows-x64.lock.json'
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
