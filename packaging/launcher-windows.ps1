[CmdletBinding()]
param(
    [ValidateSet('off', 'window', 'fullscreen')]
    [string]$Screen = 'window',
    [ValidateSet('finish', 'stop', 'resume')]
    [string]$Control
)

$ErrorActionPreference = 'Stop'
$InstallRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Active = Get-Content -LiteralPath (Join-Path $InstallRoot 'active.json') -Raw | ConvertFrom-Json
$VersionRoot = Join-Path $InstallRoot "versions\$($Active.release)"
$Python = Join-Path $VersionRoot 'runtime\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) { throw "The active MARP worker runtime is missing." }

$env:MARP_WORKER_INSTALL_ROOT = $InstallRoot
$env:MARP_WORKER_STATE_DIR = Join-Path $InstallRoot 'state'
$env:MARP_PLAYER_HOST_DIR = Join-Path $VersionRoot 'player'
$env:MARP_CHROMIUM_PATH = Join-Path $VersionRoot 'chromium\chrome.exe'
$Config = Get-Content -LiteralPath (Join-Path $InstallRoot 'config.json') -Raw | ConvertFrom-Json
$env:MARP_COORDINATOR_URL = $Config.coordinator_url
$env:MARP_WORKER_API_PORT = [string]$Config.api_port

if ($Control) {
    & $Python -m marp_inference_worker.installation.launcher --control $Control
} else {
    & $Python -m marp_inference_worker.installation.launcher --screen $Screen
}
exit $LASTEXITCODE
