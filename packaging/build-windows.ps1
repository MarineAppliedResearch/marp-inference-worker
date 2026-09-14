[CmdletBinding()]
param(
    [string]$Version,
    [string]$PlayerRepository = (Join-Path $PSScriptRoot '..\..\marp-video-player'),
    [string]$OutputRoot = (Join-Path $PSScriptRoot '..\.marp\local\windows-build'),
    [string]$ChromeArchive,
    [string]$IsccPath,
    [string]$SigningCertificateThumbprint,
    [switch]$Development
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Repository = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)
$Runtime = 'cuda12.6'

function Read-Json([string]$Path) {
    Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Assert-LastExit([string]$Action) {
    if ($LASTEXITCODE -ne 0) { throw "$Action failed with exit code $LASTEXITCODE." }
}

function Sign-Artifact([string]$Path) {
    if (-not $SigningCertificateThumbprint) { return }
    $SignTool = (Get-Command signtool.exe -ErrorAction Stop).Source
    & $SignTool sign /sha1 $SigningCertificateThumbprint /fd SHA256 /td SHA256 /tr http://timestamp.digicert.com $Path
    Assert-LastExit "Signing $Path"
}

if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT -or $env:PROCESSOR_ARCHITECTURE -ne 'AMD64') {
    throw 'The Windows payload must be built on Windows x64.'
}
if (-not $Development -and -not $SigningCertificateThumbprint) {
    throw 'Volunteer artifacts require -SigningCertificateThumbprint. Use -Development for unsigned local builds.'
}

Push-Location $Repository
try {
    if (git status --porcelain) {
        throw 'Commit the worker branch before building so every artifact has a reproducible source revision.'
    }
    $WorkerCommit = (git rev-parse HEAD).Trim()
    Assert-LastExit 'Reading the worker revision'
    if (-not $Version) {
        $ProjectText = Get-Content -LiteralPath (Join-Path $Repository 'pyproject.toml') -Raw
        $Version = [regex]::Match($ProjectText, '(?m)^version = "([^"]+)"\r?$').Groups[1].Value
    }
    if ($Version -notmatch '^[A-Za-z0-9._-]+$') { throw 'Version contains unsafe path characters.' }
    $ReleaseKey = "$Version-$Runtime"

    $PlayerLock = Read-Json (Join-Path $PSScriptRoot 'player.lock.json')
    $ChromeLock = Read-Json (Join-Path $PSScriptRoot 'chromium-windows-x64.lock.json')
    $PlayerRepository = (Resolve-Path $PlayerRepository).Path
    git -C $PlayerRepository cat-file -e "$($PlayerLock.commit)^{commit}"
    Assert-LastExit 'Resolving the pinned player revision'

    if (Test-Path -LiteralPath $OutputRoot) { Remove-Item -LiteralPath $OutputRoot -Recurse -Force }
    $Stage = Join-Path $OutputRoot 'stage'
    $VersionDir = Join-Path $Stage "versions\$ReleaseKey"
    $LauncherDir = Join-Path $Stage 'launcher'
    $Work = Join-Path $OutputRoot 'work'
    New-Item -ItemType Directory -Force -Path $VersionDir, $LauncherDir, $Work | Out-Null

    $Uv = (Get-Command uv.exe -ErrorAction Stop).Source
    $env:UV_CACHE_DIR = Join-Path $Repository '.marp\local\uv-cache'
    $BuildVenv = Join-Path $Work 'venv'
    & $Uv venv $BuildVenv --python 3.12 --managed-python
    Assert-LastExit 'Creating the Python 3.12 build environment'
    $Python = Join-Path $BuildVenv 'Scripts\python.exe'
    $PythonVersion = (& $Python -c 'import platform; print(platform.python_version())').Trim()
    if (-not $PythonVersion.StartsWith('3.12.')) { throw "Expected Python 3.12, got $PythonVersion." }
    & $Uv pip sync --python $Python (Join-Path $PSScriptRoot 'requirements-windows-cu126.lock.txt') --torch-backend cu126
    Assert-LastExit 'Installing the locked CUDA 12.6 dependency set'
    & $Uv pip install --python $Python --no-deps $Repository
    Assert-LastExit 'Installing the worker source'

    $PyInstallerDist = Join-Path $Work 'pyinstaller-dist'
    $PyInstallerWork = Join-Path $Work 'pyinstaller-work'
    $PyInstallerCommon = @(
        '--noconfirm', '--clean',
        '--paths', (Join-Path $Repository 'ByteTrack'),
        '--distpath', $PyInstallerDist,
        '--workpath', $PyInstallerWork,
        '--specpath', $Work
    )
    & $Python -m PyInstaller @PyInstallerCommon --onedir --name marp-worker `
        --collect-all torch --collect-all ultralytics --collect-all yolox `
        (Join-Path $Repository 'src\marp_inference_worker\worker_main.py')
    Assert-LastExit 'Building the worker executable'
    Copy-Item -Path (Join-Path $PyInstallerDist 'marp-worker\*') -Destination $VersionDir -Recurse -Force

    & $Python -m PyInstaller @PyInstallerCommon --onefile --name marp-worker-launcher `
        (Join-Path $Repository 'src\marp_inference_worker\installation\launcher.py')
    Assert-LastExit 'Building the stable launcher'
    Copy-Item -LiteralPath (Join-Path $PyInstallerDist 'marp-worker-launcher.exe') -Destination $LauncherDir

    $PlayerZip = Join-Path $Work 'player-source.zip'
    $PlayerSource = Join-Path $Work 'player-source'
    git -C $PlayerRepository archive --format=zip --output=$PlayerZip $PlayerLock.commit
    Assert-LastExit 'Exporting the pinned player source'
    Expand-Archive -LiteralPath $PlayerZip -DestinationPath $PlayerSource
    Push-Location $PlayerSource
    try {
        npm ci
        Assert-LastExit 'Installing pinned player dependencies'
        npm run build
        Assert-LastExit 'Building the pinned player'
        npm run pack:host
        Assert-LastExit 'Packing the player host surface'
    } finally { Pop-Location }
    Copy-Item -Path (Join-Path $PlayerSource 'release\host\*') -Destination (New-Item -ItemType Directory -Force -Path (Join-Path $VersionDir 'player')) -Recurse -Force

    if (-not $ChromeArchive) {
        $ChromeArchive = Join-Path $Repository ".marp\local\chrome-win64-$($ChromeLock.version).zip"
    }
    if (-not (Test-Path -LiteralPath $ChromeArchive)) {
        New-Item -ItemType Directory -Force -Path (Split-Path $ChromeArchive) | Out-Null
        Invoke-WebRequest -Uri $ChromeLock.url -OutFile $ChromeArchive
    }
    $ChromeFile = Get-Item -LiteralPath $ChromeArchive
    if ($ChromeFile.Length -ne [int64]$ChromeLock.size_bytes) { throw 'Chromium archive size does not match its lock.' }
    if ((Get-FileHash -LiteralPath $ChromeArchive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ChromeLock.sha256) {
        throw 'Chromium archive hash does not match its lock.'
    }
    $ChromeSource = Join-Path $Work 'chromium'
    Expand-Archive -LiteralPath $ChromeArchive -DestinationPath $ChromeSource
    $ChromeDir = New-Item -ItemType Directory -Force -Path (Join-Path $VersionDir 'chromium')
    Copy-Item -Path (Join-Path $ChromeSource 'chrome-win64\*') -Destination $ChromeDir -Recurse -Force

    Sign-Artifact (Join-Path $VersionDir 'marp-worker.exe')
    Sign-Artifact (Join-Path $LauncherDir 'marp-worker-launcher.exe')

    $Manifest = [ordered]@{
        version = $Version
        release_key = $ReleaseKey
        platform = 'windows'
        architecture = 'x86_64'
        compute_runtime = $Runtime
        entrypoint = 'marp-worker.exe'
        worker_commit = $WorkerCommit
        player = [ordered]@{ version = $PlayerLock.version; commit = $PlayerLock.commit }
        chromium = [ordered]@{ name = $ChromeLock.name; version = $ChromeLock.version; sha256 = $ChromeLock.sha256 }
        python_version = $PythonVersion
        dependency_lock_sha256 = (Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'requirements-windows-cu126.lock.txt') -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $Manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $VersionDir 'manifest.json') -Encoding utf8

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $ReleaseDir = New-Item -ItemType Directory -Force -Path (Join-Path $OutputRoot 'release')
    $ReleaseZip = Join-Path $ReleaseDir "marp-inference-worker-$ReleaseKey.zip"
    [IO.Compression.ZipFile]::CreateFromDirectory($VersionDir, $ReleaseZip, [IO.Compression.CompressionLevel]::Optimal, $false)
    $ReleaseFile = Get-Item -LiteralPath $ReleaseZip
    $ReleaseSha = (Get-FileHash -LiteralPath $ReleaseZip -Algorithm SHA256).Hash.ToLowerInvariant()

    if (-not $IsccPath) {
        $Candidates = @(
            (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe'),
            (Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe')
        ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
        $IsccPath = $Candidates | Select-Object -First 1
    }
    if (-not $IsccPath) { throw 'Inno Setup 6 was not found. Pass -IsccPath.' }
    & $IsccPath "/DStageDir=$Stage" "/DWorkerVersion=$Version" "/DReleaseKey=$ReleaseKey" (Join-Path $PSScriptRoot 'windows-installer.iss')
    Assert-LastExit 'Building the Windows installer'
    $Installers = @(Get-ChildItem -LiteralPath (Join-Path $Stage 'installer') -Filter '*.exe')
    if ($Installers.Count -ne 1) { throw "Expected one installer, found $($Installers.Count)." }
    $Installer = $Installers[0]
    Sign-Artifact $Installer.FullName
    Copy-Item -LiteralPath $Installer.FullName -Destination $ReleaseDir

    [ordered]@{
        version = $Version
        platform = 'windows'
        architecture = 'x86_64'
        compute_runtime = $Runtime
        download_url = "https://github.com/MarineAppliedResearch/marp-inference-worker/releases/download/v$Version/$($ReleaseFile.Name)"
        size_bytes = $ReleaseFile.Length
        sha256 = $ReleaseSha
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ReleaseDir 'api-release.json') -Encoding utf8

    Write-Host "Built $ReleaseKey in $ReleaseDir"
    Write-Host "Update package: $($ReleaseFile.Name) ($($ReleaseFile.Length) bytes, SHA-256 $ReleaseSha)"
} finally {
    Pop-Location
}
