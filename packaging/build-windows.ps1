[CmdletBinding()]
param(
    [string]$Version,
    [string]$PlayerRepository = (Join-Path $PSScriptRoot '..\..\marp-video-player'),
    [string]$OutputRoot = (Join-Path $PSScriptRoot '..\.marp\local\windows-build'),
    [string]$BuildPython = 'py',
    [string]$IsccPath,
    [Parameter(Mandatory)][string]$CoordinatorUrl,
    [Parameter(Mandatory)][string]$EnrollmentCode,
    [string]$SigningCertificateThumbprint,
    [switch]$Development
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Repository = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)

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
    throw 'The Windows bootstrap must be built on Windows x64.'
}
if (-not $Development -and -not $SigningCertificateThumbprint) {
    throw 'Volunteer artifacts require code signing. Use -Development for an unsigned pilot.'
}

Push-Location $Repository
try {
    if (git status --porcelain --untracked-files=no) {
        throw 'Commit the worker branch before building so the embedded source has an exact revision.'
    }
    $WorkerCommit = (git rev-parse HEAD).Trim()
    Assert-LastExit 'Reading the worker revision'
    if (-not $Version) {
        $ProjectText = Get-Content -LiteralPath (Join-Path $Repository 'pyproject.toml') -Raw
        $Version = [regex]::Match($ProjectText, '(?m)^version = "([^"]+)"\r?$').Groups[1].Value
    }

    if (Test-Path -LiteralPath $OutputRoot) { Remove-Item -LiteralPath $OutputRoot -Recurse -Force }
    $Payload = New-Item -ItemType Directory -Force -Path (Join-Path $OutputRoot 'payload')
    $Wheels = New-Item -ItemType Directory -Force -Path (Join-Path $Payload 'wheels')
    $InstallerDir = New-Item -ItemType Directory -Force -Path (Join-Path $OutputRoot 'installer')

    $WorkerZip = Join-Path $Payload 'worker-source.zip'
    git archive --format=zip --output=$WorkerZip $WorkerCommit
    Assert-LastExit 'Exporting committed worker source'
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'bootstrap-windows.ps1') -Destination $Payload
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'launcher-windows.ps1') -Destination $Payload
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'requirements-windows-cu126.lock.txt') -Destination $Payload
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'uv-windows-x64.lock.json') -Destination $Payload
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'chromium-windows-x64.lock.json') -Destination $Payload
    [ordered]@{
        version = $Version
        compute_runtime = 'cuda12.6'
        worker_commit = $WorkerCommit
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Payload 'bootstrap-manifest.json') -Encoding utf8

    # The destination computer must not need Visual Studio. This tiny compiled
    # dependency is built once here and carried by the small bootstrap.
    $BuildPythonArguments = if ([IO.Path]::GetFileName($BuildPython) -in @('py', 'py.exe')) { @('-3.12') } else { @() }
    & $BuildPython @BuildPythonArguments -m pip wheel cython-bbox==0.1.5 --no-deps --wheel-dir $Wheels
    Assert-LastExit 'Building the cython-bbox Windows wheel'

    $PlayerLock = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'player.lock.json') -Raw | ConvertFrom-Json
    $PlayerRepository = (Resolve-Path $PlayerRepository).Path
    $PlayerSource = Join-Path $OutputRoot 'player-source'
    $PlayerZip = Join-Path $OutputRoot 'player-source.zip'
    git -C $PlayerRepository archive --format=zip --output=$PlayerZip $PlayerLock.commit
    Assert-LastExit 'Exporting the pinned player source'
    Expand-Archive -LiteralPath $PlayerZip -DestinationPath $PlayerSource
    Push-Location $PlayerSource
    try {
        npm ci
        Assert-LastExit 'Installing player build dependencies'
        npm run build
        Assert-LastExit 'Building the pinned player'
        node tools/pack-host.mjs
        Assert-LastExit 'Staging the player host'
    } finally { Pop-Location }
    Copy-Item -LiteralPath (Join-Path $PlayerSource 'release\host') -Destination $Payload -Recurse
    Rename-Item -LiteralPath (Join-Path $Payload 'host') -NewName 'player'

    if (-not $IsccPath) {
        $IsccPath = @(
            (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe'),
            (Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe')
        ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
    }
    if (-not $IsccPath) { throw 'Inno Setup 6 was not found. Pass -IsccPath.' }
    & $IsccPath "/DPayloadDir=$Payload" "/DOutputDir=$InstallerDir" "/DWorkerVersion=$Version" `
        "/DCoordinatorUrl=$CoordinatorUrl" "/DEnrollmentCode=$EnrollmentCode" `
        (Join-Path $PSScriptRoot 'windows-installer.iss')
    Assert-LastExit 'Building the one-click installer'
    $Installers = @(Get-ChildItem -LiteralPath $InstallerDir -Filter '*.exe')
    if ($Installers.Count -ne 1) { throw "Expected one installer, found $($Installers.Count)." }
    $Installer = $Installers[0]
    Sign-Artifact $Installer.FullName

    $File = Get-Item -LiteralPath $Installer.FullName
    [ordered]@{
        version = $Version
        worker_commit = $WorkerCommit
        bootstrap_size_bytes = $File.Length
        bootstrap_sha256 = (Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        runtime = 'cuda12.6'
        runtime_delivery = 'downloaded during setup'
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $InstallerDir 'build-manifest.json') -Encoding utf8
    Write-Host "Built $($File.FullName) ($($File.Length) bytes)"
} finally {
    Pop-Location
}
