[CmdletBinding()]
param(
    [string]$Version,
    [string]$PlayerRepository = (Join-Path $PSScriptRoot '..\..\marp-video-player'),
    [string]$OutputRoot = (Join-Path $PSScriptRoot '..\.marp\local\windows-build'),
    [string]$BuildPython = 'py',
    [string]$IsccPath,
    [string]$CoordinatorUrl,
    [string]$EnrollmentCode,
    [string]$SigningCertificateThumbprint,
    [switch]$Development,
    [switch]$SkipEnrollmentCheck
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Repository = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$OutputRoot = [IO.Path]::GetFullPath($OutputRoot)

function Assert-LastExit([string]$Action) {
    if ($LASTEXITCODE -ne 0) { throw "$Action failed with exit code $LASTEXITCODE." }
}

# Runs a native command without letting its stderr abort the build.
#
# Under `$ErrorActionPreference = 'Stop'` PowerShell turns ANY stderr from a
# native command into a terminating error, so a pip or npm notice kills a build
# that actually succeeded. `pip wheel` wrote "A new release of pip is available"
# and the whole installer build stopped with the wheel already sitting on disk.
# The exit code is the truth; the umbrella's scripts learned this the same way.
function Invoke-Native {
    param(
        [Parameter(Mandatory)][string]$Action,
        [Parameter(Mandatory)][scriptblock]$Command
    )
    $Previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Command 2>&1 | ForEach-Object { "$_" } | Out-Host
    } finally {
        $ErrorActionPreference = $Previous
    }
    Assert-LastExit $Action
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

# Resolve-BuildPython()
# Finds a Python 3.12 to build the cython-bbox wheel with, and proves it is 3.12
# before the build does anything else.
# Inputs: $BuildPython, which may be a bare name or a full path.
# Output: a path to a python.exe that reported 3.12.
#
# The wheel is cp312 and the installed runtime is 3.12.14, so the build host
# needs 3.12 specifically -- not merely "a Python". The default was the `py`
# launcher, which is a BUILD-HOST ASSUMPTION rather than a fact: a machine with
# no launcher failed MINUTES in, at the wheel step, after the enrolment check
# and the ByteTrack prune had already run. marp-laptop-install-test hit exactly
# that. A build host missing its compiler should be told so in two seconds.
#
# `-3.12` is appended only for the launcher, because only the launcher takes it.
# That means passing an explicit interpreter path silently skips the version
# selection -- correct, but sharp enough to be worth checking the version here
# rather than trusting the caller pointed at the right one.
function Resolve-BuildPython {
    param([string]$Requested)

    $Candidates = @()
    if ($Requested) { $Candidates += $Requested }
    # Whatever uv laid down for an installed worker on this machine is a known
    # -good 3.12, and a build host is very often also a test host.
    $Managed = Join-Path $env:LOCALAPPDATA 'MARP\Worker\python'
    if (Test-Path -LiteralPath $Managed) {
        $Candidates += @(
            Get-ChildItem -LiteralPath $Managed -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match '^cpython-3\.12\.(\d+)-windows-x86_64-none$' } |
                ForEach-Object { [PSCustomObject]@{ Patch = [int]$Matches[1]; Path = Join-Path $_.FullName 'python.exe' } } |
                Where-Object { Test-Path -LiteralPath $_.Path } |
                Sort-Object Patch -Descending |
                ForEach-Object { $_.Path }
        )
    }

    $Tried = @()
    foreach ($Candidate in $Candidates) {
        $Command = Get-Command $Candidate -ErrorAction SilentlyContinue
        if (-not $Command) { $Tried += "$Candidate (not found)"; continue }
        $Executable = $Command.Source
        $Arguments = @()
        if ([IO.Path]::GetFileName($Executable) -in @('py', 'py.exe')) { $Arguments += '-3.12' }
        # No double quotes in this snippet, deliberately. Windows PowerShell
        # strips embedded double quotes when building a native command line, so
        # `print("%d.%d" % ...)` reaches Python as `print(%d.%d % ...)` and dies
        # with a SyntaxError -- which this function would then report as the
        # interpreter "not running", blaming a perfectly good Python. Found by
        # testing this resolver rather than by reading it.
        $Arguments += @('-c', 'import sys; print(sys.version.split()[0])')

        $Previous = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $Reported = (& $Executable $Arguments 2>$null | Select-Object -Last 1)
        $Code = $LASTEXITCODE
        $ErrorActionPreference = $Previous

        $Reported = "$Reported".Trim()
        if ($Code -ne 0) { $Tried += "$Candidate (would not run)"; continue }
        if ($Reported -notlike '3.12.*' -and $Reported -ne '3.12') {
            $Tried += "$Candidate (is $Reported, need 3.12)"; continue
        }
        Write-Host "Building the wheel with $Executable (Python $Reported)."
        return $Executable
    }

    throw ("No Python 3.12 was found to build the cython-bbox wheel with. Tried: " +
        ($Tried -join '; ') + ". Pass -BuildPython with a path to a 3.12 interpreter.")
}

$BuildPython = Resolve-BuildPython $BuildPython
if (-not $Development -and -not $SigningCertificateThumbprint) {
    throw 'Volunteer artifacts require code signing. Use -Development for an unsigned pilot.'
}

# Refuses to build an installer whose enrolment code the coordinator will not accept.
#
# -EnrollmentCode is an arbitrary string and nothing checked it, so a mistyped,
# revoked, expired or placeholder code produced an installer that built cleanly,
# looked correct, shipped, and then failed on the volunteer's machine at the
# activation stage with "Worker activation failed" -- which reads to them as a
# broken package. Found by building with a placeholder and watching a real
# install die at stage 7 of 8.
#
# The check is deliberately the real endpoint rather than a format test: what
# matters is whether THIS coordinator will accept THIS code today, which no
# amount of string validation can answer. `local_id` is obviously synthetic so
# the row this creates is identifiable as a build probe.
function Assert-EnrollmentCodeAccepted([string]$Url, [string]$Code) {
    $Probe = @{
        activation_code = $Code
        local_id = "buildcheck-$([guid]::NewGuid())"
        name = "buildcheck-$env:COMPUTERNAME"
        platform = 'windows'
        architecture = 'x86_64'
        compute_runtime = 'buildcheck'
    } | ConvertTo-Json

    try {
        Invoke-RestMethod -Uri "$($Url.TrimEnd('/'))/api/v2/gpu/workers/activate" `
            -Method POST -Body $Probe -ContentType 'application/json' -TimeoutSec 30 | Out-Null
        Write-Host 'Enrolment code accepted by the coordinator.'
    } catch {
        $Status = $null
        if ($_.Exception.Response) { $Status = [int]$_.Exception.Response.StatusCode }
        if ($Status -eq 401 -or $Status -eq 403 -or $Status -eq 409 -or $Status -eq 400) {
            throw ("The coordinator refused this enrolment code (HTTP $Status). Building would " +
                'produce an installer that fails on the volunteer machine rather than here. ' +
                'Seed or re-seed one with MARP_API scripts/seed-worker-enrolment-code.js.')
        }
        # Unreachable is not the same as refused, and a build host without a
        # route to the coordinator is a normal thing rather than a bad code.
        throw ("Could not reach $Url to check the enrolment code: $($_.Exception.Message). " +
            'Pass -SkipEnrollmentCheck to build anyway, accepting that the code is unverified.')
    }
}

# Finds the enrolment code and coordinator without anybody typing them.
#
# Rolling a new worker build should not mean pasting keys. Both values are
# deployment configuration for a build machine, not per-build arguments, so they
# are set once and then resolved in this order:
#
#   1. the parameter, for a one-off build against something else
#   2. the environment -- MARP_WORKER_ENROLMENT_CODE is the same name
#      MARP_API's seed-worker-enrolment-code.js accepts, deliberately, so one
#      variable serves seeding and building
#   3. .marp/local/<name>, which is git-ignored
#
# **The value never reaches the repository.** .marp/local/ is git-ignored by
# design and the committed tree carries no code at all; if a build ever writes
# one into a tracked file, that is a defect rather than a convenience.
function Resolve-BuildSetting {
    param(
        [string]$Provided,
        [Parameter(Mandatory)][string]$Environment,
        [Parameter(Mandatory)][string]$LocalFile,
        [Parameter(Mandatory)][string]$What,
        [Parameter(Mandatory)][string]$HowToSet
    )

    if ($Provided) { return $Provided }

    $FromEnvironment = [Environment]::GetEnvironmentVariable($Environment)
    if ($FromEnvironment) {
        Write-Host "Using $What from `$env:$Environment."
        return $FromEnvironment.Trim()
    }

    $Path = Join-Path $Repository ".marp\local\$LocalFile"
    if (Test-Path -LiteralPath $Path) {
        $FromFile = (Get-Content -LiteralPath $Path -Raw).Trim()
        if ($FromFile) {
            Write-Host "Using $What from .marp/local/$LocalFile."
            return $FromFile
        }
    }

    throw ("No $What was given. Set `$env:$Environment, or write it to " +
        ".marp/local/$LocalFile, or pass it explicitly. $HowToSet")
}

Push-Location $Repository
try {
    if (git status --porcelain --untracked-files=no) {
        throw 'Commit the worker branch before building so the embedded source has an exact revision.'
    }
    $WorkerCommit = (git rev-parse HEAD).Trim()
    Assert-LastExit 'Reading the worker revision'
    $WorkerRevision = $WorkerCommit.Substring(0, 8)

    $CoordinatorUrl = Resolve-BuildSetting -Provided $CoordinatorUrl `
        -Environment 'MARP_COORDINATOR_URL' -LocalFile 'coordinator-url.txt' `
        -What 'coordinator address' `
        -HowToSet 'It is the address volunteers built from this artifact will contact.'
    $EnrollmentCode = Resolve-BuildSetting -Provided $EnrollmentCode `
        -Environment 'MARP_WORKER_ENROLMENT_CODE' -LocalFile 'enrolment-code.txt' `
        -What 'enrolment code' `
        -HowToSet 'Seed one with MARP_API scripts/seed-worker-enrolment-code.js --apply.'

    if (-not $SkipEnrollmentCheck) { Assert-EnrollmentCodeAccepted $CoordinatorUrl $EnrollmentCode }
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

    # Drop ByteTrack's demo material from the exported source.
    #
    # It is 62 MB of the 65 MB ByteTrack occupies -- demo GIFs, a sample video
    # and the datasets directory -- and nothing imports any of it. Left in, the
    # volunteer downloads roughly sixty megabytes of demonstration footage
    # before anything useful happens, which made this package 66 MB against the
    # Linux one's 5.8 MB. The registry already notes the weight; this is the
    # first build that made it visible.
    #
    # Pruned from the archive rather than excluded by pathspec so the result
    # does not depend on a git version's pathspec handling, and `yolox` itself
    # is asserted present afterwards because it is what the tracking stage
    # imports.
    # Both assemblies: FileSystem carries ZipFile, and ZipArchiveMode lives in
    # System.IO.Compression. Loading only the first finds ZipFile and then fails
    # on the mode enum, which is not obvious from the error.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    Add-Type -AssemblyName System.IO.Compression
    $PrunedPrefixes = @('ByteTrack/assets/', 'ByteTrack/videos/', 'ByteTrack/datasets/', 'ByteTrack/docs/')
    $Archive = [IO.Compression.ZipFile]::Open($WorkerZip, [IO.Compression.ZipArchiveMode]::Update)
    try {
        $Doomed = @($Archive.Entries | Where-Object {
            $Entry = $_.FullName
            $PrunedPrefixes | Where-Object { $Entry.StartsWith($_, [StringComparison]::OrdinalIgnoreCase) }
        })
        foreach ($Entry in $Doomed) { $Entry.Delete() }
        $Yolox = @($Archive.Entries | Where-Object { $_.FullName.StartsWith('ByteTrack/yolox/', [StringComparison]::OrdinalIgnoreCase) })
        if ($Yolox.Count -eq 0) { throw 'Pruning removed the vendored yolox package, which the tracking stage imports.' }
        Write-Host "Pruned $($Doomed.Count) ByteTrack demo files; $($Yolox.Count) yolox files kept."
    } finally {
        $Archive.Dispose()
    }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'bootstrap-windows.ps1') -Destination $Payload
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'launcher-windows.ps1') -Destination $Payload
    foreach ($RuntimeLock in @('requirements-windows-cu126.lock.txt', 'requirements-windows-cu128.lock.txt', 'requirements-windows-cpu.lock.txt')) {
        Copy-Item -LiteralPath (Join-Path $PSScriptRoot $RuntimeLock) -Destination $Payload
        if (Select-String -LiteralPath (Join-Path $Payload $RuntimeLock) -Pattern '^\.\s*$' -Quiet) {
            throw "$RuntimeLock must not install the worker project from the current directory."
        }
    }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'uv-windows-x64.lock.json') -Destination $Payload
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'chromium-windows-x64.lock.json') -Destination $Payload
    # The variant table is how the bootstrap decides what this computer needs;
    # the redistributable lock is how it decides whether elevation is required.
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'runtime-variants.json') -Destination $Payload
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'vcredist-windows-x64.lock.json') -Destination $Payload
    [ordered]@{
        version = $Version
        compute_runtimes = @('cu126', 'cu128', 'cpu')
        worker_commit = $WorkerCommit
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Payload 'bootstrap-manifest.json') -Encoding utf8

    # The destination computer must not need Visual Studio. This tiny compiled
    # dependency is built once here and carried by the small bootstrap.
    # Built as one flat argument list rather than splatted.
    #
    # Splatting into a native command inside a scriptblock does not pass the
    # arguments through: `py` received none, started an INTERACTIVE interpreter,
    # read EOF from the null stdin and exited 0 -- so the build carried on with
    # no wheel built and nothing reported. A silent success is worse than the
    # stderr failure it replaced.
    $WheelArguments = @()
    if ([IO.Path]::GetFileName($BuildPython) -in @('py', 'py.exe')) { $WheelArguments += '-3.12' }
    $WheelArguments += @('-m', 'pip', 'wheel', 'cython-bbox==0.1.5', '--no-deps', '--wheel-dir', $Wheels)
    Invoke-Native 'Building the cython-bbox Windows wheel' {
        & $BuildPython $WheelArguments
    }
    if (-not (Get-ChildItem -LiteralPath $Wheels -Filter 'cython_bbox-*.whl')) {
        throw 'The cython-bbox wheel step reported success but produced no wheel.'
    }

    $PlayerLock = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'player.lock.json') -Raw | ConvertFrom-Json
    $PlayerRepository = (Resolve-Path $PlayerRepository).Path
    $PlayerSource = Join-Path $OutputRoot 'player-source'
    $PlayerZip = Join-Path $OutputRoot 'player-source.zip'
    git -C $PlayerRepository archive --format=zip --output=$PlayerZip $PlayerLock.commit
    Assert-LastExit 'Exporting the pinned player source'
    Expand-Archive -LiteralPath $PlayerZip -DestinationPath $PlayerSource
    Push-Location $PlayerSource
    try {
        Invoke-Native 'Installing player build dependencies' { npm ci }
        Invoke-Native 'Building the pinned player' { npm run build }
        Invoke-Native 'Staging the player host' { node tools/pack-host.mjs }
    } finally { Pop-Location }
    Copy-Item -LiteralPath (Join-Path $PlayerSource 'release\host') -Destination $Payload -Recurse
    Rename-Item -LiteralPath (Join-Path $Payload 'host') -NewName 'player'

    if (-not $IsccPath) {
        # Any installed major version, not just 6.
        #
        # This looked for the literal directory `Inno Setup 6` in three places.
        # The current release is 7.x, so somebody told to "install Inno Setup"
        # installs 7, lands in `Inno Setup 7`, and the build throws
        # "Inno Setup 6 was not found" while a perfectly good compiler sits on
        # disk. 6 is preferred because every installer we have shipped was
        # built with it; a newer one is used with a warning rather than
        # refused, because refusing a working compiler to protect an untested
        # assumption is the worse failure.
        #
        # Per-user paths come first within each version: winget installs Inno
        # Setup under LocalAppData by default, which neither Program Files
        # path finds.
        # Written out rather than with a ternary: this script has to run under
        # Windows PowerShell 5.1, which has no `? :`.
        $IsccRoots = @()
        if ($env:LOCALAPPDATA) { $IsccRoots += (Join-Path $env:LOCALAPPDATA 'Programs') }
        if (${env:ProgramFiles(x86)}) { $IsccRoots += ${env:ProgramFiles(x86)} }
        if ($env:ProgramFiles) { $IsccRoots += $env:ProgramFiles }
        $IsccFound = @(
            foreach ($Root in $IsccRoots) {
                Get-ChildItem -LiteralPath $Root -Directory -Filter 'Inno Setup *' -ErrorAction SilentlyContinue |
                    ForEach-Object {
                        $Exe = Join-Path $_.FullName 'ISCC.exe'
                        if (Test-Path -LiteralPath $Exe) {
                            $Major = 0
                            if ($_.Name -match 'Inno Setup (\d+)') { $Major = [int]$Matches[1] }
                            [PSCustomObject]@{ Major = $Major; Path = $Exe }
                        }
                    }
            }
        )
        # 6 first, then the highest other major.
        $Preferred = @($IsccFound | Where-Object { $_.Major -eq 6 })
        if ($Preferred.Count -gt 0) {
            $IsccPath = $Preferred[0].Path
        } elseif ($IsccFound.Count -gt 0) {
            $Newest = @($IsccFound | Sort-Object Major -Descending)[0]
            Write-Warning ("Inno Setup 6 was not found; building with Inno Setup $($Newest.Major) at " +
                "$($Newest.Path). Every installer shipped so far was built with 6, so this " +
                'combination is unverified -- check the installer runs before releasing it.')
            $IsccPath = $Newest.Path
        }
    }
    if (-not $IsccPath) {
        throw ('Inno Setup was not found. Install it, or pass -IsccPath. The download link on ' +
            "jrsoftware.org's download.php returns an HTML page rather than a binary; the real " +
            'asset is the innosetup-*.exe on https://github.com/jrsoftware/issrc/releases. ' +
            'Silent install: /VERYSILENT /SUPPRESSMSGBOXES /CURRENTUSER /NORESTART /SP-')
    }
    & $IsccPath "/DPayloadDir=$Payload" "/DOutputDir=$InstallerDir" "/DWorkerVersion=$Version" `
        "/DWorkerRevision=$WorkerRevision" `
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
        runtimes = @('cu126', 'cu128', 'cpu')
        runtime_delivery = 'selected from the machine and downloaded during setup'
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $InstallerDir 'build-manifest.json') -Encoding utf8
    Write-Host "Built $($File.FullName) ($($File.Length) bytes)"
} finally {
    Pop-Location
}
