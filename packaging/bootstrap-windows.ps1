[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$PayloadRoot,
    [Parameter(Mandatory)][string]$InstallRoot,
    [string]$CoordinatorUrl,
    [string]$ActivationCodeFile,
    [switch]$CheckOnly
)

# bootstrap-windows.ps1
# Installs the MARP Inference Worker on a volunteer's computer.
#
# The installer is small and fetches what this particular machine needs, rather
# than carrying every runtime for every machine. What it fetches is decided
# here, from what the computer actually has: whether there is an NVIDIA GPU,
# what it can do, and whether the driver is new enough for the CUDA build that
# would serve it. A machine with no usable GPU installs the CPU runtime and
# works slowly rather than being refused (#38).
#
# Everything it downloads is pinned and hash-verified. Nothing asks the
# volunteer a question, and nothing requires Python, a browser, a CUDA toolkit
# or Visual Studio to be installed beforehand.

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

function Read-Payload([string]$Name) {
    Get-Content -LiteralPath (Join-Path $PayloadRoot $Name) -Raw | ConvertFrom-Json
}

function Get-Sha256([string]$Path) {
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-VerifiedArchive($Lock, [string]$Destination) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    Invoke-WebRequest -Uri $Lock.url -OutFile $Destination -UseBasicParsing
    $File = Get-Item -LiteralPath $Destination
    if ($File.Length -ne [int64]$Lock.size_bytes) { throw "$($Lock.name) download has the wrong size." }
    if ((Get-Sha256 $Destination) -ne $Lock.sha256) { throw "$($Lock.name) download failed SHA-256 verification." }
}

function Stop-InstalledProcesses {
    $InstallPrefix = [System.IO.Path]::GetFullPath($InstallRoot).TrimEnd('\') + '\'
    $Processes = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ExecutablePath -and $_.ExecutablePath.StartsWith($InstallPrefix, [System.StringComparison]::OrdinalIgnoreCase)
    }
    foreach ($Process in $Processes) {
        Stop-Process -Id $Process.ProcessId -Force -ErrorAction SilentlyContinue
    }
    if ($Processes) {
        Write-Host 'Stopped the installed MARP worker so it can be updated.'
        Start-Sleep -Milliseconds 500
    }
}

$BootstrapManifest = Read-Payload 'bootstrap-manifest.json'
$RuntimeTable = Read-Payload 'runtime-variants.json'

# Chooses the runtime from what this computer has.
#
# Three facts, read from the machine: is there an NVIDIA GPU, what capability
# does it report, and is the driver new enough for the CUDA build that would
# serve it. Anything failing all the CUDA variants gets the CPU one rather than
# a refusal -- a slow worker is worth more than no worker, and a volunteer who
# cannot install has no way to find out why.
function Select-Variant {

    $Variants = $RuntimeTable.variants
    $Cpu = $Variants | Where-Object { $_.name -eq 'cpu' } | Select-Object -First 1

    if (-not (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)) {
        return @{ Variant = $Cpu; Reason = 'no NVIDIA driver is present' }
    }

    $CapabilityText = (& nvidia-smi.exe --query-gpu=compute_cap --format=csv,noheader,nounits | Select-Object -First 1)
    $DriverText = (& nvidia-smi.exe --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1)
    if (-not $CapabilityText -or -not $DriverText) {
        return @{ Variant = $Cpu; Reason = 'the GPU did not report a usable capability' }
    }
    $CapabilityText = $CapabilityText.Trim()
    $DriverText = $DriverText.Trim()

    $Capability = 0.0
    $Driver = 0.0
    $Invariant = [Globalization.CultureInfo]::InvariantCulture
    $Number = [Globalization.NumberStyles]::Number
    [void][double]::TryParse($CapabilityText, $Number, $Invariant, [ref]$Capability)
    [void][double]::TryParse($DriverText, $Number, $Invariant, [ref]$Driver)
    Write-Host "This computer reports compute capability $CapabilityText and driver $DriverText."

    foreach ($Variant in $Variants) {
        if ($Variant.name -eq 'cpu') { continue }

        # A range rather than a membership test, deliberately. CUDA
        # minor-version compatibility runs a binary on later minor versions of
        # the same major family: an RTX 4080 SUPER reports 8.9, which is in no
        # arch list we ship, and runs on sm_86 kernels. Testing membership would
        # refuse a card that has been working for weeks.
        if ($Capability -lt [double]$Variant.minimum_compute_capability) { continue }
        if ($Variant.PSObject.Properties.Name -contains 'maximum_compute_capability' -and
            $Capability -gt [double]$Variant.maximum_compute_capability) { continue }

        # The driver check exists because of a failure that otherwise looks like
        # success: CUDA wheels newer than the driver install perfectly and then
        # cannot see the card at all, with cudaGetDeviceCount() returning
        # cudaErrorNotSupported. The worker reports healthy and never uses the
        # GPU. Recorded in .marp/handoff.md for a CUDA 13 runtime on a 573.13
        # driver.
        if ($Driver -lt [double]$Variant.minimum_driver_version) {
            Write-Host ("  $($Variant.name) wants driver $($Variant.minimum_driver_version) or newer.")
            continue
        }

        return @{ Variant = $Variant; Reason = "compute capability $CapabilityText, driver $DriverText" }
    }

    Write-Host ''
    Write-Host 'MARP cannot use this GPU, so work will run on the processor instead.' -ForegroundColor Yellow
    Write-Host 'Updating the NVIDIA driver from nvidia.com may let MARP use the GPU.' -ForegroundColor Yellow
    return @{ Variant = $Cpu; Reason = 'no CUDA runtime matches this GPU and driver' }
}

# -CheckOnly answers "will this work on my computer" in a couple of seconds.
#
# The alternative is a volunteer finding out twenty minutes and several
# gigabytes in, which is the position the Linux side fixed first. It reports
# what was found and what would be installed, then exits without touching
# anything -- no download, no install, no enrolment.
if ($CheckOnly) {
    Write-Host ''
    Write-Host 'Checking whether this computer can run MARP...' -ForegroundColor Cyan
    $Selection = Select-Variant
    $Variant = $Selection.Variant
    Write-Host ''
    Write-Host "MARP would install the $($Variant.name) runtime, because $($Selection.Reason)." -ForegroundColor Green
    if ($Variant.name -eq 'cpu') {
        Write-Host 'That runs on the processor: much slower than a graphics card, and still useful.'
        Write-Host 'A newer NVIDIA driver from nvidia.com may let MARP use a GPU instead.'
    } else {
        Write-Host "GPUs this runtime serves: $($Variant.architectures)"
        Write-Host "Oldest NVIDIA driver it needs: $($Variant.minimum_driver_version)"
    }

    # The runtime the volunteer already has decides whether setup will ask for
    # administrator approval, and that is worth knowing before they start.
    $Installed = Join-Path $env:SystemRoot 'System32\msvcp140.dll'
    $Required = [version](Read-Payload 'vcredist-windows-x64.lock.json').file_version
    $Present = $null
    if (Test-Path -LiteralPath $Installed) {
        $Present = [version](Get-Item -LiteralPath $Installed).VersionInfo.FileVersion
    }
    if ($Present -and $Present -ge $Required) {
        Write-Host "Visual C++ runtime $Present is present; setup will not need administrator approval."
    } else {
        $Found = if ($Present) { "$Present" } else { 'none' }
        Write-Host "Visual C++ runtime found: $Found. Setup will install $Required and ask for administrator approval once." -ForegroundColor Yellow
    }

    Write-Host ''
    Write-Host 'Nothing was installed or downloaded. Run setup normally to install.'
    Stop-Transcript | Out-Null
    exit 0
}

if (-not $CoordinatorUrl) { throw 'A coordinator address is required to install.' }
if (-not $ActivationCodeFile) { throw 'An activation code file is required to install.' }

Write-Stage 1 'Looking at what this computer has...'
$Selection = Select-Variant
$Variant = $Selection.Variant
if (-not $Variant) { throw 'No runtime could be selected for this computer.' }
$ComputeRuntime = $Variant.name
$RequirementsLock = $Variant.lock
Write-Host "Selected the $ComputeRuntime runtime, because $($Selection.Reason)."
if ($ComputeRuntime -eq 'cpu') {
    Write-Host 'MARP will run on the processor. This is much slower than a GPU but still useful.'
}

$ReleaseKey = "$($BootstrapManifest.version)-$ComputeRuntime"
$VersionRoot = Join-Path $InstallRoot "versions\$ReleaseKey"

Stop-InstalledProcesses
if (Test-Path -LiteralPath $VersionRoot) { Remove-Item -LiteralPath $VersionRoot -Recurse -Force }
New-Item -ItemType Directory -Force -Path $VersionRoot, $StateRoot, $DownloadRoot | Out-Null

# The Visual C++ runtime, before anything imports torch, and only when what is
# already installed is too old.
#
# **Load-bearing rather than hygiene.** torch's DLLs link the MSVC runtime and
# will not load against an old one: a machine here carried 14.28 from 2020 and
# `import torch` failed outright with OSError WinError 1114 naming c10.dll. The
# error points at a torch DLL and says nothing about the runtime, so a volunteer
# hitting it reports a broken package.
#
# Installing it needs administrator rights, and nothing else in this installer
# does -- so it is checked first and elevation is asked for only when it is
# genuinely needed. Most current machines skip this in silence.
Write-Stage 2 'Checking the Microsoft Visual C++ runtime...'
$RedistributableLock = Read-Payload 'vcredist-windows-x64.lock.json'
$InstalledRuntime = Join-Path $env:SystemRoot 'System32\msvcp140.dll'
$RequiredVersion = [version]$RedistributableLock.file_version
$InstalledVersion = $null
if (Test-Path -LiteralPath $InstalledRuntime) {
    $InstalledVersion = [version](Get-Item -LiteralPath $InstalledRuntime).VersionInfo.FileVersion
}
if ($InstalledVersion -and $InstalledVersion -ge $RequiredVersion) {
    Write-Host "Visual C++ runtime $InstalledVersion is already present."
} else {
    $Found = if ($InstalledVersion) { "$InstalledVersion" } else { 'none' }
    Write-Host "Visual C++ runtime found: $Found; installing $RequiredVersion (this asks for administrator approval)."
    $Redistributable = Join-Path $DownloadRoot 'vc_redist.x64.exe'
    Get-VerifiedArchive $RedistributableLock $Redistributable
    $Redistribution = Start-Process -FilePath $Redistributable `
        -ArgumentList '/install', '/quiet', '/norestart' -Verb RunAs -Wait -PassThru
    # 0 installed, 1638 a newer one is already there, 3010 installed and wants a
    # reboot. None is a failure; everything else is.
    if ($Redistribution.ExitCode -notin @(0, 1638, 3010)) {
        throw ("The Visual C++ runtime failed to install (exit code $($Redistribution.ExitCode)). " +
            'MARP cannot run without it.')
    }
}

Write-Stage 3 'Downloading the setup tools...'
$UvLock = Read-Payload 'uv-windows-x64.lock.json'
$UvArchive = Join-Path $DownloadRoot 'uv.zip'
Get-VerifiedArchive $UvLock $UvArchive
$UvRoot = Join-Path $DownloadRoot 'uv'
if (Test-Path -LiteralPath $UvRoot) { Remove-Item -LiteralPath $UvRoot -Recurse -Force }
Expand-Archive -LiteralPath $UvArchive -DestinationPath $UvRoot
$Uv = Join-Path $UvRoot 'uv.exe'

$env:UV_PYTHON_INSTALL_DIR = Join-Path $InstallRoot 'python'
$env:UV_CACHE_DIR = Join-Path $StateRoot 'uv-cache'
Write-Stage 4 'Installing Python...'
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
& $ManagedPython -m venv --without-pip $Runtime
if ($LASTEXITCODE -ne 0) { throw 'Worker runtime creation failed.' }
$Python = Join-Path $Runtime 'Scripts\python.exe'

Write-Stage 5 "Installing the $ComputeRuntime runtime. This is the longest step..."
# cython-bbox has no Windows wheel on PyPI and needs a C compiler to build,
# which a volunteer's machine must never need. It was compiled at build time and
# travels in the payload.
$Wheel = Get-ChildItem -LiteralPath (Join-Path $PayloadRoot 'wheels') -Filter 'cython_bbox-*.whl' | Select-Object -First 1
if (-not $Wheel) { throw 'The prebuilt ByteTrack dependency is missing.' }
& $Uv pip install --python $Python --no-deps $Wheel.FullName
if ($LASTEXITCODE -ne 0) { throw 'ByteTrack dependency installation failed.' }
# --torch-backend, never --index-url: the latter replaces PyPI outright, so
# the PyTorch index becomes the only source and every ordinary dependency
# vanishes. A real install died on "altgraph was not found in the package
# registry" for exactly that reason.
& $Uv pip sync --python $Python (Join-Path $PayloadRoot $RequirementsLock) --torch-backend $Variant.torch_backend
if ($LASTEXITCODE -ne 0) { throw "Installing the $ComputeRuntime runtime failed." }

$SourceRoot = Join-Path $VersionRoot 'source'
Write-Stage 6 'Installing the MARP worker and video display...'
Expand-Archive -LiteralPath (Join-Path $PayloadRoot 'worker-source.zip') -DestinationPath $SourceRoot
& $Uv pip install --python $Python --no-deps --editable $SourceRoot
if ($LASTEXITCODE -ne 0) { throw 'MARP worker installation failed.' }

Copy-Item -LiteralPath (Join-Path $PayloadRoot 'player') -Destination $VersionRoot -Recurse -Force
$ChromeLock = Read-Payload 'chromium-windows-x64.lock.json'
$ChromeArchive = Join-Path $DownloadRoot 'chromium.zip'
Get-VerifiedArchive $ChromeLock $ChromeArchive
$ChromeExtract = Join-Path $DownloadRoot 'chromium'
if (Test-Path -LiteralPath $ChromeExtract) { Remove-Item -LiteralPath $ChromeExtract -Recurse -Force }
Expand-Archive -LiteralPath $ChromeArchive -DestinationPath $ChromeExtract
$ChromeDestination = New-Item -ItemType Directory -Force -Path (Join-Path $VersionRoot 'chromium')
Copy-Item -Path (Join-Path $ChromeExtract 'chrome-win\*') -Destination $ChromeDestination -Recurse -Force
Remove-Item -LiteralPath $ChromeExtract -Recurse -Force

# Chromium's subprocesses use Windows AppContainers. Grant those built-in
# identities read and execute on the packaged browser only.
foreach ($Sid in @('*S-1-15-2-1', '*S-1-15-2-2')) {
    & icacls.exe $ChromeDestination /grant:r "${Sid}:(OI)(CI)RX" /T /C | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Failed to grant the Chromium sandbox access to the video display.' }
}

@{
    version = $BootstrapManifest.version
    platform = 'windows'
    architecture = 'x86_64'
    compute_runtime = $ComputeRuntime
    torch_version = $RuntimeTable.torch_version
    entrypoint = 'runtime/Scripts/marp-worker.exe'
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $VersionRoot 'manifest.json') -Encoding utf8

Copy-Item -LiteralPath (Join-Path $PayloadRoot 'launcher-windows.ps1') -Destination (Join-Path $InstallRoot 'launcher.ps1') -Force
@{ coordinator_url = $CoordinatorUrl.TrimEnd('/'); screen = 'window'; api_port = 8010 } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $InstallRoot 'config.json') -Encoding utf8
@{ release = $ReleaseKey } | ConvertTo-Json |
    Set-Content -LiteralPath (Join-Path $InstallRoot 'active.json') -Encoding utf8

# Both halves of the worker must agree where the credential lives.
#
# `marp-worker-activate` defaults to the per-user application directory while
# the job loop defaults to `data/worker` relative to its working directory. An
# installer setting neither activates successfully, writes the credential where
# the worker will not look, and then fails with "this worker is not activated"
# -- which reads as a credential problem, is a path problem, and consumes the
# volunteer's single-use enrolment on the way past.
$env:MARP_COORDINATOR_URL = $CoordinatorUrl
$env:MARP_WORKER_STATE_DIR = $StateRoot

Write-Stage 7 'Connecting this computer to MARP...'
$CredentialPath = Join-Path $StateRoot 'worker-credential.dpapi'
$IdentityPath = Join-Path $StateRoot 'worker-identity.json'
if ((Test-Path -LiteralPath $CredentialPath) -and (Test-Path -LiteralPath $IdentityPath)) {
    Write-Host 'This computer is already enrolled; reusing its protected worker credential.'
    Remove-Item -LiteralPath $ActivationCodeFile -Force -ErrorAction SilentlyContinue
} else {
    & $Python -m marp_inference_worker.worker_main --activate-code-file $ActivationCodeFile `
        --coordinator-url $CoordinatorUrl --state-dir $StateRoot
    if ($LASTEXITCODE -ne 0) { throw 'Worker activation failed.' }
}

Write-Stage 8 'Starting the worker and checking it can run MARP work...'
$Launcher = Join-Path $InstallRoot 'launcher.ps1'
$LauncherArguments = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Launcher`" -Screen window"
$StartupOut = Join-Path $StateRoot 'worker-startup.stdout.log'
$StartupError = Join-Path $StateRoot 'worker-startup.stderr.log'
$WorkerProcess = Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" `
    -ArgumentList $LauncherArguments -PassThru `
    -RedirectStandardOutput $StartupOut -RedirectStandardError $StartupError -WindowStyle Hidden
$Healthy = $false
for ($Attempt = 0; $Attempt -lt 60; $Attempt += 1) {
    if ($WorkerProcess.HasExited) { break }
    try {
        $Health = Invoke-RestMethod -Uri 'http://127.0.0.1:8010/health' -TimeoutSec 2 -ErrorAction SilentlyContinue
        $Status = Invoke-RestMethod -Uri 'http://127.0.0.1:8010/status' -TimeoutSec 2 -ErrorAction SilentlyContinue
        if ($Health.status -eq 'ok' -and $Status.capabilities -and $Status.enrolled) { $Healthy = $true; break }
    } catch { }
    Start-Sleep -Seconds 1
}
if (-not $Healthy) {
    if (Test-Path -LiteralPath $StartupError) {
        Write-Host 'Worker startup error:' -ForegroundColor Red
        Get-Content -LiteralPath $StartupError | Write-Host
    }
    if (Get-Process -Id $WorkerProcess.Id -ErrorAction SilentlyContinue) {
        Stop-Process -Id $WorkerProcess.Id -Force -ErrorAction SilentlyContinue
    }
    throw 'The worker installed but did not report healthy enrollment.'
}

# CUDA being visible is weaker than CUDA being usable. A wheel built without
# this GPU's kernels reports a healthy device and fails on the first job, so a
# real kernel is launched and synchronized before the machine is called ready.
# The CPU variant has nothing to prove here and skips it.
if ($ComputeRuntime -ne 'cpu') {
    & $Python -c "import torch; x = torch.ones(1, device='cuda'); assert x.item() == 1; torch.cuda.synchronize()"
    if ($LASTEXITCODE -ne 0) {
        throw "$ComputeRuntime installed, but this computer cannot execute CUDA work with it."
    }
}

Remove-Item -LiteralPath $UvArchive, $ChromeArchive -Force -ErrorAction SilentlyContinue
Write-Host ''
Write-Host "MARP Inference Worker is installed and ready ($ComputeRuntime)." -ForegroundColor Green
Stop-Transcript | Out-Null
