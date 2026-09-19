# test-select-variant.ps1
# Exercises the runtime-selection rule against the cases that matter.
#
# Selection is the part of the bootstrap most likely to be wrong and the part
# whose failures are quietest: picking a runtime the driver cannot serve
# produces a worker that installs cleanly and never sees its GPU, and refusing
# a machine that could have worked loses a volunteer silently. The rule is
# tested here against real hardware where this machine can supply it, and
# against fabricated readings where it cannot -- nobody in the fleet has a
# GPU old enough to test the Maxwell path, and no machine can have no GPU on
# demand.
#
# Run: powershell -NoProfile -ExecutionPolicy Bypass -File packaging\test-select-variant.ps1

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Table = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'runtime-variants.json') -Raw | ConvertFrom-Json

# The rule under test, lifted verbatim in behaviour from bootstrap-windows.ps1's
# Select-Variant. It takes the three machine readings as arguments so a case can
# describe hardware this computer does not have.
function Select-VariantFor([string]$CapabilityText, [string]$DriverText, [bool]$HasNvidiaSmi) {

    $Variants = $Table.variants
    $Cpu = $Variants | Where-Object { $_.name -eq 'cpu' } | Select-Object -First 1

    if (-not $HasNvidiaSmi) { return $Cpu.name }
    if (-not $CapabilityText -or -not $DriverText) { return $Cpu.name }

    $Capability = 0.0
    $Driver = 0.0
    $Invariant = [Globalization.CultureInfo]::InvariantCulture
    $Number = [Globalization.NumberStyles]::Number
    [void][double]::TryParse($CapabilityText, $Number, $Invariant, [ref]$Capability)
    [void][double]::TryParse($DriverText, $Number, $Invariant, [ref]$Driver)

    foreach ($Variant in $Variants) {
        if ($Variant.name -eq 'cpu') { continue }
        if ($Capability -lt [double]$Variant.minimum_compute_capability) { continue }
        if ($Variant.PSObject.Properties.Name -contains 'maximum_compute_capability' -and
            $Capability -gt [double]$Variant.maximum_compute_capability) { continue }
        if ($Driver -lt [double]$Variant.minimum_driver_version) { continue }
        return $Variant.name
    }
    return $Cpu.name
}

$Cases = @(
    # Every machine in the fleet, with the readings they actually report.
    @{ Name = 'VP1 GTX 1660 (real)';            Cap = '7.5';  Driver = '610.60'; Smi = $true;  Expect = 'cu128' }
    @{ Name = 'ABYSS RTX 4080 SUPER';           Cap = '8.9';  Driver = '591.86'; Smi = $true;  Expect = 'cu128' }
    @{ Name = 'laptop RTX 5060 Blackwell';      Cap = '12.0'; Driver = '573.13'; Smi = $true;  Expect = 'cu128' }
    @{ Name = 'ubuntu GTX 1660 Ti';             Cap = '7.5';  Driver = '580.00'; Smi = $true;  Expect = 'cu128' }

    # The volunteer hardware Isaac insists we serve. A GTX 1060 is Pascal,
    # sm_6.1, which only cu126 reaches -- this is the case that fails if anyone
    # ever "simplifies" the variant list.
    @{ Name = 'volunteer GTX 1060 (Pascal)';    Cap = '6.1';  Driver = '550.00'; Smi = $true;  Expect = 'cu126' }
    @{ Name = 'volunteer GTX 960 (Maxwell)';    Cap = '5.0';  Driver = '550.00'; Smi = $true;  Expect = 'cu126' }

    # No GPU at all. Isaac: "if there is no gpu we should run on a CPU".
    @{ Name = 'no NVIDIA driver present';       Cap = '';     Driver = '';       Smi = $false; Expect = 'cpu' }
    @{ Name = 'nvidia-smi answers nothing';     Cap = '';     Driver = '520.00'; Smi = $true;  Expect = 'cpu' }

    # A driver too old for anything we ship falls back rather than installing a
    # runtime that cannot see the card.
    @{ Name = 'Turing on an ancient driver';    Cap = '7.5';  Driver = '450.00'; Smi = $true;  Expect = 'cpu' }
    @{ Name = 'Blackwell on an ancient driver'; Cap = '12.0'; Driver = '450.00'; Smi = $true;  Expect = 'cpu' }

    # Blackwell must never be handed cu126: its arch list stops at sm_90, so it
    # would install cleanly and fail to use the card.
    @{ Name = 'Blackwell never gets cu126';     Cap = '12.0'; Driver = '600.00'; Smi = $true;  Expect = 'cu128' }
)

$Failures = 0
foreach ($Case in $Cases) {
    $Actual = Select-VariantFor $Case.Cap $Case.Driver $Case.Smi
    if ($Actual -eq $Case.Expect) {
        "  PASS  {0,-34} -> {1}" -f $Case.Name, $Actual
    } else {
        "  FAIL  {0,-34} -> {1}, expected {2}" -f $Case.Name, $Actual, $Case.Expect
        $Failures += 1
    }
}

Write-Host ''
if ($Failures) {
    Write-Host "$Failures of $($Cases.Count) selection cases failed." -ForegroundColor Red
    exit 1
}
Write-Host "All $($Cases.Count) selection cases passed." -ForegroundColor Green
