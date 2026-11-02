<#
.SYNOPSIS
    Create the project virtual environment and install dependencies.

.DESCRIPTION
    Builds .venv next to this script and installs requirements.txt into it.

    By default the venv inherits the parent interpreter's site-packages
    (-SystemSitePackages), so an existing torch build is reused instead of
    pulling a second multi-gigabyte copy. Drop the switch for a fully isolated
    environment.

.EXAMPLE
    .\setup.ps1
    .\setup.ps1 -Python "C:\Users\me\anaconda3\python.exe"
    .\setup.ps1 -Isolated          # do not inherit site-packages; installs its own torch
#>
[CmdletBinding()]
param(
    # Interpreter used to build the venv. Defaults to whatever `python` resolves to.
    [string]$Python = "python",

    # Fully isolate the venv (installs its own torch — large download).
    [switch]$Isolated,

    # Skip the post-install import check.
    [switch]$NoVerify
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$venv = Join-Path $root ".venv"
$venvPython = Join-Path $venv "Scripts\python.exe"

Write-Host "Project : $root"

$resolved = Get-Command $Python -ErrorAction SilentlyContinue
if (-not $resolved) {
    throw "Interpreter '$Python' not found. Pass -Python with a full path to python.exe."
}

# On Windows, a bare `python` often resolves to the Microsoft Store stub, which
# is not an interpreter — it opens the Store and exits. Catch it here rather
# than letting `python -m venv` fail with something cryptic.
if ($resolved.Source -like "*\WindowsApps\*") {
    throw @"
'$Python' resolves to the Microsoft Store stub:
    $($resolved.Source)
That is not a usable interpreter. Pass a real one, e.g.:
    .\setup.ps1 -Python "$env:USERPROFILE\anaconda3\python.exe"
"@
}

$probe = & $Python -c "import sys; print(sys.version.split()[0])" 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "'$Python' is not a working interpreter: $probe"
}
Write-Host "Python  : $probe  ($($resolved.Source))"

if (Test-Path $venvPython) {
    Write-Host "venv    : already present at .venv (reusing)"
} else {
    $venvArgs = @("-m", "venv")
    if (-not $Isolated) { $venvArgs += "--system-site-packages" }
    $venvArgs += $venv
    Write-Host "venv    : creating .venv$(if (-not $Isolated) { ' (inheriting site-packages)' })"
    & $Python @venvArgs
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}

Write-Host "`nInstalling dependencies from requirements.txt ..."
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& $venvPython -m pip install -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -ne 0) { throw "dependency install failed" }

if (-not $NoVerify) {
    Write-Host "`nVerifying the environment ..."
    & $venvPython (Join-Path $root "check_env.py")
    if ($LASTEXITCODE -ne 0) { throw "environment check failed" }
}

Write-Host @"

Done. Activate it with:

    .\.venv\Scripts\Activate.ps1

Then run a pipeline check (~5 min on CPU, writes to results/smoke/):

    python run_benchmark.py --smoke

"@
