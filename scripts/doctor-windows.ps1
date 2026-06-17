[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "windows-common.ps1")

Assert-WindowsHost
$RepoRoot = Get-RepoRoot
Set-Location $RepoRoot

Write-Step "Running environment doctor"
$doctor = Join-Path $RepoRoot "tools/doctor/doctor.py"

if ($env:PYTHON_BIN) {
    & $env:PYTHON_BIN $doctor
}
elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python $doctor
}
elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
    & python3 $doctor
}
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3 $doctor
}
else {
    throw "Python 3 was not found. Install Python or set PYTHON_BIN."
}

Assert-LastExitCode -Context "Environment doctor"
