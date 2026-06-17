[CmdletBinding()]
param(
    [switch]$All
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "windows-common.ps1")

Assert-WindowsHost
$RepoRoot = Get-RepoRoot
$Targets = @(
    (Join-Path $RepoRoot "build\win-debug"),
    (Join-Path $RepoRoot "build\win-relwithdebinfo"),
    (Join-Path $RepoRoot "build\win-release"),
    (Join-Path $RepoRoot "build\conan-debug")
)

if ($All) {
    $Targets += @(
        (Join-Path $RepoRoot "build\conan"),
        (Join-Path $RepoRoot "CMakeUserPresets.json")
    )
}

Write-Step "Removing generated build outputs"
foreach ($target in $Targets) {
    if (Test-Path $target) {
        Write-Host "Removing $target"
        Remove-Item -Recurse -Force $target
    }
}

Write-Step "Clean complete"
