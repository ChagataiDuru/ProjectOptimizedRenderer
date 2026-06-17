[CmdletBinding()]
param(
    [string]$BuildType = "Debug"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "windows-common.ps1")

Assert-WindowsHost
$BuildType = Get-NormalizedBuildType -BuildType $BuildType
$Preset = Get-WindowsPreset -BuildType $BuildType
$RepoRoot = Get-RepoRoot

Write-Step "Bootstrapping $BuildType environment for preset '$Preset'"
Set-Location $RepoRoot
Ensure-MsvcEnvironment

Write-Step "Syncing git submodules"
git submodule update --init --recursive
Assert-LastExitCode -Context "Git submodule sync"

Write-Step "Detecting Conan profile"
conan profile detect --force
Assert-LastExitCode -Context "Conan profile detection"

Write-Step "Installing Conan dependencies"
conan install . --output-folder=build/conan --build=missing -s "build_type=$BuildType"
Assert-LastExitCode -Context "Conan install"

Write-Step "Bootstrap complete"
Write-Host "Preset: $Preset"
