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
$ToolchainPath = Get-ConanToolchainPath

Set-Location $RepoRoot
Ensure-MsvcEnvironment

if (-not (Test-Path $ToolchainPath) -or -not (Test-HasExpectedConanPresets)) {
    Write-Step "Conan toolchain or presets are missing; running bootstrap"
    & (Join-Path $RepoRoot "scripts/bootstrap-windows.ps1") -BuildType $BuildType
}

Write-Step "Configuring preset '$Preset'"
cmake --preset $Preset -DCMAKE_TOOLCHAIN_FILE="$ToolchainPath"
Assert-LastExitCode -Context "CMake configure"

Write-Step "Building preset '$Preset'"
cmake --build --preset $Preset --config $BuildType
Assert-LastExitCode -Context "CMake build"

Write-Step "Build complete"
