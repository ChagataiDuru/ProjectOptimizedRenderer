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

Set-Location $RepoRoot
Ensure-MsvcEnvironment

try {
    $ExecutablePath = Get-ExecutablePath -Preset $Preset
}
catch {
    Write-Step "Renderer executable is missing; running build"
    & (Join-Path $RepoRoot "scripts/build-windows.ps1") -BuildType $BuildType
    $ExecutablePath = Get-ExecutablePath -Preset $Preset
}

Use-ConanRunEnvironment

Write-Step "Launching $ExecutablePath"
& $ExecutablePath
Assert-LastExitCode -Context "Renderer execution"
