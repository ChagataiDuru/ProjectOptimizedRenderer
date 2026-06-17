Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -ge 7) {
    $PSNativeCommandUseErrorActionPreference = $true
}

function Assert-WindowsHost {
    if (-not $IsWindows) {
        throw "This script is for Windows hosts only."
    }
}

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)

    Write-Host ""
    Write-Host "==> $Message"
}

function Assert-LastExitCode {
    param([Parameter(Mandatory = $true)][string]$Context)

    if ($LASTEXITCODE -ne 0) {
        throw "$Context failed with exit code $LASTEXITCODE."
    }
}

function Get-RepoRoot {
    return (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

function Invoke-ToolingHelper {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $repoRoot = Get-RepoRoot
    $helper = Join-Path $repoRoot "tools/common/project_tooling.py"

    if ($env:PYTHON_BIN) {
        $output = & $env:PYTHON_BIN $helper @Arguments
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        $output = & python $helper @Arguments
    }
    elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
        $output = & python3 $helper @Arguments
    }
    elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $output = & py -3 $helper @Arguments
    }
    else {
        throw "Python 3 was not found. Install Python or set PYTHON_BIN."
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Tooling helper failed for arguments: $($Arguments -join ' ')"
    }

    return (($output | Out-String).Trim())
}

function Get-NormalizedBuildType {
    param([string]$BuildType = "Debug")

    return (Invoke-ToolingHelper -Arguments @("build-type", $BuildType))
}

function Get-WindowsPreset {
    param([string]$BuildType = "Debug")

    return (Invoke-ToolingHelper -Arguments @("preset", "--platform", "windows", "--build-type", $BuildType))
}

function Get-BuildDirForPreset {
    param([Parameter(Mandatory = $true)][string]$Preset)

    return (Invoke-ToolingHelper -Arguments @("build-dir", "--preset", $Preset))
}

function Get-ExecutablePath {
    param([Parameter(Mandatory = $true)][string]$Preset)

    return (Invoke-ToolingHelper -Arguments @("executable", "--platform", "windows", "--preset", $Preset))
}

function Get-ConanToolchainPath {
    $repoRoot = Get-RepoRoot
    return (Join-Path $repoRoot "build/conan/conan_toolchain.cmake")
}

function Get-CMakeUserPresetsPath {
    $repoRoot = Get-RepoRoot
    return (Join-Path $repoRoot "CMakeUserPresets.json")
}

function Test-HasExpectedConanPresets {
    $presetsPath = Get-CMakeUserPresetsPath
    if (-not (Test-Path $presetsPath)) {
        return $false
    }

    $content = Get-Content -Path $presetsPath -Raw
    return $content.Contains("build/conan/CMakePresets.json")
}

function Get-VsWherePath {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\Installer\vswhere.exe"
    )

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) {
            return $candidate
        }
    }

    $command = Get-Command vswhere.exe -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }

    return $null
}

function Ensure-MsvcEnvironment {
    if (Get-Command cl.exe -ErrorAction SilentlyContinue) {
        return
    }

    $vswhere = Get-VsWherePath
    if (-not $vswhere) {
        throw "Could not find vswhere.exe. Install Visual Studio 2022 Build Tools or open a Developer PowerShell."
    }

    $installationPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($LASTEXITCODE -ne 0 -or -not $installationPath) {
        throw "Could not locate a Visual Studio installation with C++ tools. Install the Desktop C++ workload."
    }

    $installationPath = $installationPath.Trim()
    $repoRoot = Get-RepoRoot
    $devShellModule = Join-Path $installationPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
    $launchScript = Join-Path $installationPath "Common7\Tools\Launch-VsDevShell.ps1"

    Write-Step "Importing Visual Studio developer environment"
    if (Test-Path $devShellModule) {
        Import-Module $devShellModule -ErrorAction Stop | Out-Null
        Enter-VsDevShell -VsInstallPath $installationPath -SkipAutomaticLocation -DevCmdArguments "-arch=amd64 -host_arch=amd64" | Out-Null
    }
    elseif (Test-Path $launchScript) {
        & $launchScript -Arch amd64 -HostArch amd64 | Out-Null
        Set-Location $repoRoot
    }
    else {
        throw "Visual Studio dev shell scripts were not found under $installationPath."
    }

    if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
        throw "MSVC environment initialization did not expose cl.exe. Open a VS 2022 Developer PowerShell or repair the C++ workload."
    }
}

function Use-ConanRunEnvironment {
    $repoRoot = Get-RepoRoot
    $conanRunPs1 = Join-Path $repoRoot "build/conan/conanrun.ps1"
    $conanRunBat = Join-Path $repoRoot "build/conan/conanrun.bat"

    if (Test-Path $conanRunPs1) {
        Write-Step "Activating Conan run environment"
        . $conanRunPs1
        return
    }

    if (Test-Path $conanRunBat) {
        Write-Step "Activating Conan run environment"
        $envLines = cmd /c "call `"$conanRunBat`" >nul && set"
        Assert-LastExitCode -Context "Conan run environment activation"
        foreach ($line in $envLines) {
            if ($line -match "^(.*?)=(.*)$") {
                Set-Item -Path ("Env:{0}" -f $matches[1]) -Value $matches[2]
            }
        }
    }
}
