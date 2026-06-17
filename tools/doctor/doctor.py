#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.common.project_tooling import (  # noqa: E402
    MINIMUM_CMAKE_VERSION,
    ToolMatch,
    find_program,
    find_repo_root,
    find_shader_compiler,
    platform_key,
)


@dataclass(frozen=True)
class CheckResult:
    status: str
    label: str
    detail: str
    required: bool = False


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the local ProjectOptimizedRenderer environment.")
    parser.add_argument(
        "--strict-linux",
        action="store_true",
        help="Treat Linux host detection as a hard failure instead of a warning.",
    )
    return parser.parse_args(argv)


def run_command(args: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError:
        return 127, ""
    return completed.returncode, completed.stdout.strip()


def extract_version(text: str) -> Optional[tuple[int, ...]]:
    match = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", text)
    if not match:
        return None
    parts = [int(part) for part in match.groups(default="0")]
    return tuple(parts)


def format_version(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)


def check_command_version(
    label: str,
    command: str,
    version_args: list[str],
    minimum: Optional[tuple[int, ...]] = None,
    required: bool = True,
) -> tuple[CheckResult, Optional[ToolMatch]]:
    tool = find_program([command])
    if not tool:
        return CheckResult("FAIL", label, f"{command} was not found on PATH.", required=required), None

    code, output = run_command([str(tool.path), *version_args])
    if code != 0:
        return (
            CheckResult("FAIL", label, f"Found at {tool.path} but version check failed.", required=required),
            tool,
        )

    version = extract_version(output)
    if minimum and version and version < minimum:
        return (
            CheckResult(
                "FAIL",
                label,
                f"{format_version(version)} found at {tool.path}; need >= {format_version(minimum)}.",
                required=required,
            ),
            tool,
        )

    version_label = format_version(version) if version else output.splitlines()[0]
    return CheckResult("OK", label, f"{version_label} ({tool.path})", required=required), tool


def check_python() -> CheckResult:
    version = sys.version_info
    if version.major < 3:
        return CheckResult("FAIL", "Python 3", f"Detected Python {platform.python_version()}.", required=True)
    return CheckResult(
        "OK",
        "Python 3",
        f"{platform.python_version()} ({Path(sys.executable).resolve()})",
        required=True,
    )


def check_architecture(current_platform: str, strict_linux: bool) -> CheckResult:
    machine = platform.machine().lower()
    system_name = platform.system()

    if current_platform == "macos":
        if machine in {"arm64", "aarch64"}:
            return CheckResult("OK", "Architecture", f"Apple Silicon detected ({machine})", required=True)
        return CheckResult(
            "WARN",
            "Architecture",
            f"macOS host is {machine}; Apple Silicon is the primary supported path.",
            required=False,
        )

    if current_platform == "windows":
        if machine in {"amd64", "x86_64"}:
            return CheckResult("OK", "Architecture", f"Windows x64 detected ({machine})", required=True)
        return CheckResult(
            "WARN",
            "Architecture",
            f"Windows host is {machine}; x64 is the primary supported path.",
            required=False,
        )

    status = "FAIL" if strict_linux else "WARN"
    return CheckResult(
        status,
        "Platform",
        f"{system_name} / {machine} detected; local wrapper scripts are only provided for macOS and Windows.",
        required=strict_linux,
    )


def check_conan_profile() -> CheckResult:
    default_profile = Path.home() / ".conan2" / "profiles" / "default"
    if default_profile.is_file():
        return CheckResult("OK", "Conan default profile", str(default_profile), required=True)
    return CheckResult(
        "FAIL",
        "Conan default profile",
        f"Missing {default_profile}. Run 'conan profile detect --force'.",
        required=True,
    )


def check_build_dir(repo_root: Path) -> CheckResult:
    build_dir = repo_root / "build"
    if build_dir.exists():
        if build_dir.is_dir() and os.access(build_dir, os.W_OK):
            return CheckResult("OK", "Build directory", f"{build_dir} is writable", required=True)
        return CheckResult("FAIL", "Build directory", f"{build_dir} is not writable", required=True)

    if os.access(repo_root, os.W_OK):
        return CheckResult("OK", "Build directory", f"{build_dir} can be created", required=True)
    return CheckResult(
        "FAIL",
        "Build directory",
        f"{build_dir} does not exist and {repo_root} is not writable.",
        required=True,
    )


def check_vma_submodule(repo_root: Path) -> CheckResult:
    vma_header = repo_root / "external" / "VulkanMemoryAllocator" / "include" / "vk_mem_alloc.h"
    if vma_header.is_file():
        return CheckResult("OK", "VMA submodule", str(vma_header), required=True)
    return CheckResult(
        "FAIL",
        "VMA submodule",
        f"Missing {vma_header}. Run 'git submodule update --init --recursive'.",
        required=True,
    )


def find_first_existing(paths: list[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def check_vulkan_runtime(current_platform: str) -> CheckResult:
    vulkan_sdk = os.environ.get("VULKAN_SDK")

    if current_platform == "macos":
        candidates = []
        if vulkan_sdk:
            sdk_root = Path(vulkan_sdk)
            candidates.extend(
                [
                    sdk_root / "lib" / "libMoltenVK.dylib",
                    sdk_root / "share" / "vulkan" / "icd.d" / "MoltenVK_icd.json",
                ]
            )
        candidates.extend(
            [
                Path("/opt/homebrew/lib/libMoltenVK.dylib"),
                Path("/opt/homebrew/opt/molten-vk/lib/libMoltenVK.dylib"),
                Path("/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json"),
                Path("/usr/local/lib/libMoltenVK.dylib"),
                Path("/usr/local/share/vulkan/icd.d/MoltenVK_icd.json"),
            ]
        )
        found = find_first_existing(candidates)
        if found:
            return CheckResult("OK", "Vulkan runtime", f"MoltenVK found at {found}", required=True)
        return CheckResult(
            "FAIL",
            "Vulkan runtime",
            "MoltenVK was not found. Install Homebrew MoltenVK or the LunarG Vulkan SDK.",
            required=True,
        )

    if current_platform == "windows":
        candidates = []
        if vulkan_sdk:
            candidates.append(Path(vulkan_sdk))
        sdk_root = Path("C:/VulkanSDK")
        if sdk_root.is_dir():
            candidates.extend(sorted((path for path in sdk_root.iterdir() if path.is_dir()), reverse=True))
        found = find_first_existing(candidates)
        if found:
            return CheckResult("OK", "Vulkan runtime", f"Vulkan SDK found at {found}", required=True)
        return CheckResult(
            "FAIL",
            "Vulkan runtime",
            "Vulkan SDK was not found. Install the LunarG Vulkan SDK or set VULKAN_SDK.",
            required=True,
        )

    vulkaninfo = find_program(["vulkaninfo"])
    if vulkan_sdk:
        return CheckResult("OK", "Vulkan runtime", f"VULKAN_SDK={vulkan_sdk}", required=True)
    if vulkaninfo:
        return CheckResult("OK", "Vulkan runtime", f"vulkaninfo found at {vulkaninfo.path}", required=True)
    return CheckResult(
        "WARN",
        "Vulkan runtime",
        "No Vulkan SDK/runtime probe succeeded on Linux; install Vulkan tooling if you intend to build here.",
        required=False,
    )


def check_shader_compiler(repo_root: Path) -> tuple[CheckResult, Optional[ToolMatch]]:
    compiler = find_shader_compiler(repo_root, "auto")
    if compiler:
        return CheckResult("OK", "Shader compiler", f"{compiler.name} ({compiler.path})", required=True), compiler
    return (
        CheckResult(
            "FAIL",
            "Shader compiler",
            "Neither glslc nor glslangValidator was found. Install glslang or the Vulkan SDK.",
            required=True,
        ),
        None,
    )


def check_sponza(repo_root: Path) -> CheckResult:
    sponza = repo_root / "assets" / "source" / "Sponza.gltf"
    if sponza.is_file():
        return CheckResult("OK", "Default Sponza asset", str(sponza), required=True)
    return CheckResult("FAIL", "Default Sponza asset", f"Missing {sponza}", required=True)


def check_hdr_assets(repo_root: Path) -> CheckResult:
    hdr_files = sorted(repo_root.glob("assets/**/*.hdr"))
    if hdr_files:
        return CheckResult("OK", "HDR panorama assets", f"{len(hdr_files)} found under assets/", required=False)
    return CheckResult(
        "WARN",
        "HDR panorama assets",
        "No .hdr files were found under assets/. HDR sky loading will need an external panorama.",
        required=False,
    )


def print_result(result: CheckResult) -> None:
    print(f"[{result.status}] {result.label}: {result.detail}")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = find_repo_root()
    current_platform = platform_key()

    print("ProjectOptimizedRenderer Environment Check")
    print()

    results: list[CheckResult] = []
    discovered_paths: dict[str, Path] = {}

    cmake_result, cmake_tool = check_command_version(
        "CMake",
        "cmake",
        ["--version"],
        minimum=MINIMUM_CMAKE_VERSION,
        required=True,
    )
    results.append(cmake_result)
    if cmake_tool:
        discovered_paths["cmake"] = cmake_tool.path

    conan_result, conan_tool = check_command_version(
        "Conan 2",
        "conan",
        ["--version"],
        minimum=(2, 0, 0),
        required=True,
    )
    results.append(conan_result)
    if conan_tool:
        discovered_paths["conan"] = conan_tool.path

    ninja_result, ninja_tool = check_command_version(
        "Ninja",
        "ninja",
        ["--version"],
        required=True,
    )
    results.append(ninja_result)
    if ninja_tool:
        discovered_paths["ninja"] = ninja_tool.path

    results.append(check_python())
    results.append(check_architecture(current_platform, args.strict_linux))
    results.append(check_conan_profile())
    results.append(check_build_dir(repo_root))
    results.append(check_vma_submodule(repo_root))

    vulkan_result = check_vulkan_runtime(current_platform)
    results.append(vulkan_result)

    shader_result, shader_tool = check_shader_compiler(repo_root)
    results.append(shader_result)
    if shader_tool:
        discovered_paths["shader_compiler"] = shader_tool.path

    results.append(check_sponza(repo_root))
    results.append(check_hdr_assets(repo_root))

    env_vulkan_sdk = os.environ.get("VULKAN_SDK")
    if env_vulkan_sdk:
        results.append(CheckResult("OK", "Environment", f"VULKAN_SDK={env_vulkan_sdk}", required=False))
    else:
        results.append(
            CheckResult(
                "WARN",
                "Environment",
                "VULKAN_SDK is not set. That is fine if your platform-specific runtime is discoverable.",
                required=False,
            )
        )

    for result in results:
        print_result(result)

    if discovered_paths:
        print()
        print("Resolved tool paths:")
        for label, path in sorted(discovered_paths.items()):
            print(f"  {label}: {path}")

    failures = [result for result in results if result.status == "FAIL" and result.required]
    warnings = [result for result in results if result.status == "WARN"]

    print()
    if failures:
        print(f"Environment not ready: {len(failures)} required check(s) failed.")
        if warnings:
            print(f"Warnings: {len(warnings)}")
        return 1

    print("Environment ready.")
    if warnings:
        print(f"Warnings: {len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
