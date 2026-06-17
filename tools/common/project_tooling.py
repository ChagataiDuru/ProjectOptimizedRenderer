#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

PROJECT_NAME = "ProjectOptimizedRenderer"
EXECUTABLE_BASENAME = PROJECT_NAME
SUPPORTED_SHADER_EXTENSIONS = {
    ".vert": "vertex",
    ".frag": "fragment",
    ".comp": "compute",
}
MINIMUM_CMAKE_VERSION = (3, 25, 0)

PRIMARY_PRESET_BY_PLATFORM = {
    "linux": "linux-debug",
    "macos": "debug",
    "windows": "win-debug",
}

PRESET_BY_PLATFORM_AND_BUILD_TYPE = {
    "linux": {
        "Debug": "linux-debug",
    },
    "macos": {
        "Debug": "debug",
        "RelWithDebInfo": "relwithdebinfo",
        "Release": "release",
    },
    "windows": {
        "Debug": "win-debug",
        "RelWithDebInfo": "win-relwithdebinfo",
        "Release": "win-release",
    },
}


@dataclass(frozen=True)
class ToolMatch:
    name: str
    path: Path
    source: str


def find_repo_root(start: Optional[Path] = None) -> Path:
    origin = (start or Path(__file__)).resolve()
    for candidate in [origin, *origin.parents]:
        if (candidate / "CMakePresets.json").is_file() and (candidate / "conanfile.txt").is_file():
            return candidate
    raise RuntimeError(f"Could not locate repository root from {origin}")


def platform_key(system_name: Optional[str] = None) -> str:
    system_name = (system_name or platform.system()).lower()
    if system_name.startswith("darwin"):
        return "macos"
    if system_name.startswith("windows"):
        return "windows"
    if system_name.startswith("linux"):
        return "linux"
    return system_name


def executable_name(target_platform: Optional[str] = None) -> str:
    return EXECUTABLE_BASENAME + (".exe" if (target_platform or platform_key()) == "windows" else "")


def normalize_build_type(value: Optional[str]) -> str:
    if not value:
        return "Debug"

    lowered = value.strip().lower()
    aliases = {
        "debug": "Debug",
        "release": "Release",
        "relwithdebinfo": "RelWithDebInfo",
        "reldeb": "RelWithDebInfo",
        "rel-with-deb-info": "RelWithDebInfo",
    }
    if lowered not in aliases:
        supported = ", ".join(["Debug", "RelWithDebInfo", "Release"])
        raise ValueError(f"Unsupported build type '{value}'. Expected one of: {supported}")
    return aliases[lowered]


def default_preset(target_platform: Optional[str] = None) -> str:
    key = target_platform or platform_key()
    if key not in PRIMARY_PRESET_BY_PLATFORM:
        raise ValueError(f"No default preset is defined for platform '{key}'")
    return PRIMARY_PRESET_BY_PLATFORM[key]


def preset_for_build_type(build_type: str, target_platform: Optional[str] = None) -> str:
    normalized = normalize_build_type(build_type)
    key = target_platform or platform_key()
    presets = PRESET_BY_PLATFORM_AND_BUILD_TYPE.get(key, {})
    if normalized not in presets:
        supported = ", ".join(sorted(presets)) or "<none>"
        raise ValueError(
            f"Build type '{normalized}' is not supported for platform '{key}'. "
            f"Supported build types: {supported}"
        )
    return presets[normalized]


def build_dir_for_preset(repo_root: Path, preset: str) -> Path:
    return repo_root / "build" / preset


def resolve_build_dir(repo_root: Path, preset: Optional[str], build_dir: Optional[Path]) -> Path:
    if build_dir:
        return build_dir.resolve()
    resolved_preset = preset or default_preset()
    return build_dir_for_preset(repo_root, resolved_preset)


def candidate_executable_paths(
    repo_root: Path,
    preset: Optional[str],
    build_dir: Optional[Path] = None,
    target_platform: Optional[str] = None,
) -> list[Path]:
    platform_name = target_platform or platform_key()
    resolved_build_dir = resolve_build_dir(repo_root, preset, build_dir)
    exe_name = executable_name(platform_name)
    return [
        resolved_build_dir / exe_name,
        resolved_build_dir / "Debug" / exe_name,
        resolved_build_dir / "Release" / exe_name,
        repo_root / exe_name,
    ]


def find_executable(
    repo_root: Path,
    preset: Optional[str],
    build_dir: Optional[Path] = None,
    target_platform: Optional[str] = None,
) -> Optional[Path]:
    for candidate in candidate_executable_paths(repo_root, preset, build_dir, target_platform):
        if candidate.is_file():
            return candidate
    return None


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _iter_windows_sdk_bin_dirs() -> Iterable[Path]:
    sdk_root = Path("C:/VulkanSDK")
    if not sdk_root.is_dir():
        return []
    return sorted((path / "Bin" for path in sdk_root.iterdir() if path.is_dir()), reverse=True)


def _iter_conan_glslang_hints(repo_root: Path) -> Iterable[Path]:
    pattern = re.compile(r'set\(glslang_PACKAGE_FOLDER_[A-Z0-9_]+\s+"([^"]+)"\)')
    build_root = repo_root / "build"
    if not build_root.is_dir():
        return []

    matches: list[Path] = []
    for cmake_data in build_root.glob("**/glslang-*-data.cmake"):
        try:
            text = cmake_data.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in pattern.finditer(text):
            matches.append(Path(match.group(1)) / "bin")
    return matches


def shader_compiler_hints(repo_root: Path) -> list[Path]:
    hints: list[Path] = list(_iter_conan_glslang_hints(repo_root))

    vulkan_sdk = os.environ.get("VULKAN_SDK")
    current_platform = platform_key()
    if current_platform == "macos":
        hints.extend(
            [
                Path("/opt/homebrew/bin"),
                Path("/usr/local/bin"),
            ]
        )
        if vulkan_sdk:
            hints.append(Path(vulkan_sdk) / "bin")
    elif current_platform == "windows":
        if vulkan_sdk:
            hints.append(Path(vulkan_sdk) / "Bin")
        hints.extend(_iter_windows_sdk_bin_dirs())
    else:
        if vulkan_sdk:
            hints.append(Path(vulkan_sdk) / "bin")
        hints.append(Path("/usr/local/bin"))

    return _dedupe_paths(path for path in hints if str(path))


def _candidate_program_paths(directory: Path, name: str) -> list[Path]:
    candidates = [directory / name]
    if os.name == "nt" and not name.lower().endswith(".exe"):
        candidates.append(directory / f"{name}.exe")
    return candidates


def find_program(names: Iterable[str], hints: Optional[Iterable[Path]] = None) -> Optional[ToolMatch]:
    hint_list = [Path(path) for path in (hints or [])]

    for directory in hint_list:
        if not directory.is_dir():
            continue
        for name in names:
            for candidate in _candidate_program_paths(directory, name):
                if candidate.is_file():
                    return ToolMatch(name=name, path=candidate.resolve(), source=f"hint:{directory}")

    for name in names:
        resolved = shutil.which(name)
        if resolved:
            return ToolMatch(name=name, path=Path(resolved).resolve(), source="PATH")
    return None


def find_shader_compiler(repo_root: Path, compiler: str = "auto") -> Optional[ToolMatch]:
    hints = shader_compiler_hints(repo_root)
    requested = compiler.strip()

    if requested == "auto":
        return find_program(["glslc"], hints) or find_program(["glslangValidator"], hints)
    if requested == "glslc":
        return find_program(["glslc"], hints)
    if requested == "glslangValidator":
        return find_program(["glslangValidator"], hints)
    raise ValueError(f"Unsupported compiler '{compiler}'. Expected auto, glslc, or glslangValidator.")


def print_cli_value(value: str) -> int:
    sys.stdout.write(f"{value}\n")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shared tooling helpers for ProjectOptimizedRenderer.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("repo-root", help="Print the repository root.")

    preset_parser = subparsers.add_parser("preset", help="Resolve a preset for a build type and platform.")
    preset_parser.add_argument("--platform", default=platform_key(), help="Target platform key.")
    preset_parser.add_argument("--build-type", default="Debug", help="Build type to resolve.")

    build_dir_parser = subparsers.add_parser("build-dir", help="Print the build directory for a preset.")
    build_dir_parser.add_argument("--preset", required=True, help="Preset name.")

    exe_parser = subparsers.add_parser("executable", help="Locate the built renderer executable.")
    exe_parser.add_argument("--platform", default=platform_key(), help="Target platform key.")
    exe_parser.add_argument("--preset", help="Preset name.")
    exe_parser.add_argument("--build-dir", help="Explicit build directory.")

    build_type_parser = subparsers.add_parser("build-type", help="Normalize a build type string.")
    build_type_parser.add_argument("value", help="Build type to normalize.")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    repo_root = find_repo_root()

    if args.command == "repo-root":
        return print_cli_value(str(repo_root))
    if args.command == "preset":
        return print_cli_value(preset_for_build_type(args.build_type, args.platform))
    if args.command == "build-type":
        return print_cli_value(normalize_build_type(args.value))
    if args.command == "build-dir":
        return print_cli_value(str(build_dir_for_preset(repo_root, args.preset)))
    if args.command == "executable":
        resolved_build_dir = Path(args.build_dir).resolve() if args.build_dir else None
        executable = find_executable(repo_root, args.preset, resolved_build_dir, args.platform)
        if not executable:
            return 1
        return print_cli_value(str(executable))

    parser.error(f"Unknown command '{args.command}'")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
