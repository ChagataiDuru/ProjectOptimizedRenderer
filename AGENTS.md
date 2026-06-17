# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**ProjectOptimizedRenderer** is a C++20 Vulkan renderer backend. The project uses CMake as the build system and Conan 2 for dependency management. SPIR-V shaders (`.spv`) are compiled separately and excluded from version control.

## Architecture Guardrails

- Keep this repository focused on the C++20 Vulkan renderer backend.
- Do not integrate the Odin runtime yet.
- Do not expose raw Vulkan handles externally; keep Vulkan implementation details encapsulated behind renderer-owned APIs.
- Avoid scope expansion into a render graph, ECS, asset database, or editor unless explicitly requested by the project owner.
- Preserve existing renderer behavior when making infrastructure, build, documentation, or refactor changes.

## Build System

This project uses **CMake + Conan 2**. Presets are defined in `CMakePresets.json`.

### macOS Debug Build

Use the real macOS debug preset name: `debug`.

```bash
# Install dependencies via Conan (run from repo root)
conan profile detect --force
conan install . --output-folder=build/conan --build=missing -s build_type=Debug

# Configure with CMake (using a preset defined in CMakePresets.json).
# Presets inherit build/conan/conan_toolchain.cmake from the Conan install step.
cmake --preset <preset-name>

# Build
cmake --build --preset debug
```

### Windows Debug Build

Use the Conan-generated Windows debug preset name: `conan-debug`. Run from a Visual Studio 2022 Developer PowerShell so Ninja can find the MSVC toolchain. `Launch-VsDevShell.ps1` may change the current directory, so return to the repo root before running Conan/CMake.

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Professional\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64
Set-Location "C:\Users\DUC2IB\Repos\ProjectOptimizedRenderer"

conan install . -of build/win-debug -s build_type=Debug --build=missing -c tools.cmake.cmaketoolchain:generator=Ninja

cmake --preset conan-debug
cmake --build --preset conan-debug
```

### Linux Build Status

Linux has a `linux-debug` preset in `CMakePresets.json`, but Linux CI/builds remain experimental and unvalidated unless a maintainer explicitly runs and documents them.

Do not describe Linux as supported without validation. If Linux support is requested, validate and document exact commands such as:

```bash
conan install . --output-folder=build --build=missing
cmake --preset linux-debug
cmake --build --preset linux-debug
```

## Key Technology Stack

- **Language:** C++20
- **Graphics API:** Vulkan
- **Memory Allocator:** VulkanMemoryAllocator (VMA)
- **Build:** CMake with `CMakePresets.json`
- **Package Manager:** Conan 2 (`conanfile.txt`)

## Directory Structure

- `src/` — C++ source files
- `include/` — Header files
- `cmake/` — CMake helper modules/scripts
- `build/` — Build output (gitignored)

## Shaders

SPIR-V compiled shader artifacts (`.spv`) are gitignored. Shaders need to be compiled separately (e.g. via `glslc` or `glslangValidator`) before the renderer can run.
