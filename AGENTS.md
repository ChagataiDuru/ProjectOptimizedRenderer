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
conan install . --output-folder=build --build=missing

# Configure with CMake
cmake --preset debug

# Build
cmake --build --preset debug
```

### Windows Debug Build

Use the Windows debug preset name: `win-debug`.

```powershell
# Install dependencies via Conan (run from repo root)
conan install . --output-folder=build --build=missing

# Configure with CMake
cmake --preset win-debug

# Build
cmake --build --preset win-debug
```

### Linux Build Status

Linux CI/builds are disabled and unsupported until a Linux CMake preset (for example, `linux-debug`) is added to `CMakePresets.json`.

Do not assume a Linux preset exists. If Linux support is requested, add the preset first, then document and validate exact commands such as:

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
