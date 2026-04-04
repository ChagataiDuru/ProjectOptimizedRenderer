# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

**ProjectOptimizedRenderer** is a C++ Vulkan renderer. The project uses CMake as the build system and Conan for dependency management. SPIR-V shaders (`.spv`) are compiled separately and excluded from version control.

## Build System

This project uses **CMake + Conan 2**. The typical workflow:

```bash
# Install dependencies via Conan (run from repo root)
conan install . --output-folder=build --build=missing

# Configure with CMake (using a preset defined in CMakePresets.json)
cmake --preset <preset-name>

# Build
cmake --build --preset <preset-name>
```

CMake presets are defined in `CMakePresets.json`. Check that file for available preset names (e.g. `debug`, `release`).

## Key Technology Stack

- **Language:** C++20 (expected, standard for modern Vulkan renderers)
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
