# ProjectOptimizedRenderer

ProjectOptimizedRenderer is a modern C++ Vulkan renderer focused on native image quality, explicit GPU control, and long-term engine-readiness.

The project is intentionally not a generic commercial-engine clone. Its goal is to explore and implement renderer techniques directly in Vulkan, with clear ownership of the graphics stack instead of relying on opaque engine abstractions.

## Project Direction

Short term, this repository is a **C++ Vulkan renderer research viewer**.

Long term, it should become an **engine-ready renderer backend** with a stable API boundary.

A future engine/editor host may be written separately in Odin, but this repository remains the core C++ renderer. Any future Odin integration should happen through a narrow C ABI using plain data structs and opaque handles.

See:

- [`docs/analysis/00-current-state.md`](docs/analysis/00-current-state.md)
- [`docs/analysis/01-architecture-boundaries.md`](docs/analysis/01-architecture-boundaries.md)
- [`docs/analysis/04-engine-api-direction.md`](docs/analysis/04-engine-api-direction.md)
- [`docs/decisions/ADR-0003-cpp-core-odin-host-boundary.md`](docs/decisions/ADR-0003-cpp-core-odin-host-boundary.md)

## Core Goals

- Build a renderer from owned C++/Vulkan code.
- Explore native rendering quality instead of depending on upscaling-first pipelines.
- Implement modern spatial rendering techniques.
- Keep the renderer understandable, inspectable, and measurable.
- Support Windows/NVIDIA and macOS/Apple Silicon development paths.
- Grow toward an engine-facing renderer backend without prematurely building a full engine.

## Current Feature Baseline

The renderer is well past a minimal triangle sample. It has a PBR scene pass with
normal mapping and alpha masking, clustered forward-plus point lights, cascaded
shadow maps, a tone mapping pass, procedural/HDR sky, MSAA, ImGui tooling, and an
extracted pass structure.

The maintained feature baseline and status table live in
[`docs/analysis/00-current-state.md`](docs/analysis/00-current-state.md) and
[`docs/analysis/03-feature-roadmap.md`](docs/analysis/03-feature-roadmap.md).

## Technology Stack

| Area | Choice |
|---|---|
| Language | C++20 |
| Graphics API | Vulkan 1.4 |
| Windowing/Input | SDL3 |
| Build system | CMake |
| Package manager | Conan 2 |
| Vulkan loader | volk |
| GPU allocator | VulkanMemoryAllocator / VMA |
| Math | glm |
| Logging | spdlog |
| Debug UI | Dear ImGui |
| Shader tooling | glslang / SPIRV-Tools |

## Repository Layout

```txt
.
├── assets/                 # Test assets, including Sponza when available
├── cmake/                  # CMake helper modules
├── docs/
│   ├── analysis/           # Architecture and roadmap notes
│   └── decisions/          # ADR-style project decisions
├── external/               # Third-party source dependencies/submodules
├── include/                # Public/internal C++ headers
├── shaders/                # GLSL shader sources
├── src/                    # C++ implementation
├── CMakeLists.txt
├── CMakePresets.json
├── conanfile.txt
└── README.md
```

## Dependencies

Required tools:

- CMake 3.25+
- Conan 2
- Ninja
- C++20 compiler
- Vulkan SDK / Vulkan-capable runtime
- Git with submodule support

Platform-specific notes:

- **Windows:** native Vulkan SDK path is auto-detected from `VULKAN_SDK` or common `C:/VulkanSDK/*` locations.
- **macOS Apple Silicon:** MoltenVK/Homebrew path support is expected through the existing CMake configuration.

The project has a VMA submodule:

```bash
git submodule update --init --recursive
```

## Build Instructions

From the repository root:

```bash
# Recommended macOS quickstart
./scripts/doctor-macos.sh
./scripts/bootstrap-macos.sh Debug
./scripts/build-macos.sh Debug
./scripts/run-macos.sh Debug
```

For Windows:

```powershell
.\scripts\doctor-windows.ps1
.\scripts\bootstrap-windows.ps1 Debug
.\scripts\build-windows.ps1 Debug
.\scripts\run-windows.ps1 Debug
```

Available presets are defined in [`CMakePresets.json`](CMakePresets.json), including:

- `linux-debug`
- `debug`
- `macos-debug`
- `relwithdebinfo`
- `release`
- `win-debug`
- `win-relwithdebinfo`
- `win-release`

The wrapper scripts pass the Conan toolchain explicitly from `build/conan/conan_toolchain.cmake`. If you configure manually, pass that toolchain path yourself and rerun `conan install` with the matching `-s build_type=<BuildType>`. Detailed workflow notes live in [`docs/developer-tooling.md`](docs/developer-tooling.md).

## Shader Compilation

GLSL shaders live in [`shaders/`](shaders/).

Shader compilation is integrated through the CMake helper in [`cmake/CompileShaders.cmake`](cmake/CompileShaders.cmake). Compiled SPIR-V outputs are build artifacts and should not be committed.

## Scripted Capture Mode

The viewer can render a fixed settings matrix to PNGs and exit, which makes rendering
findings reproducible without driving the ImGui overlay by hand:

```bash
# One PNG per preset into screenshots/<dir>/<preset>.png
./build/debug/ProjectOptimizedRenderer --capture screenshots/run --preset all --frames 30

# List preset ids (shadow filter modes, cascade debug, MSAA variants, tone maps, ...)
./build/debug/ProjectOptimizedRenderer --list-presets

# Inspect a single preset against a different scene
./build/debug/ProjectOptimizedRenderer --capture screenshots/run --preset shadow-vsm --scene assets/source/Sponza.gltf
```

Supported arguments: `--capture <dir>`, `--preset <id|all>` (repeatable), `--scene <gltf>`,
`--frames <n>`, `--width <n>`, `--height <n>`, `--list-presets`, `-h/--help`.
Capture output is written under `screenshots/`, which is gitignored. Preset definitions
live in `include/viewer/CapturePresets.h` / `src/viewer/CapturePresets.cpp`; the artifact
workflow that uses them is described in [`docs/workstreams/README.md`](docs/workstreams/README.md).

Standalone tooling is also available:

```bash
python tools/shaders/compile_shaders.py
python tools/shaders/compile_shaders.py --validate-only --preset debug
```

Currently referenced shader stages include:

- PBR vertex/fragment
- normal visualization fragment
- shadow vertex/fragment
- VSM blur compute
- clustered light culling compute
- tone map vertex/fragment
- sky vertex/fragment

## Assets

The C++ viewer currently loads a default Sponza test scene path similar to:

```txt
assets/source/Sponza.gltf
```

If the application fails to start because the default model is missing, ensure the expected test assets exist under `assets/` or adjust the viewer's default model path. glTF import remains viewer/application policy; the renderer backend receives CPU model data as uploaded GPU resources.

## Runtime Controls

Current viewer controls include:

| Input | Action |
|---|---|
| WASD | Move camera |
| Mouse | Look around when mouse capture is enabled |
| F1 | Toggle mouse capture / UI interaction mode |
| F2 | Request screenshot |
| F11 | Toggle ImGui overlay visibility |
| Escape | Exit application |

The ImGui overlay exposes panels for:

- performance
- camera
- lighting
- cascaded shadows
- shadow filter mode
- sky mode
- tone mapping
- anti-aliasing sample count, sample shading, and alpha-to-coverage
- render stats
- GPU timings
- scene hierarchy
- selected mesh/material properties
- log console

## Architecture Notes

The current codebase should be understood as four practical layers:

```txt
C++ Research Viewer
  - SDL window loop
  - camera controls
  - ImGui panels
  - file dialogs
  - debug interaction

Renderer Instance Facade
  - renderer-owned SDL window/runtime for the first ABI model
  - VulkanContext / Swapchain / Renderer lifecycle
  - resize(width, height)
  - renderFrame(RenderFramePacket)
  - viewer-only overlay attachment

Renderer Orchestration Layer
  - RenderFramePacket
  - RenderScenePacket
  - RenderSettings
  - DrawCommand
  - RendererResourceManager
  - frame submission flow
  - renderer-owned GPU resource coordination

C++ Vulkan Pass/Backend Layer
  - Vulkan resources
  - pass objects
  - shaders/pipelines
  - frame synchronization
  - GPU profiling
  - screenshots
```

The viewer is useful and should remain, but it is not the final engine/editor layer.

Future work should gradually move toward:

- using `RendererInstance` as the C++ boundary tested by the viewer
- growing stable mesh/material/texture resource create/destroy operations behind handles
- keeping glTF import and reload behavior in viewer/application code
- keeping ImGui viewer-only and out of the public ABI
- tightening pass-local ownership without introducing a full render graph yet
- treating `include/por/por_renderer.h` as a draft ABI contract until resource semantics settle

## Capability Policy

Runtime feature handling follows a simple renderer policy:

- detect capabilities once during device selection
- store them centrally in `RendererDeviceFeatures`
- enforce only the current renderer baseline at startup
- keep unsupported optional features functional via fallback behavior
- have future systems query cached capability state instead of reprobeing Vulkan ad hoc

This keeps optional work such as SMAA/VRS research additive without quietly turning it into a new startup requirement.

## Future Odin Host Direction

A future Odin host is possible, but it should be a separate project.

The intended future model is:

```txt
Odin Engine / Editor Host
  -> C ABI with POD structs and handles
  -> C++ ProjectOptimizedRenderer backend
  -> Vulkan
```

The Odin side may eventually own:

- scene graph
- object/component model
- transform hierarchy
- asset metadata
- editor UI
- audio/animation/game systems
- frame packet generation

The C++ renderer should continue to own:

- Vulkan device/swapchain
- GPU resources
- render pass sequencing
- shader/pipeline objects
- profiling and screenshots

This direction is recorded in [`ADR-0003`](docs/decisions/ADR-0003-cpp-core-odin-host-boundary.md).

## Documentation Map

Architecture and planning:

- [`docs/analysis/00-current-state.md`](docs/analysis/00-current-state.md)
- [`docs/analysis/01-architecture-boundaries.md`](docs/analysis/01-architecture-boundaries.md)
- [`docs/analysis/02-renderer-technical-debt.md`](docs/analysis/02-renderer-technical-debt.md)
- [`docs/analysis/03-feature-roadmap.md`](docs/analysis/03-feature-roadmap.md)
- [`docs/analysis/04-engine-api-direction.md`](docs/analysis/04-engine-api-direction.md)

Decisions:

- [`docs/decisions/ADR-0003-cpp-core-odin-host-boundary.md`](docs/decisions/ADR-0003-cpp-core-odin-host-boundary.md)

Artifact / cleanup workstreams:

- [`docs/workstreams/README.md`](docs/workstreams/README.md)
- [`docs/workstreams/inventory.md`](docs/workstreams/inventory.md)

Agent/project guidance:

- [`AGENTS.md`](AGENTS.md)

## Near-Term Roadmap

The maintained roadmap and its current status live in
[`docs/analysis/03-feature-roadmap.md`](docs/analysis/03-feature-roadmap.md).

## Status

This is an active renderer research and architecture project. APIs, file structure, and feature boundaries are expected to evolve.

The current priority is to keep renderer development fast while consolidating the new architecture direction clearly enough that the project can grow into an engine-ready renderer backend later.
