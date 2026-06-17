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

The current renderer is already beyond a minimal triangle/sample project.

Implemented or partially implemented systems include:

- Vulkan 1.4 initialization
- `volk` Vulkan function loading
- VMA allocation support
- SDL3 windowing
- dynamic rendering
- synchronization2
- reverse-Z depth usage
- glTF/Sponza loading
- mesh and texture upload
- material descriptor sets
- Cook-Torrance PBR direct lighting
- normal mapping
- clustered forward-plus point lights
- HDR offscreen rendering
- Reinhard, AgX, and Khronos PBR Neutral tone mapping
- split-screen tone map comparison
- cascaded shadow maps
- hard shadows, PCF, and VSM modes
- VSM compute blur
- procedural/HDR sky support
- ImGui debug/editor panels
- GPU timing
- render statistics
- screenshot support
- runtime model reload
- extracted render passes for tone mapping, sky, shadows, and clustered light culling
- centralized shader interface/binding definitions
- internal render settings structs
- internal frame packet plus sticky scene packet submission
- device capability modeling for optional feature gating
- capability-driven native MSAA foundation for HDR scene rendering
- explicit scene HDR/depth attachment ownership with single-sample HDR resolve for tone mapping
- optional sample shading controls for MSAA experiments
- masked-material alpha-to-coverage controls for the main HDR scene path

Planned research directions include:

- HDR MSAA, sample shading, and alpha-to-coverage quality comparisons
- alpha-masked shadow quality research, separate from main-scene alpha-to-coverage
- SMAA as a later comparison or fallback path
- perceptual VRS
- improved render pass ownership
- clearer frame packet model
- renderer resource handles
- eventual C ABI boundary

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

The renderer currently expects a Sponza test scene path similar to:

```txt
assets/source/Sponza.gltf
```

If the application fails to start because the default model is missing, ensure the expected test assets exist under `assets/` or adjust the model path in the viewer/runtime path.

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

The current codebase should be understood as three conceptual layers:

```txt
C++ Research Viewer
  - SDL window loop
  - camera controls
  - ImGui panels
  - file dialogs
  - debug interaction

Renderer Orchestration Layer
  - RenderFramePacket
  - RenderScenePacket
  - RenderSettings
  - DrawCommand
  - frame submission flow
  - renderer-owned scene/resource coordination

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

- grouped renderer settings structs as the dominant state handoff
- internal frame packet abstraction as the primary submission path
- explicit scene packet submission beside frame submission
- resource handles for mesh/material/texture objects
- clearer pass ownership
- explicit resize pathway
- eventual C ABI boundary

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

Agent/project guidance:

- [`AGENTS.md`](AGENTS.md)

## Near-Term Roadmap

Recommended next engineering direction:

1. Consolidate `RenderFramePacket` as the main internal submission path.
2. Decide how `DrawCommand` participates in scene/frame submission.
3. Continue extracting pass-local ownership from `Renderer`.
4. Document capability-driven optional feature policy for unsupported platforms.
5. Implement capability-driven HDR MSAA as the first major AA feature on top of the new pass architecture.
6. Keep SMAA as a later native spatial comparison or fallback path, not the immediate AA milestone.
7. Add VRS only as an optional, capability-gated research path.
8. Introduce stronger resource-handle semantics before real external host integration.
9. Draft the public C ABI only after internal concepts stabilize.

## Status

This is an active renderer research and architecture project. APIs, file structure, and feature boundaries are expected to evolve.

The current priority is to keep renderer development fast while consolidating the new architecture direction clearly enough that the project can grow into an engine-ready renderer backend later.
