# 00 — Current Renderer State Analysis

Date: 2026-06-10

## Project Direction

ProjectOptimizedRenderer remains the core C++ Vulkan renderer repository.

A future Odin engine/editor host may exist as a separate project, connected through a narrow C ABI. The Odin side should not directly depend on C++ classes, STL containers, Vulkan internals, or renderer implementation details. Its future role is engine/editor host: scene graph, object management, asset database, editor UI, animation/audio/game systems, and frame packet production.

The C++ renderer remains responsible for Vulkan, GPU resource ownership, render passes, shader pipelines, profiling, screenshots, and visual research features.

This makes the project a hybrid:

- short term: renderer research prototype and graphics programming portfolio
- long term: engine-ready renderer backend with a stable external API

## Current Repository Shape

The repository is already beyond a minimal Vulkan triangle renderer. It currently resembles a research viewer/editor shell wrapped around a growing renderer backend.

Observed high-level structure:

- `src/main.cpp` owns application lifecycle, SDL window loop, camera, ImGui panels, file dialogs, render toggles, model reload, panorama load, and frame submission.
- `include/core` and `src/core` contain Vulkan context, window, swapchain, command buffers, frame sync, camera, and renderer orchestration.
- `include/resource` and `src/resource` contain GPU resource wrappers, glTF loading, models, textures, samplers, descriptors, scene normalization, and helper resource types.
- `include/debug` and `src/debug` contain ImGui integration, GPU timing, screenshots, and log UI support.
- `shaders` contains PBR, shadow, VSM blur, tone mapping, sky, and debug shader stages.
- `cmake` contains build helpers for shader compilation, MoltenVK detection, and volk discovery.

## Build and Platform Baseline

Current build direction:

- C++20
- CMake 3.25+
- Conan 2 dependency management
- Vulkan 1.4 baseline
- SDL3 for windowing/input
- volk for Vulkan function loading
- VMA for allocation
- glm for math
- spdlog for logging
- glslang/SPIRV-Tools for shader compilation support
- Dear ImGui integrated from `external/imgui`

Current platform intent:

- macOS / Apple Silicon through MoltenVK
- Windows / NVIDIA through native Vulkan SDK

This is already aligned with the future goal of a cross-platform renderer backend.

## Existing Renderer Capabilities

The renderer currently appears to include:

- Vulkan 1.4 initialization
- validation/debug messenger support in debug builds
- device selection with scoring for discrete/integrated GPUs
- graphics/compute/transfer queue family probing
- dynamic rendering
- dynamic rendering local read feature requirement
- synchronization2 feature enablement
- VMA allocator setup with buffer device address support
- swapchain-backed frame rendering
- reverse-Z depth usage
- glTF/Sponza loading
- mesh flattening into combined vertex/index buffers
- texture loading with fallback white texture
- material descriptor sets
- Cook-Torrance PBR shader path
- normal mapping path using derivative-reconstructed TBN
- HDR render target
- tone mapping pass
- Reinhard, AgX, and Khronos PBR Neutral operators
- split-screen tone map comparison
- cascaded shadow maps
- cascade debug coloring
- PCF shadows
- VSM moment shadows
- separable compute blur for VSM
- sky rendering with procedural and HDR panorama modes
- ImGui-based render/debug panels
- render statistics
- GPU timing
- screenshots
- runtime model reload

## Main Architectural Issue

The renderer is feature-rich, but ownership boundaries are not yet clean enough for engine integration.

The largest current risk is `Renderer` becoming a monolithic object that owns too many responsibilities:

- model loading
- resource upload
- descriptor allocation
- render pass creation
- pipeline creation
- shadow resources
- tone mapping resources
- sky resources
- runtime render stats
- screenshot requests
- ImGui integration point
- render loop-facing frame orchestration

This is acceptable for a research viewer, but it is not yet a stable engine-facing backend.

## Demo Shell vs Renderer Backend

The current application should be treated as two conceptual layers even before code is physically split.

### Demo / Viewer Shell

Currently represented mostly by `src/main.cpp` and ImGui setup.

Responsibilities:

- window creation
- input loop
- editor/debug panels
- camera movement
- file dialogs
- light sliders
- shadow controls
- tonemap controls
- screenshot hotkeys
- model/panorama selection

This layer can remain in C++ for now because it is useful for fast graphics research.

### Renderer Backend

Currently represented mostly by `Renderer`, Vulkan context/resource classes, and shaders.

Responsibilities:

- Vulkan device/swapchain interaction
- GPU resource ownership
- shader/pipeline creation
- frame synchronization
- render pass sequencing
- draw submission
- profiling/timing hooks
- backend feature support

This is the part that should eventually expose a stable C ABI for a future Odin host.

## Future Odin Boundary Rule

The future Odin engine/editor must communicate with the renderer only through plain C-compatible types.

Allowed across boundary:

- integer handles
- POD structs
- fixed-width numeric types
- pointers with explicit counts
- UTF-8 `const char*` paths where lifetime is caller-owned for the duration of the call
- explicit create/destroy functions

Forbidden across boundary:

- C++ classes
- `std::string`
- `std::vector`
- RAII-owned objects
- Vulkan object ownership
- VMA allocations
- raw renderer internals
- per-frame high-frequency single-object FFI calls

Preferred future model:

```c
typedef uint32_t RendererMeshHandle;
typedef uint32_t RendererMaterialHandle;
typedef uint32_t RendererTextureHandle;

typedef struct RendererDrawCommand {
    float transform[16];
    RendererMeshHandle mesh;
    RendererMaterialHandle material;
} RendererDrawCommand;

typedef struct RendererFramePacket {
    float view[16];
    float projection[16];
    float camera_position[3];
    RendererDrawCommand* draws;
    uint32_t draw_count;
} RendererFramePacket;
```

The Odin side should build a frame packet. The C++ renderer should consume that packet. This prevents the engine host from depending on renderer internals and avoids excessive FFI call overhead.

## Immediate Analysis Priorities

1. Document the actual frame flow from `main.cpp` into `Renderer::beginFrame()` / `Renderer::endFrame()` / internal `render()`.
2. Map current renderer-owned resources into categories: frame resources, scene resources, pass resources, persistent backend resources, and debug resources.
3. Identify which responsibilities must remain in the C++ renderer and which belong to the future Odin host.
4. Sketch the first stable C ABI concepts without implementing them yet.
5. Keep the current C++ viewer intact as the graphics research harness.

## Current Strategic Recommendation

Do not integrate Odin directly into this repository yet.

Instead:

1. Continue developing the C++ renderer in this repo.
2. Use the current C++ viewer/editor shell for visual research and debugging.
3. Design the C ABI as documentation first.
4. Later, create a separate Odin engine/editor prototype that initially talks to a mock renderer API.
5. Replace the mock renderer with the real C++ backend only after the renderer API boundary stabilizes.

This avoids destabilizing the Vulkan renderer while still moving toward an engine-ready architecture.

## Next Document

Recommended next file:

`docs/analysis/01-architecture-boundaries.md`

Purpose:

Define exact subsystem boundaries:

- application shell
- platform/window layer
- renderer backend
- resource system
- scene representation
- render passes
- debug tooling
- future C ABI surface
- future Odin host responsibilities
