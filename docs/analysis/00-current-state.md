# 00 — Current Renderer State Analysis

Date: 2026-06-13

## Project Direction

ProjectOptimizedRenderer remains the core C++ Vulkan renderer repository.

Its role is now clearer than it was a few days ago:

- short term: a C++ Vulkan research viewer for rapid graphics iteration
- medium term: a cleaner renderer backend with explicit internal submission/data boundaries
- long term: an engine-ready renderer backend that can sit behind a narrow C ABI
- future optional host: a separate Odin engine/editor project, not a rewrite of this repository

The repo is no longer just "a viewer with renderer code inside it." It is actively moving toward a layered model where the C++ viewer remains useful, but renderer-facing concepts are being extracted into reusable interfaces and pass objects.

## Latest Architectural Movement

Recent commits materially changed the repo shape:

- `Refactor viewer UI and renderer frame/pass interfaces`
- `Prepare renderer for internal draw command submission`
- `Add renderer device capability model for optional feature gating`

Those changes mean the project is now past the purely aspirational stage for several roadmap items.

Implemented architectural progress includes:

- viewer panel registration extracted into `include/viewer/ViewerPanels.h` and `src/viewer/ViewerPanels.cpp`
- grouped renderer settings extracted into `include/core/RenderSettings.h`
- an internal `RenderFramePacket` introduced in `include/core/RenderFramePacket.h`
- tone mapping extracted into `include/renderpasses/TonemapPass.h` / `src/renderpasses/TonemapPass.cpp`
- `DrawCommand` introduced as an explicit internal submission primitive
- handle types introduced in `include/resource/RenderHandles.h`
- centralized shader interface constants and layout structs in `include/core/ShaderInterface.h`
- runtime device capability modeling added in `VulkanContext`
- optional-feature groundwork added for fragment shading rate, mesh shader/task shader, descriptor indexing, and timeline semaphore capability queries

This is still not a fully modular renderer backend, but it is no longer accurate to describe those concepts as only future intentions.

## Current Repository Shape

Observed high-level structure:

- `src/main.cpp` still owns application lifecycle, SDL loop, camera/input flow, deferred file actions, and viewer bootstrap.
- `include/viewer` and `src/viewer` now hold viewer-specific panel registration rather than keeping all panel setup in `main.cpp`.
- `include/core` and `src/core` hold renderer-facing orchestration and internal submission/state types such as `Renderer`, `RenderFramePacket`, `RenderSettings`, `DrawCommand`, `ShaderInterface`, `VulkanContext`, `Swapchain`, and sync/command helpers.
- `include/renderpasses` and `src/renderpasses` now contain extracted pass objects including `ShadowPass`, `SkyPass`, `TonemapPass`, and `ClusteredLightCullingPass`.
- `include/resource` and `src/resource` contain GPU resource wrappers, glTF loading, models, textures, samplers, descriptors, scene normalization, and render handles.
- `include/debug` and `src/debug` contain ImGui integration, GPU timing, screenshots, and log UI support.
- `shaders` contains scene, shadow, sky, tone mapping, VSM blur, and clustered light culling shader stages.
- `cmake` contains build helpers for shader compilation, MoltenVK detection, and dependency discovery.

## Build and Platform Baseline

Current build direction remains:

- C++20
- CMake 3.25+
- Conan 2 dependency management
- Vulkan 1.4 baseline
- SDL3 for windowing/input
- volk for Vulkan function loading
- VMA for allocation
- glm for math
- spdlog for logging
- glslang / SPIRV-Tools for shader compilation support
- Dear ImGui integrated from `external/imgui`

Current platform intent remains:

- macOS / Apple Silicon through MoltenVK
- Windows / NVIDIA through native Vulkan SDK

The repository is still explicitly cross-platform in intent even if feature parity and validation depth are stronger on some paths than others.

## Current Renderer Capabilities

The current renderer includes or clearly exposes the following systems:

### Core backend and platform

- Vulkan 1.4 initialization
- validation/debug messenger support in debug builds
- device scoring and physical-device selection
- queue family probing
- dynamic rendering
- dynamic rendering local read requirement
- synchronization2 enablement
- VMA allocator setup with buffer device address support
- swapchain-backed frame rendering
- reverse-Z depth usage

### Scene and material baseline

- glTF/Sponza loading
- scene normalization transform
- flattened mesh/index upload
- material descriptor sets
- texture loading with fallback white texture
- explicit internal mesh/material vectors
- explicit internal draw-command list generation

### Lighting and shading

- Cook-Torrance PBR shading
- normal mapping path
- directional light controls
- clustered forward-plus point light support
- compute clustered light culling pass

### Image formation and post-processing

- HDR offscreen rendering
- tone mapping pass extraction via `TonemapPass`
- Reinhard, AgX, and Khronos PBR Neutral operators
- split-screen tone map comparison

### Shadows and atmosphere

- cascaded shadow maps
- cascade debug coloring
- PCF shadows
- VSM moment shadows
- separable compute blur for VSM
- extracted `ShadowPass`
- procedural sky and HDR panorama sky via `SkyPass`

### Debugging and tooling

- ImGui debug/editor panels
- render statistics
- GPU timing
- screenshots
- runtime model reload
- centralized shader binding/layout constants
- static layout checks for shader-facing structs

### Capability and future-feature groundwork

- device capability model exposed through `RendererDeviceFeatures`
- queried mesh-shader property surface through `RendererMeshShaderProperties`
- optional feature queries for:
  - fragment shading rate
  - mesh shader
  - task shader
  - descriptor indexing
  - timeline semaphore

This is important because perceptual VRS and other optional research paths now have a concrete capability-discovery foundation instead of requiring ad hoc probing later.

## Main Architectural State Today

The repository now sits in an intermediate but much healthier state.

### What improved

The following roadmap ideas have moved from concept into code:

- viewer-only panel registration is no longer fully trapped in `main.cpp`
- grouped settings structs exist
- a first internal frame-packet type exists
- pass extraction has started and is real
- internal resource handles exist
- internal draw submission vocabulary exists
- shader interface ownership is more centralized
- optional feature gating groundwork exists

### What is still unresolved

The renderer is still not yet a clean external backend boundary.

`Renderer` still owns too much of the end-to-end system, including:

- asset import orchestration
- GPU resource creation and upload
- scene/material vectors and draw-list rebuilding
- scene pass orchestration
- pass coordination
- resize handling
- screenshot/profiling integration
- legacy setter-based state mutation alongside packet-based submission

That means the architecture is improving, but the project is still in transition rather than in a final backend form.

## Viewer Shell vs Renderer Backend

The current application should now be understood as three practical layers instead of only two:

### 1. Viewer shell

Represented mainly by:

- `src/main.cpp`
- `include/viewer/ViewerPanels.h`
- `src/viewer/ViewerPanels.cpp`

Responsibilities:

- app lifecycle
- window/input loop
- camera movement and user interaction
- file dialogs / asset browsing
- debug UI registration
- building immediate research-viewer state

### 2. Renderer orchestration layer

Represented mainly by:

- `Renderer`
- `RenderFramePacket`
- `RenderSettings`
- `DrawCommand`
- `ShaderInterface`

Responsibilities:

- frame submission boundary
- renderer-side persistent state
- draw-list rebuilding
- resource ownership coordination
- per-frame settings handoff
- pass ordering

### 3. Pass/backend layer

Represented mainly by:

- `ShadowPass`
- `SkyPass`
- `TonemapPass`
- `ClusteredLightCullingPass`
- Vulkan resource/context helpers

Responsibilities:

- pass-local pipelines and descriptors
- per-pass record logic
- backend feature usage
- low-level Vulkan lifetime management

This layering is still incomplete, but it is a real shift toward an engine-facing backend architecture.

## Current Roadmap Position

A practical status read of the roadmap is:

| Item | Status | Notes |
|---|---|---|
| Viewer shell separation | Partial | `ViewerPanels` extracted, but `main.cpp` still owns lifecycle and input flow. |
| Renderer settings structs | Done (initial) | `RenderSettings.h` exists and is in active use. |
| Internal frame packet | Partial | `RenderFramePacket` exists and `submitFrame` exists, but packet ownership is not yet the only submission path. |
| Resource handle model | Partial | `MeshHandle`, `MaterialHandle`, `TextureHandle` exist; generation/versioning and public resource APIs do not. |
| Pass ownership extraction | Partial, meaningful | `TonemapPass`, `ShadowPass`, `SkyPass`, and clustered light culling pass exist. |
| Shader interface validation | Partial | Centralized constants and static asserts exist; reflection/codegen does not. |
| Capability-driven optional features | Started well | `RendererDeviceFeatures` is now a real foundation for VRS and mesh-shader experiments. |
| SMAA | Not started | Still a next major renderer-feature milestone. |
| Perceptual VRS | Not started, groundwork present | Capability query and optional feature gating now exist. |
| C ABI boundary | Not started | Still documentation-first, not implementation-ready. |

## Future Odin Boundary Rule

The future Odin engine/editor must still communicate with the renderer only through plain C-compatible types.

Allowed across boundary:

- integer handles
- POD structs
- fixed-width numeric types
- pointers with explicit counts
- UTF-8 `const char*` paths with caller-owned lifetime for the duration of the call
- explicit create/destroy functions

Forbidden across boundary:

- C++ classes
- `std::string`
- `std::vector`
- RAII-owned objects
- Vulkan object ownership
- VMA allocations
- raw renderer internals
- high-frequency per-object FFI mutator chatter

The recent internal additions are useful precisely because they move the codebase closer to C-compatible concepts without prematurely locking the external ABI.

## Immediate Strategic Recommendation

The next slice should not be "jump straight into Odin" and it also should not be "add more big features without consolidating the new architecture." The right next move is:

1. make `RenderFramePacket` the dominant internal submission path rather than keeping many parallel setters
2. decide how `DrawCommand` enters the frame packet or a closely related scene submission object
3. continue moving pass-local resources out of `Renderer`
4. document capability-driven feature policy for unsupported platforms
5. implement SMAA on top of the new pass/resource direction

## Current Strategic Recommendation

Do not integrate Odin directly into this repository yet.

Instead:

1. continue using this repo as the C++ renderer and research viewer
2. consolidate the internal submission boundary now that packet/handle/pass concepts exist
3. use the current viewer as the first client of that cleaner renderer interface
4. add SMAA as the first major feature that truly stress-tests the extracted pass architecture
5. only draft the public C ABI after the internal packet/handle/pass model stops shifting

## Next Document

Recommended next file:

`docs/analysis/03-feature-roadmap.md`

Purpose:

Re-mark the roadmap based on actual progress already landed in the repository, especially:

- viewer extraction progress
- pass extraction progress
- handle/draw-command progress
- frame-packet progress
- capability-gated optional feature groundwork
- what still remains before SMAA, VRS, and any future C ABI work
