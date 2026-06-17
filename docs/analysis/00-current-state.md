# 00 — Current Renderer State Analysis

Date: 2026-06-17

## Project Direction

ProjectOptimizedRenderer remains the core C++ Vulkan renderer repository.

Its role is now clearer than it was a few days ago:

- short term: a C++ Vulkan research viewer for rapid graphics iteration
- medium term: a cleaner renderer backend with explicit internal submission/data boundaries
- long term: an engine-ready renderer backend that can sit behind a narrow C ABI
- future optional host: a separate Odin engine/editor project, not a rewrite of this repository

The repo is no longer just "a viewer with renderer code inside it." It is actively moving toward a layered model where the C++ viewer remains useful, but renderer-facing concepts are being extracted into reusable interfaces and pass objects.

## Latest Architectural Movement

Recent work materially changed the repo shape:

- build boundary split into `por_renderer_core`, `por_viewer`, `por_shaders`, and ABI helper/test targets
- `RendererInstance` introduced as the lifecycle facade over window, Vulkan context, swapchain, and `Renderer`
- glTF import/reload policy moved to the viewer; `Renderer` now accepts uploaded CPU model data
- `RendererResourceManager` owns renderer-side GPU mesh/material/texture data behind existing handles
- ImGui dependency removed from `Renderer` and routed through a viewer-only `RendererOverlay`
- `include/por/por_renderer.h` added as an ABI draft with C11/C++20 compile tests and a mock implementation

Those changes mean the project is now past the purely aspirational stage for ABI preparation, though the ABI is still a draft rather than a stable external contract.

Implemented architectural progress includes:

- viewer panel registration extracted into `include/viewer/ViewerPanels.h` and `src/viewer/ViewerPanels.cpp`
- grouped renderer settings extracted into `include/core/RenderSettings.h`
- an internal `RenderFramePacket` introduced in `include/core/RenderFramePacket.h`
- tone mapping extracted into `include/renderpasses/TonemapPass.h` / `src/renderpasses/TonemapPass.cpp`
- `DrawCommand` introduced as an explicit internal submission primitive
- `RendererInstance` introduced as the viewer-facing lifecycle boundary
- `RenderScenePacket` introduced as sticky scene submission beside per-frame state
- `RendererResourceManager` introduced for renderer-owned GPU resources
- `include/por/por_renderer.h` introduced as a C ABI draft
- handle types introduced in `include/resource/RenderHandles.h`
- centralized shader interface constants and layout structs in `include/core/ShaderInterface.h`
- runtime device capability modeling added in `VulkanContext`
- optional-feature groundwork added for fragment shading rate, mesh shader/task shader, descriptor indexing, and timeline semaphore capability queries

This is still not a fully modular renderer backend, but it is no longer accurate to describe those concepts as only future intentions.

## Current Repository Shape

Observed high-level structure:

- `src/main.cpp` owns viewer policy: camera/input flow, file dialogs, CPU glTF import/reload, and ImGui interaction.
- `RendererInstance` owns renderer lifecycle: SDL window creation, Vulkan context, swapchain, renderer creation, resize, frame rendering, and overlay attachment.
- `include/viewer` and `src/viewer` now hold viewer-specific panel registration rather than keeping all panel setup in `main.cpp`.
- `include/core` and `src/core` hold renderer-facing orchestration and internal submission/state types such as `RendererInstance`, `Renderer`, `RenderFramePacket`, `RenderScenePacket`, `RenderSettings`, `DrawCommand`, `ShaderInterface`, `VulkanContext`, `Swapchain`, and sync/command helpers.
- `include/renderpasses` and `src/renderpasses` now contain extracted pass objects including `ShadowPass`, `SkyPass`, `TonemapPass`, and `ClusteredLightCullingPass`.
- `include/resource` and `src/resource` contain GPU resource wrappers, the renderer resource manager, glTF loading, CPU model data, textures, samplers, descriptors, scene normalization, and render handles.
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
- Linux through a `linux-debug` preset, currently experimental/unvalidated unless explicitly tested

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

- viewer-side glTF/Sponza loading
- viewer-side scene normalization transform
- renderer-side flattened mesh/index upload through `RendererResourceManager`
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
- runtime model reload owned by the viewer/application path
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

- renderer lifecycle is now hidden behind `RendererInstance`
- viewer-owned model import/reload is separated from renderer GPU resource upload
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

`Renderer` is less application-shaped than before, but it still owns substantial backend orchestration:

- scene pass orchestration
- pass coordination
- resize handling
- screenshot/profiling integration
- descriptor/pipeline ownership across resource and resize changes

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

### 2. Renderer instance facade

Represented mainly by:

- `RendererInstance`
- `RendererOverlay`

Responsibilities:

- renderer-owned SDL window/runtime for the first ABI model
- `VulkanContext`, `Swapchain`, and `Renderer` lifecycle
- formal `resize(width, height)`
- `renderFrame(RenderFramePacket)`
- viewer-only overlay attachment

### 3. Renderer orchestration layer

Represented mainly by:

- `Renderer`
- `RenderFramePacket`
- `RenderScenePacket`
- `RenderSettings`
- `DrawCommand`
- `RendererResourceManager`
- `ShaderInterface`

Responsibilities:

- frame and sticky scene submission boundaries
- renderer-side persistent state
- resource ownership coordination
- per-frame settings handoff
- pass ordering

### 4. Pass/backend layer

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
| Viewer shell separation | Partial, improved | `RendererInstance` now owns renderer lifecycle; `main.cpp` still owns viewer loop/input/import policy. |
| Renderer settings structs | Done (initial) | `RenderSettings.h` exists and is in active use. |
| Internal frame packet | Done, internal-only | `RenderFramePacket` is the primary per-frame submission path and now pairs with sticky `RenderScenePacket` scene submission. |
| Resource handle model | Partial | Handles exist and `RendererResourceManager` owns uploaded resources; stable per-resource create/destroy APIs are still incomplete. |
| Pass ownership extraction | Partial, meaningful | `TonemapPass`, `ShadowPass`, `SkyPass`, and clustered light culling pass exist. |
| Shader interface validation | Partial | Centralized constants and static asserts exist; reflection/codegen does not. |
| Capability-driven optional features | Established foundation | `RendererDeviceFeatures` is a centralized detect-once snapshot with baseline-vs-optional policy clarified in code and logging. |
| HDR MSAA | Foundation implemented | Immediate native AA milestone with explicit scene attachments, resolve, pipeline compatibility, sample-shading controls, and masked-material A2C. |
| SMAA | Not started | Later native spatial comparison or fallback path. |
| Perceptual VRS | Not started, groundwork present | Capability query and optional feature gating now exist. |
| C ABI boundary | Drafted, not stable | `include/por/por_renderer.h` compiles as C11/C++20 with a mock backend; real wrapper exists as an internal target but resource APIs are not mature. |

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

1. harden `RendererInstance` as the only viewer-facing renderer lifecycle boundary
2. evolve `RendererResourceManager` from batch model upload into stable handle create/destroy operations
3. extend C ABI compile/mock tests while keeping Odin out
4. continue moving pass-local resources out of `Renderer` only where it lowers current coupling
5. keep SMAA, VRS, and render-graph work deferred until the boundary stabilizes

## Current Strategic Recommendation

Do not integrate Odin directly into this repository yet.

Instead:

1. continue using this repo as the C++ renderer and research viewer
2. consolidate the `RendererInstance`/packet/handle/resource boundary now that the viewer is its first client
3. use the C header/mock tests to validate ABI shape without treating it as stable
4. avoid new major renderer features until resource lifetime and C++ boundary semantics settle

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
- what still remains before deeper AA comparisons, VRS, and any future C ABI work
