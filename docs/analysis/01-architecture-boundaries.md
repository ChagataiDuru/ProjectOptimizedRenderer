# 01 — Architecture Boundaries

Date: 2026-06-10

## Decision Context

ProjectOptimizedRenderer remains the core C++ Vulkan renderer repository.

A future Odin engine/editor host is allowed, but it must remain a separate project connected through a narrow C ABI. The purpose of this document is to define the architectural boundaries before the renderer grows further.

The immediate goal is not to split the code physically. The immediate goal is to establish conceptual ownership so future refactors are deliberate.

## High-Level Target Architecture

```txt
+------------------------------------------------------------+
| Future Odin Engine / Editor Host                           |
|                                                            |
| Owns:                                                      |
| - editor application lifecycle                             |
| - scene graph / ECS / object hierarchy                     |
| - transform system                                         |
| - asset database metadata                                  |
| - animation and audio systems                              |
| - editor panels and tools                                  |
| - frame packet production                                  |
+----------------------------+-------------------------------+
                             |
                             | C ABI only
                             | POD structs, handles, explicit lifetime
                             v
+------------------------------------------------------------+
| C++ Renderer API Boundary                                  |
|                                                            |
| Owns:                                                      |
| - renderer_create / renderer_destroy                       |
| - renderer_load_* / renderer_unload_*                      |
| - renderer_submit_frame                                    |
| - renderer_resize                                          |
| - renderer_get_stats                                       |
+----------------------------+-------------------------------+
                             |
                             v
+------------------------------------------------------------+
| ProjectOptimizedRenderer C++ Vulkan Backend                |
|                                                            |
| Owns:                                                      |
| - Vulkan instance/device/swapchain                         |
| - GPU resource lifetime                                    |
| - shader and pipeline management                           |
| - render pass sequencing                                   |
| - profiling and screenshots                                |
| - visual research features                                 |
+------------------------------------------------------------+
```

## Current Practical Architecture

The current codebase is closer to this:

```txt
+------------------------------------------------------------+
| src/main.cpp                                               |
|                                                            |
| Owns:                                                      |
| - Window creation                                          |
| - VulkanContext creation                                   |
| - Swapchain creation                                       |
| - Renderer creation                                        |
| - ImGuiManager creation                                    |
| - Camera state                                             |
| - Light state                                              |
| - Shadow UI state                                          |
| - Tone mapping UI state                                    |
| - File dialogs                                             |
| - Model reload requests                                    |
| - HDR panorama load requests                               |
| - Main loop                                                |
| - SDL event handling                                       |
+----------------------------+-------------------------------+
                             |
                             v
+------------------------------------------------------------+
| Renderer                                                   |
|                                                            |
| Owns:                                                      |
| - Model loading and reload                                 |
| - Mesh buffers                                             |
| - Texture upload                                           |
| - Material descriptor sets                                 |
| - PBR pipeline                                             |
| - Shadow resources                                         |
| - VSM blur resources                                       |
| - HDR target                                               |
| - Tone map pipeline                                        |
| - Sky pipeline                                             |
| - Frame begin/end                                          |
| - Render stats                                             |
| - GPU timer                                                |
| - Screenshot system                                        |
| - ImGui overlay hook                                       |
+------------------------------------------------------------+
```

This is acceptable for a research viewer. It is not yet the final engine-ready shape.

## Boundary 1 — Application / Demo Viewer Shell

### Current Owner

Mostly `src/main.cpp`.

### Purpose

The demo viewer exists to support fast renderer development and visual research. It should remain lightweight and disposable compared to the renderer core.

### Valid Responsibilities

The C++ demo shell may own:

- SDL window creation for the current standalone viewer
- input/event loop for the current viewer
- free-fly camera controls
- ImGui debug panels
- model/panorama file dialogs
- runtime sliders for lighting, shadows, tone mapping, and debug modes
- screenshot hotkeys
- logging UI
- local test-scene setup

### Invalid Long-Term Responsibilities

The demo shell should not become the real engine layer.

It should not own:

- long-term asset database
- engine scene graph
- serialized world format
- animation system
- audio system
- gameplay ECS
- renderer internals
- GPU resource lifetime

### Future Direction

When the Odin host exists, this shell may continue to exist as a C++ research viewer. It does not need to be deleted.

The renderer should support both:

1. C++ research viewer
2. external Odin engine/editor host

This is useful because the C++ viewer remains the fastest place to test renderer-only features such as SMAA, VRS, CSM tuning, tone mapping, GPU captures, and profiling.

## Boundary 2 — Platform / Window Layer

### Current Owner

`Window`, SDL3 usage, swapchain surface setup, and parts of `main.cpp`.

### Purpose

Own OS/window integration and provide the renderer with a valid presentation surface.

### Current Responsibilities

- create SDL window
- query drawable/pixel extent
- handle resize/minimize concerns
- expose native window handle needed by swapchain/surface creation
- provide event polling in the standalone viewer path

### Future Boundary Question

The future Odin host may want to own the application window and editor UI. That means the renderer API must eventually support an externally provided native window/surface description.

There are two viable models:

### Model A — Renderer-Owned Window

```txt
Odin calls renderer_create_windowed(...)
C++ creates SDL window and owns swapchain.
```

Pros:

- easier first integration
- less cross-platform ABI complexity
- renderer controls surface creation

Cons:

- Odin editor has weaker ownership of app lifecycle
- harder to build a fully custom editor shell
- less like a commercial engine editor structure

### Model B — Host-Owned Window

```txt
Odin creates the application window.
Odin passes native window information to C++ renderer.
C++ creates/owns Vulkan surface and swapchain.
```

Pros:

- better long-term engine/editor architecture
- Odin owns the editor application lifecycle
- renderer becomes a backend service

Cons:

- harder cross-platform ABI
- native window handles differ by OS
- must carefully define resize/lifetime rules

### Recommendation

Document both, but implement Model A first when integration begins. Move to Model B only when the Odin editor has enough value to justify it.

For now, do not refactor window ownership prematurely.

## Boundary 3 — Vulkan Backend Core

### Current Owner

`VulkanContext`, `Swapchain`, `CommandBuffer`, `FrameSync`, and renderer-side Vulkan helpers.

### Purpose

Own direct Vulkan interaction and hide low-level GPU API details from higher layers.

### Valid Responsibilities

- Vulkan instance creation
- physical device selection
- logical device creation
- queue family discovery
- feature/extension enabling
- allocator creation
- debug messenger setup
- swapchain creation/recreation
- command pool/buffer management
- frame synchronization
- image layout transitions
- descriptor/pipeline creation helpers
- GPU timestamp queries

### External API Rule

No future Odin code should directly touch:

- `VkInstance`
- `VkDevice`
- `VkQueue`
- `VkCommandBuffer`
- `VkImage`
- `VkBuffer`
- `VmaAllocation`
- descriptor sets
- pipeline objects

The C API should expose renderer-level handles, not Vulkan handles.

Exception: an explicit low-level extension API may be added in the far future for tools, but it should not be part of the default engine/editor path.

## Boundary 4 — Renderer Orchestration

### Current Owner

`Renderer`.

### Purpose

Coordinate per-frame rendering.

### Current Responsibilities

`Renderer` currently handles a wide range of work:

- model load/reload
- GPU buffer upload
- texture upload
- descriptor allocation
- PBR pipeline
- shadow pipeline
- VSM blur pipeline
- HDR target
- tone map pass
- sky pass
- camera/light UBO upload
- frame begin/end
- render stats
- screenshot requests
- ImGui pass hook

### Problem

`Renderer` is currently a working but broad façade. If it keeps accumulating features directly, it will become hard to expose as a clean backend.

### Desired Direction

Over time, split the conceptual responsibilities into:

```txt
RendererBackend
  - owns Vulkan-facing backend lifetime

RenderWorld / RenderScene
  - owns renderable scene data known by the renderer

RenderResourceManager
  - owns meshes, textures, materials, samplers, buffers

RenderPasses
  - PBR/scene pass
  - shadow pass
  - VSM blur pass
  - sky pass
  - tone map pass
  - future SMAA pass
  - future VRS classification pass

FrameOrchestrator / RenderGraph
  - owns pass sequencing and transient attachments

RendererDebugTools
  - GPU timers
  - screenshots
  - debug overlays
```

This does not need to be coded immediately. It should guide future changes.

## Boundary 5 — Resource System

### Current Owner

`Buffer`, `Image`, `Texture`, `SamplerCache`, descriptor wrappers, `GLTFLoader`, `Model`, `Material`, and renderer-side upload logic.

### Purpose

Load CPU-side data, upload GPU-side data, and provide stable render resource handles.

### Current Strengths

- device-local buffer creation exists
- staged uploads exist
- texture loading exists
- fallback texture path exists
- sampler cache exists
- material texture descriptors exist
- glTF loading exists

### Current Risk

Asset loading and renderer resource creation are still tightly coupled.

A future engine/editor host should not force the renderer to load every asset directly from disk at draw time. Eventually it may need:

- asset database IDs
- hot reload
- async texture upload
- streaming
- background CPU decode
- GPU upload queues
- material instance updates

### Future Direction

The renderer should expose resource handles:

```c
typedef uint32_t RendererMeshHandle;
typedef uint32_t RendererTextureHandle;
typedef uint32_t RendererMaterialHandle;
```

The C API can initially support simple file-path loading:

```c
RendererMeshHandle renderer_load_mesh(RendererHandle renderer, const char* path);
RendererTextureHandle renderer_load_texture(RendererHandle renderer, const char* path, uint32_t flags);
```

Later, it can support memory-based loading:

```c
RendererMeshHandle renderer_create_mesh(RendererHandle renderer, const RendererMeshDesc* desc);
RendererTextureHandle renderer_create_texture(RendererHandle renderer, const RendererTextureDesc* desc);
```

## Boundary 6 — Scene Representation

### Current Owner

Renderer-owned `Model`, `MeshRenderData`, `SceneInfo`, and current test-scene state.

### Problem

The renderer currently loads a model and renders the model as its internal scene.

That is fine for Sponza/research testing, but an engine renderer should not assume the world is one loaded glTF.

### Future Direction

The future Odin host should own the engine scene graph.

The C++ renderer should own a renderable representation only:

- render mesh handles
- material handles
- per-frame transform matrices
- camera data
- light data
- draw command list
- optional visibility/culling data

A minimal future frame packet:

```c
typedef struct RendererCameraData {
    float view[16];
    float projection[16];
    float position[3];
    float near_z;
    float far_z;
} RendererCameraData;

typedef struct RendererDirectionalLightData {
    float direction[3];
    float intensity;
    float color[3];
    float ambient;
} RendererDirectionalLightData;

typedef struct RendererDrawCommand {
    float transform[16];
    uint32_t mesh_id;
    uint32_t material_id;
    uint32_t flags;
} RendererDrawCommand;

typedef struct RendererFramePacket {
    RendererCameraData camera;
    RendererDirectionalLightData sun;
    const RendererDrawCommand* draws;
    uint32_t draw_count;
} RendererFramePacket;
```

This preserves separation:

- Odin owns objects.
- Renderer owns render resources.
- Frame packet connects them.

## Boundary 7 — Render Passes and Render Graph

### Current Owner

Mostly `Renderer` and shader files.

### Current Pass Concepts

The renderer already has pass-level concepts even if they are not fully separated as objects:

- shadow pass
- VSM moment pass
- compute blur pass
- scene/PBR pass
- sky pass
- tone map pass
- ImGui overlay pass

### Future Risk

Adding SMAA, VRS, more post-processing, transparency, IBL, clustered lighting, or temporal passes directly into `Renderer` will increase coupling.

### Future Direction

Move toward explicit pass objects or a lightweight render graph:

```txt
ShadowPass
VsmBlurPass
ScenePass
SkyPass
TonemapPass
SmaaPass
VrsClassificationPass
DebugOverlayPass
```

Each pass should eventually declare:

- required input images/buffers
- output images/buffers
- descriptor requirements
- pipeline state
- resize behavior
- debug/profiling label

This is especially important for:

- SMAA edge/color/blend passes
- VRS image generation
- HDR/post-processing chains
- transient attachment reuse
- render doc/GPU capture readability

## Boundary 8 — Debug and Tooling

### Current Owner

`ImGuiManager`, `GPUTimer`, `Screenshot`, `LogSink`, and `main.cpp` panels.

### Purpose

Make renderer behavior inspectable.

### Valid Responsibilities

- GPU timings
- render stats
- pass timings
- screenshot capture
- debug overlays
- shader/pipeline toggles
- cascade visualization
- tone map comparison
- logs

### Future Direction

Debug tooling should be available to both:

1. C++ research viewer
2. future Odin editor host

But the API surface should be different.

C++ viewer can use direct ImGui callbacks.

Odin host should query plain data:

```c
typedef struct RendererStats {
    uint32_t draw_calls;
    uint32_t triangle_count;
    uint32_t mesh_count;
    uint32_t material_count;
    uint32_t texture_count;
    float gpu_frame_ms;
    float pass_shadow_ms;
    float pass_scene_ms;
    float pass_tonemap_ms;
} RendererStats;

void renderer_get_stats(RendererHandle renderer, RendererStats* out_stats);
```

The renderer should not require Odin to embed C++ ImGui objects.

## Boundary 9 — C ABI

### Purpose

The C ABI is the future contract between the separate Odin engine/editor and the C++ renderer.

### Rule

The C ABI should be designed after the renderer concepts stabilize, but documented early.

### API Style

Good:

```c
RendererHandle renderer_create(const RendererCreateInfo* info);
void renderer_destroy(RendererHandle renderer);
void renderer_resize(RendererHandle renderer, uint32_t width, uint32_t height);
void renderer_submit_frame(RendererHandle renderer, const RendererFramePacket* packet);
void renderer_get_stats(RendererHandle renderer, RendererStats* out_stats);
```

Bad:

```c
Renderer* renderer_get_cpp_renderer();
VkDevice renderer_get_vk_device();
std::vector<Mesh>* renderer_get_meshes();
void renderer_draw_one_object_every_call(...);
```

### Lifetime Rule

The side that creates an object destroys it.

If the C++ renderer creates a mesh resource, Odin receives only a handle. Odin requests deletion by handle.

If Odin passes a frame packet, the data is valid only for the duration of the call unless explicitly documented otherwise.

## Boundary 10 — Threading Model

### Target Direction

Future engine/editor integration should use frame packet handoff rather than direct shared state.

Conceptual model:

```txt
Engine/Editor Thread
  - update scene
  - update transforms
  - build RendererFramePacket N+1
  - submit packet

Render Thread
  - consume RendererFramePacket N
  - record Vulkan commands
  - submit GPU work
  - present
```

### Important Constraint

Thread separation is not automatically faster. It becomes useful only when:

- synchronization is controlled
- renderer consumes immutable frame snapshots
- resource lifetime is explicit
- per-object cross-language calls are avoided
- GPU submission and CPU scene update can overlap

### First Implementation Recommendation

When Odin integration begins, start single-threaded first:

```txt
Odin update -> renderer_submit_frame -> renderer_present
```

Then add double-buffered render packets and a render thread after the API is stable.

Avoid debugging cross-language FFI, renderer refactors, and multithreading at the same time.

## Ownership Matrix

| Responsibility | Current Owner | Long-Term Owner |
|---|---|---|
| Vulkan instance/device | C++ renderer | C++ renderer |
| Swapchain | C++ renderer | C++ renderer |
| Standalone window | C++ viewer | C++ viewer or Odin host |
| Demo camera controls | `main.cpp` | C++ viewer only |
| Engine scene graph | none / renderer test model | Odin host |
| Renderable resources | `Renderer` / resource classes | C++ renderer |
| Asset database | none | Odin host |
| glTF test loading | C++ renderer | C++ renderer initially; asset pipeline later |
| Material GPU descriptors | C++ renderer | C++ renderer |
| Transform hierarchy | none | Odin host |
| Per-frame draw list | implicit in renderer model | Odin host produces, C++ consumes |
| PBR/shadow/sky/tonemap passes | `Renderer` | C++ renderer pass system |
| SMAA/VRS research | C++ renderer | C++ renderer |
| Editor UI | C++ ImGui currently | C++ viewer now; Odin host later |
| GPU profiling | C++ renderer/debug | C++ renderer, queried by host |
| Screenshots | C++ renderer/debug | C++ renderer |

## Refactor Pressure Points

The following areas should be watched as the project grows:

1. `Renderer` size and responsibility count
2. `main.cpp` editor/demo growth
3. pass-specific descriptor/pipeline duplication
4. manual C++/GLSL layout synchronization
5. resize handling across HDR, depth, shadow, sky, and post-processing resources
6. resource reload and lifetime safety
7. debug UI directly reaching renderer internals
8. lack of explicit render packet abstraction
9. lack of stable handle-based resource API
10. absence of pass dependency declaration

## Short-Term Rule

Do not block renderer feature work with a premature engine rewrite.

For now:

- keep the C++ viewer
- continue renderer research features
- document the future boundary
- avoid adding irreversible coupling
- keep Odin as a future separate host, not an immediate dependency

## Recommended Next Document

`docs/analysis/02-renderer-technical-debt.md`

Purpose:

List concrete technical debt items with severity, why they matter, when to fix them, and what not to over-engineer yet.
