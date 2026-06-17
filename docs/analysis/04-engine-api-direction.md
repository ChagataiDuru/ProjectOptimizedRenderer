# 04 — Engine API Direction

Date: 2026-06-17

## Purpose

This document defines the future API direction between ProjectOptimizedRenderer and a possible external engine/editor host.

The current decision is:

> ProjectOptimizedRenderer remains the core C++ Vulkan renderer repository. A future Odin engine/editor is a separate host project connected through a narrow C ABI.

This document is a design contract. A draft C header and initial wrapper target now exist, but they are ABI preparation artifacts rather than a stable external integration commitment.

## Non-Goals

This document does not require immediate implementation of:

- Odin integration
- a new engine repository
- threaded rendering
- a full render graph
- an ECS
- a complete asset database
- multi-window editor support
- exposing Vulkan handles

The goal is to prevent future architecture drift.

## Target Relationship

```txt
+------------------------------------------------------------+
| Future Odin Engine / Editor Host                           |
|                                                            |
| Owns:                                                      |
| - app/editor lifecycle                                     |
| - scene graph / ECS / object model                         |
| - transform hierarchy                                      |
| - asset database metadata                                  |
| - animation/audio/game systems                             |
| - editor panels and tooling                                |
| - frame packet construction                                |
+----------------------------+-------------------------------+
                             |
                             | C ABI
                             | POD structs + handles only
                             v
+------------------------------------------------------------+
| ProjectOptimizedRenderer C API                             |
|                                                            |
| Owns contract only:                                        |
| - create/destroy renderer                                  |
| - resize                                                   |
| - load/unload resources                                    |
| - submit frame packet                                      |
| - query stats/errors                                       |
+----------------------------+-------------------------------+
                             |
                             v
+------------------------------------------------------------+
| C++ Vulkan Renderer Backend                                |
|                                                            |
| Owns:                                                      |
| - Vulkan instance/device/swapchain                         |
| - GPU memory/resources                                     |
| - shader/pipeline/pass systems                             |
| - frame synchronization                                    |
| - profiling/screenshots/debug data                         |
+------------------------------------------------------------+
```

## Core Principle

The C ABI must be a stable renderer boundary, not a leak of the C++ implementation.

Allowed:

- opaque renderer handles
- integer resource handles
- plain structs
- explicit counts for arrays
- explicit lifetime rules
- explicit error reporting

Forbidden:

- C++ classes
- `std::string`
- `std::vector`
- templates
- exceptions crossing the boundary
- RAII objects crossing the boundary
- Vulkan handles as normal API objects
- VMA allocations
- internal descriptor/pipeline objects
- pointers to renderer-owned internal containers

## API Maturity Stages

The API should evolve in stages.

## Stage 0 — Documentation Only

Completed.

Actions:

- document architecture direction
- define rough API concepts
- keep renderer development in C++
- do not add Odin dependency
- do not force C ABI into unstable internals

Exit criteria:

- analysis docs exist
- API concepts are stable enough to discuss
- renderer roadmap knows what future host needs

## Stage 1 — Internal C++ Boundary

Before exposing C ABI, create internal C++ equivalents:

```cpp
struct RendererCreateInfo;
struct RenderFramePacket;
struct RendererStats;
struct ShadowSettings;
struct TonemapSettings;
struct SkySettings;
struct DebugViewSettings;
```

Current status:

- `RendererInstance` owns the first renderer-owned SDL window/runtime lifecycle
- the viewer submits frame state through `RenderFramePacket`
- scene content is sticky through `RenderScenePacket`
- glTF import/reload policy lives in viewer code
- GPU resources are grouped behind `RendererResourceManager`

This reduces risk because the API design is tested inside C++ before FFI becomes a long-term contract.

## Stage 2 — C Header Draft

Create a header such as:

```txt
include/api/por_renderer_api.h
```

This header should compile as C and C++.

Current status: `include/por/por_renderer.h` exists and is compile/run tested from C11 and C++20 against a mock implementation.

## Stage 3 — C Wrapper Around C++ Backend

An initial thin C wrapper target exists:

```txt
src/api/por_renderer.cpp
```

The wrapper owns translation between C API structs and C++ renderer internals.

Rules:

- catch all C++ exceptions inside the wrapper
- return explicit error codes
- never expose C++ exceptions through C ABI
- never expose C++ object layout
- never require Odin to know about Vulkan internals

Current limitation: individual mesh/material/texture create/destroy functions are declared but return `POR_ERROR_UNSUPPORTED_FEATURE` in the real wrapper until the resource manager grows stable per-resource lifetime semantics.

## Stage 4 — Mock Host Test

Before using Odin, test from C or C++ as if it were an external host.

Goal:

Validate the ABI without adding cross-language complexity.

## Stage 5 — Odin Mock Renderer

Separate Odin project.

The Odin project first talks to a mock C renderer API that only validates frame packets and prints stats.

Goal:

Learn Odin and build engine/editor concepts without destabilizing the Vulkan renderer.

## Stage 6 — Real Odin Host Integration

Only after:

- internal frame packet exists
- renderer handles exist
- C header exists
- C wrapper is tested
- single-threaded host flow works
- resource lifetime rules are explicit

## Naming Direction

Use a stable API prefix.

Recommended prefix:

```c
por_
```

Example:

```c
por_renderer_create
por_renderer_destroy
por_renderer_submit_frame
por_renderer_get_stats
```

This avoids generic symbol names like `renderer_create` that may collide in larger engine/editor projects.

## Basic Type Direction

```c
typedef struct por_renderer_t* por_renderer_handle;

typedef uint32_t por_mesh_handle;
typedef uint32_t por_texture_handle;
typedef uint32_t por_material_handle;
typedef uint32_t por_shader_handle;
```

For first implementation, `uint32_t` handles are acceptable.

Later, if deletion/hot reload/resource reuse becomes serious, move to generation handles:

```c
typedef struct por_resource_handle {
    uint32_t index;
    uint32_t generation;
} por_resource_handle;
```

Do not start with over-engineered handles until the renderer actually needs it.

## Error Handling Direction

C ABI functions should not throw. They should return explicit result codes.

```c
typedef enum por_result {
    POR_SUCCESS = 0,
    POR_ERROR_UNKNOWN = 1,
    POR_ERROR_INVALID_ARGUMENT = 2,
    POR_ERROR_UNSUPPORTED_FEATURE = 3,
    POR_ERROR_DEVICE_LOST = 4,
    POR_ERROR_RESOURCE_NOT_FOUND = 5,
    POR_ERROR_OUT_OF_MEMORY = 6,
} por_result;
```

For create functions:

```c
por_result por_renderer_create(
    const por_renderer_create_info* info,
    por_renderer_handle* out_renderer);
```

For detailed diagnostics:

```c
const char* por_renderer_get_last_error(por_renderer_handle renderer);
```

Rules:

- returned error string is owned by the renderer
- valid until next API call on the same renderer
- never require caller to free it
- do not use exceptions across ABI

## Memory Ownership Rules

### General Rule

The side that allocates an object destroys it.

### Caller-Owned Input

For input pointers:

```c
por_result por_renderer_submit_frame(
    por_renderer_handle renderer,
    const por_frame_packet* packet);
```

The packet memory is owned by the caller. Unless explicitly documented otherwise, it only needs to remain valid for the duration of the call.

### Renderer-Owned Resources

For renderer-created resources:

```c
por_mesh_handle mesh;
por_renderer_load_mesh(renderer, path, &mesh);
por_renderer_unload_mesh(renderer, mesh);
```

The host receives only a handle. It never owns the GPU object directly.

### Output Buffers

For stats/debug info, prefer caller-provided output structs:

```c
por_result por_renderer_get_stats(
    por_renderer_handle renderer,
    por_renderer_stats* out_stats);
```

Avoid APIs where renderer allocates memory that Odin must free.

## Window Ownership Direction

There are two models.

## Model A — Renderer-Owned Window

```c
por_result por_renderer_create_windowed(
    const por_windowed_renderer_create_info* info,
    por_renderer_handle* out_renderer);
```

The C++ renderer creates SDL window, Vulkan surface, and swapchain.

Pros:

- easier first integration
- fewer platform-specific ABI types
- current repo already works this way conceptually

Cons:

- Odin editor has weaker lifecycle control
- harder to make a commercial-style editor shell
- renderer owns more application concerns

## Model B — Host-Owned Window

```c
por_result por_renderer_create_for_native_window(
    const por_native_window_info* window_info,
    por_renderer_handle* out_renderer);
```

Odin owns the editor window. C++ renderer creates Vulkan surface/swapchain against the supplied native handle.

Pros:

- better long-term editor architecture
- Odin owns app lifecycle
- renderer behaves like a backend

Cons:

- harder cross-platform ABI
- native handle types differ per platform
- more platform-specific testing required

## Recommendation

For first real integration, support Model A or a restricted host-window model only.

Long-term, Model B is the better engine/editor architecture.

Do not block renderer progress on this decision now.

## Future Native Window Info Sketch

This is not final, but it shows the intended style.

```c
typedef enum por_platform {
    POR_PLATFORM_UNKNOWN = 0,
    POR_PLATFORM_WIN32 = 1,
    POR_PLATFORM_MACOS = 2,
    POR_PLATFORM_X11 = 3,
    POR_PLATFORM_WAYLAND = 4,
} por_platform;

typedef struct por_native_window_info {
    por_platform platform;
    void* window_handle;
    void* display_handle;
    uint32_t width;
    uint32_t height;
} por_native_window_info;
```

This may need platform-specific refinement later.

## Renderer Creation Sketch

```c
typedef struct por_renderer_create_info {
    uint32_t width;
    uint32_t height;
    uint32_t enable_validation;
    uint32_t preferred_gpu_index;
    uint32_t flags;
} por_renderer_create_info;

por_result por_renderer_create(
    const por_renderer_create_info* info,
    por_renderer_handle* out_renderer);

void por_renderer_destroy(por_renderer_handle renderer);
```

Rules:

- `por_renderer_destroy(NULL)` should be safe or documented as invalid.
- create returns error code and writes `out_renderer` only on success.
- renderer owns Vulkan resources after creation.

## Resize Sketch

```c
por_result por_renderer_resize(
    por_renderer_handle renderer,
    uint32_t width,
    uint32_t height);
```

Rules:

- width/height zero should be treated as minimized/suspended
- resize must refresh swapchain-dependent resources
- future passes such as SMAA/VRS must plug into this path

## Resource Loading Sketch

Initial path-based API:

```c
por_result por_renderer_load_mesh(
    por_renderer_handle renderer,
    const char* path,
    por_mesh_handle* out_mesh);

por_result por_renderer_load_texture(
    por_renderer_handle renderer,
    const char* path,
    uint32_t flags,
    por_texture_handle* out_texture);

por_result por_renderer_create_material(
    por_renderer_handle renderer,
    const por_material_desc* desc,
    por_material_handle* out_material);
```

Unload:

```c
por_result por_renderer_unload_mesh(
    por_renderer_handle renderer,
    por_mesh_handle mesh);

por_result por_renderer_unload_texture(
    por_renderer_handle renderer,
    por_texture_handle texture);

por_result por_renderer_destroy_material(
    por_renderer_handle renderer,
    por_material_handle material);
```

Future memory-based API:

```c
por_result por_renderer_create_mesh(
    por_renderer_handle renderer,
    const por_mesh_desc* desc,
    por_mesh_handle* out_mesh);
```

Do path-based loading first. Add memory-based loading after the asset pipeline becomes clearer.

## Material Description Sketch

```c
typedef struct por_material_desc {
    float base_color_factor[4];
    float metallic_factor;
    float roughness_factor;
    float alpha_cutoff;
    uint32_t flags;
    por_texture_handle albedo_texture;
    por_texture_handle normal_texture;
    por_texture_handle metallic_roughness_texture;
} por_material_desc;
```

Material handles should remain renderer-owned.

Odin may own editor material metadata, but the GPU material resource belongs to C++.

## Frame Packet Sketch

The frame packet is the most important API concept.

```c
typedef struct por_camera_data {
    float view[16];
    float projection[16];
    float position[3];
    float near_z;
    float far_z;
} por_camera_data;

typedef struct por_directional_light_data {
    float direction[3];
    float intensity;
    float color[3];
    float ambient;
} por_directional_light_data;

typedef struct por_draw_command {
    float transform[16];
    por_mesh_handle mesh;
    por_material_handle material;
    uint32_t flags;
} por_draw_command;

typedef struct por_shadow_settings {
    float csm_lambda;
    uint32_t debug_cascades;
    uint32_t filter_mode;
    float pcf_spread_radius;
    float vsm_bleed_reduction;
} por_shadow_settings;

typedef struct por_tonemap_settings {
    uint32_t mode;
    float exposure;
    uint32_t split_screen;
    uint32_t split_right_mode;
} por_tonemap_settings;

typedef struct por_sky_settings {
    uint32_t enabled;
    uint32_t mode;
} por_sky_settings;

typedef struct por_debug_view_settings {
    uint32_t wireframe;
    uint32_t show_normals;
    uint32_t reserved0;
    uint32_t reserved1;
} por_debug_view_settings;

typedef struct por_frame_packet {
    por_camera_data camera;
    por_directional_light_data sun;
    por_shadow_settings shadows;
    por_tonemap_settings tonemap;
    por_sky_settings sky;
    por_debug_view_settings debug;
    const por_draw_command* draws;
    uint32_t draw_count;
} por_frame_packet;
```

Submission:

```c
por_result por_renderer_submit_frame(
    por_renderer_handle renderer,
    const por_frame_packet* packet);
```

Rules:

- frame packet is caller-owned
- renderer copies any data it needs after the call
- no pointers inside the packet are retained unless explicitly documented
- draw list must not be mutated while the call is active

## Coordinate and Matrix Rules

The API must document matrix conventions before real integration.

Initial convention recommendation:

- matrices are 4x4 floats
- column-major memory layout, matching GLM default
- transforms are object-to-world
- view matrix is world-to-view
- projection matrix must match renderer/Vulkan clip expectations
- renderer should clearly state whether it expects reverse-Z projection

Because the current renderer uses reverse-Z depth internally, the frame packet must either:

1. require the host to provide a compatible projection matrix, or
2. expose a helper/API convention so the renderer builds projection from camera parameters

Safer future API:

```c
typedef struct por_camera_desc {
    float position[3];
    float orientation_quat[4];
    float vertical_fov_degrees;
    float near_z;
    float far_z;
} por_camera_desc;
```

But for first integration, explicit matrices are more flexible.

## Stats API Sketch

```c
typedef struct por_renderer_stats {
    uint32_t draw_calls;
    uint32_t triangle_count;
    uint32_t mesh_count;
    uint32_t material_count;
    uint32_t texture_count;
    uint64_t texture_memory_bytes;
    float gpu_frame_ms;
    float pass_shadow_ms;
    float pass_blur_ms;
    float pass_scene_ms;
    float pass_tonemap_ms;
    float pass_imgui_ms;
} por_renderer_stats;

por_result por_renderer_get_stats(
    por_renderer_handle renderer,
    por_renderer_stats* out_stats);
```

The C++ viewer may keep richer ImGui-only diagnostics. The C ABI should expose stable, plain stats.

## Feature Capability API Sketch

The renderer should expose capabilities instead of requiring the host to know Vulkan details.

```c
typedef struct por_renderer_caps {
    uint32_t vulkan_major;
    uint32_t vulkan_minor;
    uint32_t supports_vrs;
    uint32_t supports_dynamic_rendering_local_read;
    uint32_t supports_push_descriptor;
    uint32_t supports_hdr_swapchain;
    char gpu_name[256];
} por_renderer_caps;

por_result por_renderer_get_caps(
    por_renderer_handle renderer,
    por_renderer_caps* out_caps);
```

This is especially important for VRS because it may not exist on every target platform.

## Screenshot API Sketch

```c
por_result por_renderer_request_screenshot(
    por_renderer_handle renderer,
    const char* output_path);
```

Rules:

- output path is copied during the call
- screenshot completes after a future rendered frame unless documented otherwise
- stats/error API should report failure if capture fails

## Threading Direction

Initial C ABI should be single-threaded from the host perspective.

Recommended first flow:

```txt
host update
host builds frame packet
por_renderer_submit_frame(renderer, &packet)
```

Only after single-threaded API works should the project consider:

```txt
host thread builds packet N+1
render thread consumes packet N
```

Requirements before threading:

- immutable frame packets
- double-buffered packet storage
- explicit resource lifetime rules
- deferred destruction
- no renderer mutation from multiple host threads unless documented

Possible future threading API:

```c
por_result por_renderer_start_thread(por_renderer_handle renderer);
por_result por_renderer_stop_thread(por_renderer_handle renderer);
por_result por_renderer_enqueue_frame(por_renderer_handle renderer, const por_frame_packet* packet);
```

Do not start here.

## Odin Usage Direction

The future Odin host should initially own:

- engine/editor main loop
- scene graph
- transform hierarchy
- asset metadata
- object selection
- editor panels
- frame packet generation

The future Odin host should not initially own:

- Vulkan setup
- GPU memory
- shader compilation
- renderer pass sequencing
- low-level swapchain management
- renderer resource internals

## Odin Mock Prototype Recommendation

Create a separate project later, not inside this repo.

Possible layout:

```txt
OdinEnginePrototype/
  src/
    main.odin
    scene/
    transform/
    editor/
    renderer_api/
  mock_renderer/
    por_renderer_api.h
    por_renderer_mock.c
```

Mock renderer behavior:

- accepts `por_frame_packet`
- validates draw count
- validates handles are non-zero or known
- prints fake stats
- returns deterministic errors for bad input

This allows learning Odin without destabilizing Vulkan work.

## C++ Viewer Compatibility

The C++ viewer should remain.

Long-term, there can be two frontends:

```txt
C++ Research Viewer -> internal C++ renderer API
Odin Engine Host    -> C ABI -> same renderer backend
```

The C++ viewer is useful for:

- fast renderer-only iteration
- graphics debugging
- GPU timing
- RenderDoc captures
- shader experiments
- visual comparisons

Do not remove it just because Odin is planned.

## Implementation Preconditions

Before treating the real C ABI as stable, these must exist or be actively introduced:

1. grouped renderer settings structs
2. internal `RenderFramePacket`
3. explicit resize method
4. stable resource handle create/destroy model
5. at least one extracted pass or clearer pass ownership
6. clearer viewer/backend split
7. documented frame flow
8. documented resource lifetime rules

## First Stable Implementation Slice

The draft already contains lifecycle/frame/stat functions. The first stable slice should remain intentionally small.

Recommended first slice:

```c
por_renderer_create
por_renderer_destroy
por_renderer_resize
por_renderer_get_stats
por_renderer_render_frame
```

No Odin yet. Resource creation should only become stable after deletion/reuse/lifetime rules are explicit.

Second slice:

```c
por_renderer_create_material
por_renderer_create_mesh
por_renderer_create_texture
por_renderer_render_frame with draw/scene submission
```

This sequence reduces risk.

## Open Questions

These should remain open until implementation pressure exists:

1. Should the first real C ABI own its own SDL window or consume a host window?
2. Should frame packet submission also present, or should present be a separate call?
3. Should resource loading be synchronous first, or immediately return async handles?
4. Should the API use matrix packets or camera descriptors?
5. How should shader hot reload be exposed, if at all?
6. Should ImGui remain C++ viewer only, or should renderer expose debug data for Odin UI?
7. What is the minimum useful Odin editor prototype?

## Current Recommendation

For the near term:

1. Keep developing the renderer in C++.
2. Keep the C++ viewer as the research/debug shell.
3. Harden `RendererInstance`, `RenderScenePacket`, and resource handles before adding more feature scope.
4. Add render feature work only in ways that support current pass ownership.
5. Keep Odin integration on paper only.
6. Build an Odin mock host later, separate from this repository.
7. Connect a real Odin host only after resource lifetime and ABI tests are stable.

## Next Recommended Document

`docs/decisions/ADR-0003-cpp-core-odin-host-boundary.md`

Purpose:

Record the key project decision in ADR format so future context windows do not reopen the same architectural question.
