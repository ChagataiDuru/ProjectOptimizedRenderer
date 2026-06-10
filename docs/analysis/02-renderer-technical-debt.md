# 02 — Renderer Technical Debt Map

Date: 2026-06-10

## Purpose

This document lists current and expected technical debt in ProjectOptimizedRenderer.

The goal is not to shame the codebase. The renderer is still in a productive research/prototype phase. Some debt is acceptable because it helps rapid feature iteration. The goal is to identify which debt blocks the long-term direction:

- C++ Vulkan renderer remains the core project.
- The current C++ viewer remains useful as a research/debug shell.
- A future Odin engine/editor host may exist as a separate project connected through C ABI.
- The renderer should eventually become a stable backend, not only a single demo application.

## Severity Scale

| Severity | Meaning |
|---|---|
| Critical | Will directly block future architecture or cause recurring breakage soon. |
| High | Will become expensive if more features are added before cleanup. |
| Medium | Should be handled during adjacent work, but does not block immediate progress. |
| Low | Acceptable prototype compromise or documentation issue. |

## Timing Scale

| Timing | Meaning |
|---|---|
| Now | Address before major new renderer features. |
| Soon | Address during next architecture pass or before Odin boundary work. |
| Later | Address when the relevant system is expanded. |
| Do Not Fix Yet | Known imperfection, but premature abstraction would hurt more than help. |

---

# TD-001 — `Renderer` Is Becoming a Monolith

Severity: **High**  
Timing: **Soon**

## Current State

`Renderer` currently owns or coordinates nearly everything after Vulkan/swapchain setup:

- model loading
- model reload
- mesh upload
- texture upload
- fallback texture creation
- material descriptor creation
- camera/light UBOs
- PBR pipeline
- wireframe/normal variants
- shadow resources
- cascade matrices
- VSM moment images
- compute blur resources
- HDR target
- tone mapping pass
- sky pipeline
- GPU timing
- screenshots
- render statistics
- ImGui overlay hook
- frame begin/end orchestration

This is functional and acceptable for a research viewer, but it is not a clean backend boundary.

## Why It Matters

Future features such as SMAA, perceptual VRS, IBL, transparency, async loading, material editing, and external Odin integration will all naturally try to attach to `Renderer`. Without decomposition, every system becomes coupled to every other system.

## Risk

- difficult resource lifetime reasoning
- large resize/reload bugs
- hard-to-test pass interactions
- hard-to-expose C ABI
- renderer becomes a demo app instead of backend

## Recommended Direction

Do not split everything immediately. Instead, introduce boundaries gradually:

```txt
Renderer
  RenderResourceManager
  RenderScene / RenderWorld
  Pass objects or lightweight render graph
  Debug/profiling module
```

First extraction candidates:

1. `RenderResourceManager`
2. `TonemapPass`
3. `ShadowPass`
4. `SkyPass`
5. `RendererStats` / debug query surface

## Not To Do Yet

Do not build a large commercial-style render graph immediately. A small pass abstraction is enough until the renderer has more post-processing and VRS/SMAA dependencies.

---

# TD-002 — `main.cpp` Is Both Viewer, Editor, and Engine Shell

Severity: **High**  
Timing: **Soon**

## Current State

`src/main.cpp` owns:

- app startup
- logger setup
- window creation
- Vulkan context creation
- swapchain creation
- renderer creation
- ImGui manager creation
- camera setup
- frame timing
- SDL event handling
- mouse capture
- all ImGui panel registration
- light/shadow/tonemap UI state
- file dialogs
- deferred model reloads
- deferred HDR panorama loads
- render loop

This is common in prototypes, but it makes the viewer look like the future engine shell.

## Why It Matters

The future direction explicitly says the Odin engine/editor host is separate. Therefore, `main.cpp` should be treated as a disposable C++ research viewer, not the architectural center of the engine.

## Risk

- editor features become trapped in C++ viewer code
- renderer API remains shaped around local UI callbacks
- future Odin host must duplicate or reverse-engineer viewer behavior
- harder to keep renderer backend independent

## Recommended Direction

Eventually split C++ viewer into:

```txt
ViewerApp
  - lifecycle
  - input
  - camera
  - UI panel registration

RendererBackend
  - no dependency on viewer-specific behavior
```

A low-cost first step is to move ImGui panel registration out of `main.cpp` into a viewer-specific file:

```txt
src/viewer/ViewerPanels.cpp
include/viewer/ViewerPanels.h
```

## Not To Do Yet

Do not remove the C++ viewer. It is valuable for graphics research and GPU debugging.

---

# TD-003 — No Explicit Frame Packet Abstraction

Severity: **Critical for Odin Integration, Medium for Current Viewer**  
Timing: **Soon, before C ABI implementation**

## Current State

The renderer currently receives state through direct calls such as:

- set camera matrices
- set camera frustum
- set light parameters
- set CSM lambda
- set shadow filter mode
- set tone map params
- set sky mode
- reload model

This is fine for the current viewer, but it is not the right external host API.

## Why It Matters

A future Odin host should submit a stable, plain-data frame snapshot. It should not call many renderer mutators per object or per subsystem.

## Risk

- high FFI call overhead
- unstable host/renderer contract
- renderer state bugs from partial updates
- difficult multithreaded handoff

## Recommended Direction

Introduce a C++ internal frame packet before exposing C ABI:

```cpp
struct RenderFramePacket {
    CameraData camera;
    DirectionalLightData sun;
    TonemapSettings tonemap;
    ShadowSettings shadows;
    SkySettings sky;
    std::span<const DrawCommand> draws;
};
```

Then the viewer can build this packet internally. Later, the C ABI can expose a POD equivalent.

## Not To Do Yet

Do not force every current renderer feature into the frame packet immediately. Start with camera, light, draw commands, and debug toggles.

---

# TD-004 — Renderer Assumes a Loaded glTF Model as the Scene

Severity: **High**  
Timing: **Soon, before engine host integration**

## Current State

The renderer loads Sponza by default and treats the loaded model as the render scene. Model reload replaces renderer-owned scene data.

## Why It Matters

An engine renderer should render many objects, each potentially referencing shared mesh/material resources with different transforms. The engine/editor host should own object hierarchy and transforms.

## Risk

- difficult multi-object rendering model
- no stable mesh/material handle API
- renderer and asset loader remain coupled
- future Odin scene graph cannot map cleanly onto renderer state

## Recommended Direction

Separate these concepts:

```txt
Asset import model
  - loaded glTF CPU data

Renderer resources
  - mesh handles
  - material handles
  - texture handles

Renderable scene
  - draw commands referencing handles
```

Short-term bridge:

- keep glTF loading in renderer
- but have loadModel produce one or more internal resource handles
- have rendering consume a generated draw list instead of directly iterating the model as the scene

## Not To Do Yet

Do not design a complete asset database inside the C++ renderer. That belongs to the future engine/editor host or external tooling.

---

# TD-005 — Manual Descriptor and Pipeline Growth

Severity: **High**  
Timing: **Soon**

## Current State

The renderer manually creates many descriptors and pipelines for:

- PBR
- material textures
- shadow maps
- VSM moments
- VSM blur
- HDR target
- tone mapping
- sky panorama
- ImGui

This works, but every new pass adds more manual setup, update, resize, and shutdown logic.

## Why It Matters

SMAA and VRS will require additional passes and resources:

- SMAA edge detection
- SMAA blend weight calculation
- SMAA neighborhood blending
- VRS classification image
- optional debug visualization
- additional transient images and descriptors

Without better organization, pass setup will become fragile.

## Risk

- duplicated descriptor code
- resize bugs
- mismatched descriptor layouts
- difficult pass ordering
- hard-to-debug resource lifetime errors

## Recommended Direction

Introduce pass-local ownership:

```cpp
class TonemapPass {
public:
    void init(...);
    void resize(...);
    void record(...);
    void shutdown();
};
```

Do this incrementally, starting with post-processing passes because they are easier to isolate than the scene pass.

## Not To Do Yet

Do not introduce an overly abstract material/pipeline compiler yet. First make pass ownership explicit.

---

# TD-006 — No Lightweight Render Graph / Pass Dependency Model

Severity: **Medium Now, High Later**  
Timing: **Later, after SMAA/VRS pressure increases**

## Current State

Pass sequencing is implicit inside renderer code. Resources such as HDR target, depth, shadow maps, VSM moments, and swapchain images are manually transitioned/used.

## Why It Matters

A modern renderer with native anti-aliasing, shadow passes, HDR, post-processing, VRS, and debug overlays needs clear pass dependencies.

## Risk

- barrier/layout transition mistakes
- transient image lifetime confusion
- hard to reorder passes
- hard to insert optional debug passes
- hard to reason about performance

## Recommended Direction

Do not jump immediately to a full render graph. First define a minimal pass declaration model:

```cpp
struct RenderPassDesc {
    const char* name;
    ImageHandle inputs[];
    ImageHandle outputs[];
    bool resizeDependent;
};
```

Then evolve into a real graph if useful.

## Not To Do Yet

Do not copy Unreal/RenderDoc-scale render graph complexity. The project needs a small renderer-owned pass graph, not an engine-scale rendering framework yet.

---

# TD-007 — C++ / GLSL Interface Contracts Are Manual

Severity: **Medium**  
Timing: **Soon, during shader system cleanup**

## Current State

Shader UBOs, push constants, descriptor bindings, and material layouts are manually mirrored between C++ and GLSL.

Examples include:

- camera UBO
- light UBO
- shadow UBO
- material push constants
- tonemap push constants
- descriptor set layout binding numbers

## Why It Matters

Manual layout sync is manageable now, but becomes risky as the renderer gains more passes and external API types.

## Risk

- silent shader/C++ mismatch
- hard-to-debug rendering corruption
- broken push constant offsets
- descriptor binding mismatch during refactor

## Recommended Direction

Short-term:

- centralize binding constants in C++ headers and GLSL include-like generated files if possible
- add comments that state exact binding and set ownership
- static assert C++ struct sizes and alignment

Medium-term:

- generate shader interface headers
- or use SPIR-V reflection to validate pipeline layouts during development

## Not To Do Yet

Do not build a full shader compiler framework before SMAA/VRS are implemented. Use targeted validation first.

---

# TD-008 — Resize Handling Is Spread Across Multiple Resources

Severity: **Medium**  
Timing: **Soon**

## Current State

Resize affects:

- swapchain
- depth image
- HDR target
- tone map descriptor set
- viewport/scissor
- camera aspect ratio
- possibly ImGui frame state
- future SMAA images
- future VRS image

## Why It Matters

As post-processing grows, resize bugs become common.

## Risk

- stale descriptors after image recreation
- invalid image views
- mismatched viewport size
- wrong camera projection
- swapchain recreation instability

## Recommended Direction

Define a common resize pathway:

```cpp
void Renderer::resize(uint32_t width, uint32_t height);
```

Internally this calls:

```txt
Depth.resize
HdrTarget.resize
TonemapPass.resize
SmaaPass.resize
VrsPass.resize
```

## Not To Do Yet

Do not over-design multi-window or multiple viewport support yet.

---

# TD-009 — Debug UI Directly Shapes Renderer API

Severity: **Medium**  
Timing: **Later**

## Current State

The viewer UI writes directly into renderer settings through specific setters and reads renderer stats directly.

This is fine for C++ ImGui, but not ideal for external engine/editor API.

## Why It Matters

A future Odin editor should not need to understand renderer internals or C++ UI callbacks.

## Risk

- renderer API becomes UI-driven instead of data-driven
- settings become scattered
- host integration needs many small calls

## Recommended Direction

Introduce grouped settings structs:

```cpp
struct ShadowSettings;
struct TonemapSettings;
struct SkySettings;
struct DebugViewSettings;
```

Viewer UI edits these settings. Renderer consumes them per frame or through explicit apply calls.

## Not To Do Yet

Do not remove convenient ImGui controls. They are important for research iteration.

---

# TD-010 — Resource Lifetime Depends on Full `vkDeviceWaitIdle` During Reload

Severity: **Medium**  
Timing: **Later, before async loading**

## Current State

Model reload drains the GPU with `vkDeviceWaitIdle` before destroying and recreating model-dependent resources.

## Why It Matters

This is safe and simple, but it will not scale to streaming, async texture loading, or editor hot reload.

## Risk

- frame hitching
- poor editor interactivity during reload
- difficult background asset loading
- incompatible with future streaming model

## Recommended Direction

Keep this for now. Later introduce deferred destruction by frame index:

```txt
destroy resource only after N frames-in-flight have completed
```

Then async upload can use staging queues and delayed GPU visibility.

## Not To Do Yet

Do not implement full async streaming until the renderer resource handle model is clearer.

---

# TD-011 — Asset Pipeline Is Not Yet Defined

Severity: **Medium**  
Timing: **Later**

## Current State

The renderer loads glTF and texture files directly. This is good for research and test scenes.

## Why It Matters

A future engine/editor needs an asset pipeline:

- import source assets
- generate runtime asset formats
- build thumbnails/previews
- track dependencies
- manage hot reload
- maybe convert textures to KTX2/BasisU
- maybe optimize meshes offline

## Risk

- renderer becomes responsible for editor asset database concerns
- runtime loading path becomes slow
- hard to support packaged projects later

## Recommended Direction

Keep direct glTF loading in the renderer for now.

Use Odin, if desired, for standalone offline tools first:

- texture packer/converter
- mesh optimizer wrapper
- asset manifest generator
- scene description exporter

This gives useful Odin practice without creating runtime FFI instability.

## Not To Do Yet

Do not make Odin the runtime asset manager before the renderer has stable resource handles.

---

# TD-012 — No Stable External Handle System

Severity: **High for Odin Integration**  
Timing: **Soon, before C ABI**

## Current State

Renderer resources are mostly C++ objects and vectors owned internally.

## Why It Matters

A C ABI cannot expose internal object pointers safely. The host needs opaque handles.

## Risk

- accidental lifetime bugs
- external host depends on vector indices that may move
- impossible to validate stale resources cleanly

## Recommended Direction

Introduce internal handle types:

```cpp
using MeshHandle = uint32_t;
using TextureHandle = uint32_t;
using MaterialHandle = uint32_t;
```

Later upgrade to generation handles:

```cpp
struct Handle {
    uint32_t index;
    uint32_t generation;
};
```

For the first API, simple `uint32_t` handles are acceptable if deletion is limited.

## Not To Do Yet

Do not implement a full ECS-style handle registry unless real deletion/reload pressure appears.

---

# TD-013 — Feature Roadmap Is Not Yet Connected to Architecture Roadmap

Severity: **Medium**  
Timing: **Now**

## Current State

Renderer features exist or are planned:

- CSM/VSM shadows
- HDR tone mapping
- AgX / PBR Neutral
- sky
- SMAA
- VRS
- async texture loading
- future engine API

But the architecture documents should explicitly connect feature phases to required structural changes.

## Why It Matters

Without this, it is easy to add features in the wrong order and create avoidable refactors.

## Recommended Direction

Create a roadmap document that separates:

1. research features
2. backend architecture work
3. engine/API preparation
4. Odin host experiments

Recommended file:

```txt
docs/analysis/03-feature-roadmap.md
```

---

# TD-014 — Documentation Is Starting Late But Recoverable

Severity: **Low**  
Timing: **Now**

## Current State

The repo has `AGENTS.md` and a loose `documentation.md`, but project decisions, architecture rationale, and future engine direction were not clearly structured before this analysis folder.

## Why It Matters

The project will be developed across long context windows and multiple sessions. Durable documents are necessary so future work does not restart from memory.

## Recommended Direction

Maintain three documentation types:

```txt
docs/analysis/    current state and planning
docs/research/    technique research and notes
docs/decisions/   ADR-style decisions
```

## Not To Do Yet

Do not try to write perfect documentation for every file. Prioritize decisions and architecture boundaries.

---

# Priority Summary

## Fix / Design Soon

1. `Renderer` responsibility split direction
2. `main.cpp` viewer-shell separation
3. frame packet abstraction
4. render resource handles
5. grouped renderer settings
6. pass ownership around post-processing/shadows
7. feature roadmap tied to architecture roadmap

## Keep For Now

1. C++ research viewer
2. direct glTF loading
3. `vkDeviceWaitIdle` model reload safety
4. ImGui debug panels
5. simple descriptor/pipeline setup where still manageable

## Avoid For Now

1. full Odin runtime integration
2. full render graph framework
3. full asset database
4. multi-window editor support
5. engine-wide ECS inside this renderer repo
6. exposing Vulkan handles through the future C ABI

## Next Recommended Document

`docs/analysis/03-feature-roadmap.md`

Purpose:

Define the renderer roadmap in terms of research milestones and architecture milestones, so future feature work supports the hybrid goal instead of fighting it.
