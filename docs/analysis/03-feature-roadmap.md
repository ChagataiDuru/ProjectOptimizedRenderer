# 03 — Feature and Architecture Roadmap

Date: 2026-06-10

## Purpose

This document connects renderer feature development with architecture work.

The project direction is hybrid:

- short term: C++ Vulkan renderer research prototype and portfolio project
- long term: engine-ready renderer backend
- future optional host: separate Odin engine/editor connected through a C ABI

The goal is to avoid two failure modes:

1. adding renderer features so quickly that the architecture becomes unmaintainable
2. over-engineering engine architecture before the renderer research value is proven

## Strategic Thesis

ProjectOptimizedRenderer should not become a generic clone of a commercial engine renderer.

The project should have a clear technical identity:

> A modern Vulkan renderer focused on native spatial quality, stable shadows, perceptual image quality, explicit GPU control, and engine-ready architecture without hiding the rendering model behind proprietary black-box abstractions.

Core pillars:

1. native rendering quality over upscaling-first design
2. explicit Vulkan 1.4 backend
3. stable cascaded shadow maps and VSM experimentation
4. HDR scene pipeline with AgX and PBR Neutral comparison
5. SMAA / spatial anti-aliasing path
6. perceptual VRS driven by image/scene metrics
7. cross-platform Windows + macOS support
8. future C ABI boundary for external engine/editor hosts

## Roadmap Tracks

The roadmap is split into four tracks.

```txt
Track A — Renderer Research Features
Track B — Renderer Architecture Stabilization
Track C — Tooling / Documentation / Portfolio Evidence
Track D — Future Odin Host Preparation
```

These tracks should advance together, but not at the same speed.

---

# Track A — Renderer Research Features

## A1 — Stabilize Existing Visual Baseline

Priority: **Immediate**

Current features already present:

- glTF/Sponza loading
- PBR direct lighting
- normal mapping
- HDR render target
- Reinhard / AgX / PBR Neutral tonemapping
- CSM shadows
- PCF
- VSM moments
- VSM blur
- procedural/HDR sky
- GPU timing
- render stats
- screenshots

Goal:

Make the current renderer baseline reproducible before adding more visual systems.

Deliverables:

- stable build instructions
- known-good test scene list
- screenshot capture convention
- render setting presets
- GPU timing baseline
- documented visual limitations

Success criteria:

- the same test scene can be run on macOS and Windows
- screenshots can be compared between tone mapping modes
- pass timings are visible and understandable
- visual regressions are easy to detect manually

## A2 — SMAA Path

Priority: **High**

Purpose:

Add a native spatial anti-aliasing solution that fits the project thesis: clear, stable, non-upscaled rendering.

Expected passes:

```txt
Scene HDR/LDR source
  -> SMAA edge detection
  -> SMAA blend weight calculation
  -> SMAA neighborhood blending
  -> tone mapping / present integration depending on chosen placement
```

Design questions:

1. Should SMAA operate before or after tone mapping?
2. Should the source be HDR linear, tonemapped linear, or swapchain-space LDR?
3. How should debug modes visualize edge detection and blend weights?
4. How should it interact with UI/ImGui overlays?

Architecture requirements:

- explicit post-process pass ownership
- resize-aware intermediate images
- descriptor update path for intermediate images
- debug toggles grouped into a settings struct
- GPU timer labels per SMAA pass

Do not implement SMAA by simply adding more loose code into `Renderer`.

Recommended first implementation:

```txt
SmaaPass
  - init
  - resize
  - recordEdgePass
  - recordBlendPass
  - recordNeighborhoodPass
  - shutdown
```

## A3 — Perceptual VRS

Priority: **High, after SMAA foundation**

Purpose:

Implement variable rate shading as a perceptual optimization path rather than a generic performance checkbox.

Expected concept:

- analyze luminance/contrast/edge structure
- produce shading rate image or equivalent control data
- reduce shading where perceptually safe
- preserve edges, silhouettes, highlights, text, and high-frequency regions

Design questions:

1. Which devices/platforms expose usable VRS support?
2. How should MoltenVK/macOS behave when VRS is unavailable?
3. Should VRS be feature-gated by runtime capabilities?
4. Should the renderer show a VRS debug overlay?
5. What visual metric proves that VRS is not just blurring quality?

Architecture requirements:

- device capability query layer
- optional pass enablement
- fallback path when unsupported
- settings struct for VRS mode/debug
- GPU timing around VRS classification
- screenshot comparisons with/without VRS

Important:

VRS should not be mandatory for renderer startup. It should be optional and capability-driven.

## A4 — Shadow Quality and Stability Pass

Priority: **Medium-High**

Current features:

- cascaded shadows
- PCF
- VSM
- cascade debug overlay
- shadow culling stats

Goals:

- reduce shimmering
- validate texel snapping
- document cascade split behavior
- tune VSM bleed reduction
- compare PCF vs VSM visually and by timing
- define shadow presets

Potential deliverables:

```txt
Shadow Quality Presets:
  - Hard
  - PCF Low
  - PCF High
  - VSM Fast
  - VSM Quality
```

Architecture requirements:

- `ShadowSettings` struct
- pass-local shadow ownership
- debug output for cascade splits
- stable GPU timing per shadow path

## A5 — HDR / Tone Mapping Research

Priority: **Medium**

Current features:

- HDR offscreen target
- Reinhard
- AgX
- PBR Neutral
- exposure control
- split-screen comparison

Goals:

- document differences between operators
- define why AgX/PBR Neutral are included
- add reproducible screenshots
- expose tone mapping presets
- consider histogram/luminance tools later

Architecture requirements:

- `TonemapSettings` struct
- `TonemapPass` extraction candidate
- consistent color-space comments and validation

## A6 — Image-Based Lighting / Environment Lighting

Priority: **Later**

Current sky support is present, but full IBL is not the immediate priority.

Future goals:

- diffuse irradiance
- specular prefiltered environment
- BRDF LUT
- material response validation
- HDR panorama as lighting source

Do this after the pass/resource boundaries are cleaner.

---

# Track B — Renderer Architecture Stabilization

## B1 — Viewer Shell Separation

Priority: **Soon**

Problem:

`main.cpp` currently owns too much viewer/editor behavior.

Goal:

Keep the C++ research viewer, but make it conceptually separate from renderer backend.

First low-risk move:

```txt
include/viewer/ViewerPanels.h
src/viewer/ViewerPanels.cpp
```

Move panel registration and viewer-only ImGui logic out of `main.cpp`.

Non-goal:

Do not delete the C++ viewer.

## B2 — Renderer Settings Structs

Priority: **Soon**

Problem:

Renderer state is currently set through many separate methods and UI-specific calls.

Introduce grouped settings:

```cpp
struct ShadowSettings {
    float csmLambda;
    bool showCascadeDebug;
    int shadowFilterMode;
    float pcfSpreadRadius;
    float vsmBleedReduction;
};

struct TonemapSettings {
    int mode;
    float exposure;
    bool splitScreen;
    int splitRightMode;
};

struct SkySettings {
    bool enabled;
    int mode;
};

struct DebugViewSettings {
    bool wireframe;
    bool showNormals;
};
```

Benefits:

- easier frame packet design
- easier future C ABI mapping
- fewer scattered setters
- cleaner UI-to-renderer handoff

## B3 — Internal Render Frame Packet

Priority: **Before C ABI work**

Goal:

Create a C++ internal version of the future external frame packet.

Initial contents:

```cpp
struct RenderFramePacket {
    CameraData camera;
    DirectionalLightData sun;
    ShadowSettings shadows;
    TonemapSettings tonemap;
    SkySettings sky;
    DebugViewSettings debug;
};
```

Later contents:

```cpp
std::span<const DrawCommand> draws;
```

Important:

This should first be used by the C++ viewer. Once stable, expose a POD C version for Odin.

## B4 — Resource Handle Model

Priority: **Before multi-object/Odin integration**

Goal:

Move from renderer-internal model-as-scene to handle-based render resources.

Initial handles:

```cpp
using MeshHandle = uint32_t;
using MaterialHandle = uint32_t;
using TextureHandle = uint32_t;
```

Later handles:

```cpp
struct ResourceHandle {
    uint32_t index;
    uint32_t generation;
};
```

First target:

Have loaded glTF generate internal mesh/material handles and a draw list.

This allows the current viewer to keep loading Sponza while moving toward engine-ready draw submission.

## B5 — Pass Ownership Extraction

Priority: **During SMAA/post-processing work**

Extract pass ownership gradually.

Best first candidates:

1. `TonemapPass`
2. `SkyPass`
3. `SmaaPass`
4. `ShadowPass`

Each pass should own:

- pipelines
- pipeline layout
- descriptor set layout
- descriptor pool/set if pass-local
- resize-dependent images if applicable
- `record(...)`
- `shutdown()`

Non-goal:

Do not build a large render graph before pass extraction provides real pressure.

## B6 — Minimal Render Graph / Pass Dependency Layer

Priority: **Later**

Trigger point:

Implement this only when SMAA/VRS/post-processing chains make manual pass ordering painful.

Minimum useful design:

```cpp
struct RenderPassNode {
    const char* name;
    void (*record)(RenderContext&);
    Span<ImageUse> reads;
    Span<ImageUse> writes;
};
```

Expected benefits:

- clearer barriers/layout transitions
- easier optional passes
- better GPU capture labels
- easier transient image reuse later

## B7 — Shader Interface Validation

Priority: **Medium**

Short-term:

- centralize binding constants
- static assert C++ struct layout
- document shader set/binding ownership

Medium-term:

- add SPIR-V reflection validation for debug builds
- or generate C++/GLSL interface headers from a shared schema

Do not overbuild this before pass boundaries are clearer.

---

# Track C — Tooling, Documentation, Portfolio Evidence

## C1 — Build and Run Documentation

Priority: **Immediate**

Create a real `README.md` or `docs/getting-started.md`.

Must include:

- repository goal
- supported platforms
- dependencies
- Conan install command
- CMake presets
- shader compilation behavior
- asset expectations
- screenshots
- known limitations

## C2 — Technique Research Notes

Priority: **Ongoing**

Recommended files:

```txt
docs/research/smaa.md
docs/research/cascaded-shadow-maps.md
docs/research/variance-shadow-maps.md
docs/research/tone-mapping-agx-pbr-neutral.md
docs/research/perceptual-vrs.md
docs/research/native-rendering-vs-upscaling.md
```

Each research note should include:

- problem statement
- why the technique matters
- implementation approach
- tradeoffs
- validation method
- screenshots/metrics to capture

## C3 — Architecture Decision Records

Priority: **Soon**

Recommended files:

```txt
docs/decisions/ADR-0001-vulkan-14-baseline.md
docs/decisions/ADR-0002-cmake-conan-build.md
docs/decisions/ADR-0003-cpp-core-odin-host-boundary.md
docs/decisions/ADR-0004-dynamic-rendering.md
docs/decisions/ADR-0005-cpp-research-viewer-stays.md
```

Purpose:

Avoid re-litigating important decisions across context windows.

## C4 — Screenshot and Metrics Protocol

Priority: **Soon**

For graphics work, claims need evidence.

Create a protocol:

```txt
assets/test-scenes/
screenshots/baseline/
screenshots/smaa/
screenshots/vrs/
screenshots/tonemap/
metrics/frame-times/
```

Each comparison should record:

- GPU
- OS
- resolution
- render settings
- scene
- camera position
- average GPU timings
- screenshot pair

## C5 — Public Portfolio Framing

Priority: **Later**

Eventually the project should explain itself clearly:

- what problem it explores
- why native rendering quality matters
- what techniques are implemented
- what is measured
- what is not included
- how it differs from engine-level feature lists

This is important for portfolio/research credibility.

---

# Track D — Future Odin Host Preparation

## D1 — C ABI Design Document

Priority: **Soon as documentation, not implementation**

Recommended file:

```txt
docs/analysis/04-engine-api-direction.md
```

Purpose:

Define the future API without coupling implementation yet.

Minimum concepts:

```c
typedef struct RendererHandle_T* RendererHandle;
typedef uint32_t RendererMeshHandle;
typedef uint32_t RendererMaterialHandle;
typedef uint32_t RendererTextureHandle;

typedef struct RendererCreateInfo RendererCreateInfo;
typedef struct RendererFramePacket RendererFramePacket;
typedef struct RendererStats RendererStats;
```

Must define:

- lifetime rules
- memory ownership rules
- frame packet validity
- threading assumptions
- error reporting
- resource loading model
- window ownership model

## D2 — Odin Mock Renderer Prototype

Priority: **Later, separate repository/project**

Goal:

Learn Odin and test engine/editor structure without linking to the real Vulkan renderer yet.

Prototype can include:

- window/app loop experiment
- scene graph
- transform hierarchy
- asset metadata structs
- editor panels
- fake renderer C API
- frame packet generation

Mock renderer behavior:

```txt
renderer_submit_frame(packet):
  print draw count
  validate transforms
  collect fake stats
```

This gives practical Odin learning while protecting the C++ renderer from premature FFI churn.

## D3 — Real Renderer C ABI

Priority: **After internal frame packet and resource handles exist**

Implementation conditions:

Do not begin until:

- renderer has internal frame packet concept
- renderer has resource handles
- viewer/backend boundary is cleaner
- at least one pass extraction has happened
- resize path is explicit

First real API should be minimal:

```c
RendererHandle renderer_create(const RendererCreateInfo* info);
void renderer_destroy(RendererHandle renderer);
void renderer_resize(RendererHandle renderer, uint32_t width, uint32_t height);
void renderer_submit_frame(RendererHandle renderer, const RendererFramePacket* packet);
void renderer_get_stats(RendererHandle renderer, RendererStats* out_stats);
```

Do not expose:

- Vulkan handles
- C++ object pointers
- STL containers
- internal descriptors/pipelines
- renderer-owned memory pointers

## D4 — Threaded Host/Renderer Model

Priority: **After single-threaded C ABI works**

First do:

```txt
Odin update -> submit frame -> C++ render -> present
```

Only then try:

```txt
Odin engine/editor thread builds packet N+1
C++ render thread consumes packet N
```

Required before threading:

- immutable frame packets
- double-buffered packet storage
- explicit resource lifetime rules
- deferred destruction
- clear synchronization strategy

---

# Recommended Development Order

## Phase 1 — Documentation and Stabilization

1. finish current analysis docs
2. write `README.md` / getting started
3. add ADR for C++ core + future Odin host
4. document current frame flow
5. document current pass flow

## Phase 2 — Viewer/Renderer Boundary Cleanup

1. move viewer panels out of `main.cpp`
2. introduce grouped settings structs
3. create internal `RenderFramePacket` for camera/light/settings
4. keep existing viewer behavior working

## Phase 3 — Post-Processing Architecture

1. extract `TonemapPass`
2. add SMAA pass objects
3. add debug views and GPU timings
4. validate screenshot comparison workflow

## Phase 4 — Resource and Scene Boundary

1. introduce mesh/material/texture handles internally
2. generate draw commands from loaded glTF
3. stop treating one loaded model as the permanent renderer scene
4. prepare for external frame packet draw submission

## Phase 5 — VRS Research

1. query feature support
2. design fallback behavior
3. implement classification/debug pass
4. measure quality/performance tradeoff

## Phase 6 — Future C ABI Preparation

1. write API header draft
2. implement C++ wrapper around stable internal API
3. test with a C mock host
4. later test with Odin mock host

## Phase 7 — Odin Engine/Editor Prototype

Separate project.

1. scene graph
2. transform system
3. editor shell
4. mock renderer API
5. frame packet generation
6. later real renderer integration

---

# What Not To Do Yet

Avoid these until there is real pressure:

- rewriting the renderer as a library immediately
- deleting the C++ viewer
- moving editor UI to Odin immediately
- adding a full engine ECS to this repo
- exposing Vulkan handles through C ABI
- building a huge render graph framework
- implementing async asset streaming before handles exist
- doing FFI and multithreading at the same time
- optimizing before baseline screenshots/timings exist

---

# Near-Term Checklist

Recommended next concrete tasks:

```txt
[ ] Create docs/analysis/04-engine-api-direction.md
[ ] Create docs/decisions/ADR-0003-cpp-core-odin-host-boundary.md
[ ] Create README.md or docs/getting-started.md
[ ] Document current frame flow
[ ] Document current pass flow
[ ] Plan ViewerPanels extraction from main.cpp
[ ] Plan RenderSettings structs
[ ] Plan internal RenderFramePacket
```

## Next Recommended Document

`docs/analysis/04-engine-api-direction.md`

Purpose:

Define the future C ABI and Odin host relationship as a design contract without implementing it yet.
