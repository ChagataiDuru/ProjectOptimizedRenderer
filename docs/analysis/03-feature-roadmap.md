# 03 — Feature and Architecture Roadmap

Date: 2026-06-13

## Purpose

This document connects renderer feature development with architecture work and explicitly marks what is already done, what is in progress, and what is still ahead.

The project direction remains hybrid:

- short term: C++ Vulkan renderer research prototype and portfolio project
- long term: engine-ready renderer backend
- future optional host: separate Odin engine/editor connected through a C ABI

The roadmap exists to avoid two failure modes:

1. adding renderer features so quickly that the architecture becomes unmaintainable
2. over-engineering engine architecture before the renderer research value is proven

## Strategic Thesis

ProjectOptimizedRenderer should not become a generic clone of a commercial engine renderer.

It should keep a clear technical identity:

> A modern Vulkan renderer focused on native spatial quality, stable shadows, perceptual image quality, explicit GPU control, and engine-ready architecture without hiding the rendering model behind proprietary black-box abstractions.

Core pillars:

1. native rendering quality over upscaling-first design
2. explicit Vulkan 1.4 backend
3. stable cascaded shadow maps and VSM experimentation
4. HDR scene pipeline with AgX and PBR Neutral comparison
5. HDR MSAA / native anti-aliasing path
6. perceptual VRS driven by image/scene metrics
7. cross-platform Windows + macOS support
8. future C ABI boundary for external engine/editor hosts

## Current Status Snapshot

The repo has moved materially since the first version of this roadmap.

Already landed in code:

- `ViewerPanels` extraction from `main.cpp`
- grouped renderer settings in `RenderSettings.h`
- internal `RenderFramePacket`
- internal `DrawCommand`
- handle types in `RenderHandles.h`
- `TonemapPass`, `ShadowPass`, `SkyPass`, and `ClusteredLightCullingPass`
- centralized shader interface constants/layout structs in `ShaderInterface.h`
- device capability modeling for optional features in `VulkanContext`
- clustered forward-plus point light support

Because of that, several items below have been re-marked from future-only to partial or done.

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
Status: **In Progress**

Current features already present:

- glTF/Sponza loading
- PBR direct lighting
- normal mapping
- directional lighting
- clustered forward-plus point lights
- HDR render target
- Reinhard / AgX / PBR Neutral tonemapping
- CSM shadows
- PCF
- VSM moments
- VSM blur
- procedural / HDR sky
- GPU timing
- render stats
- screenshots

Goal:

Make the current renderer baseline reproducible before adding more visual systems.

Remaining deliverables:

- known-good test scene list
- screenshot capture convention
- render setting presets worth comparing repeatedly
- per-platform build sanity notes
- documented visual limitations
- documented baseline timing captures

Success criteria:

- the same test scene can be run on macOS and Windows
- screenshots can be compared between tone mapping modes and shadow modes
- pass timings are visible and understandable
- visual regressions are easy to detect manually

## A2 — Capability-Driven HDR MSAA Path

Priority: **High**  
Status: **Foundation Implemented**

Purpose:

Add a native multisampling anti-aliasing solution that fits the project thesis: clear, stable, non-upscaled rendering without temporal accumulation.

Expected scene flow:

```txt
Scene geometry + sky
  -> multisampled HDR color + multisampled scene depth
  -> single hardware resolve into HDR
  -> tone mapping / present
```

Design questions:

1. Which sample counts are reliable across Windows/native Vulkan and macOS/MoltenVK?
2. How much quality does controlled sample shading add relative to cost?
3. How well does alpha-to-coverage improve masked material stability?
4. Which benchmark scenes best expose geometry and alpha-mask aliasing?

Architecture requirements:

- persistent AA settings outside `RenderFramePacket`
- exact-format sample-count capability discovery
- multisampled scene color/depth with a single resolved HDR target
- pipeline sample-count compatibility for scene-space contributors
- runtime sample-count and sample-shading controls
- GPU timing and screenshots for repeatable comparisons

Recommended first implementation:

```txt
Renderer-owned HDR MSAA foundation
  - supported scene sample-count discovery
  - persistent AA settings
  - MSAA color/depth attachments
  - single-sample HDR resolve for TonemapPass
  - optional sample shading controls
```

SMAA remains a later native spatial comparison or fallback path rather than the immediate AA milestone.

## A3 — Perceptual VRS

Priority: **High, after HDR MSAA foundation**
Status: **Not Started, Foundation Present**

Purpose:

Implement variable rate shading as a perceptual optimization path rather than a generic performance checkbox.

What is already in place:

- device capability querying in `RendererDeviceFeatures`
- optional feature gating groundwork
- fragment shading rate support detection
- fallback-friendly design direction

Expected concept:

- analyze luminance / contrast / edge structure
- produce shading-rate image or equivalent control data
- reduce shading where perceptually safe
- preserve edges, silhouettes, highlights, text, and high-frequency regions

Design questions:

1. Which devices/platforms expose usable VRS support?
2. How should MoltenVK/macOS behave when VRS is unavailable?
3. How should debug overlays expose VRS decisions?
4. What visual metric proves that VRS is not just blurring quality?

Architecture requirements:

- capability-driven enablement
- optional pass/path activation
- fallback path when unsupported
- settings struct for VRS mode/debug
- GPU timing around VRS classification
- screenshot comparisons with/without VRS

Important:

VRS must remain optional. Unsupported platforms should run normally without it.

## A4 — Shadow Quality and Stability Pass

Priority: **Medium-High**  
Status: **In Progress**

Current features:

- cascaded shadows
- PCF
- VSM
- cascade debug overlay
- shadow culling stats
- extracted `ShadowPass`
- `ShadowSettings` struct
- several quality-facing knobs already present

Goals:

- reduce shimmering
- validate texel snapping behavior
- document cascade split behavior
- tune VSM bleed reduction
- compare PCF vs VSM visually and by timing
- define useful shadow presets

Remaining deliverables:

```txt
Shadow Quality Presets:
  - Hard
  - PCF Low
  - PCF High
  - VSM Fast
  - VSM Quality
```

Recommended focus:

- make the current shadow feature set reproducible and measurable before broadening it further
- document what each shadow setting actually changes visually and architecturally

## A5 — HDR / Tone Mapping Research

Priority: **Medium**  
Status: **Partially Completed**

Current features:

- HDR offscreen target
- Reinhard
- AgX
- PBR Neutral
- exposure control
- split-screen comparison
- extracted `TonemapPass`

Goals:

- document differences between operators
- define why AgX / PBR Neutral are included
- add reproducible screenshots
- expose tone mapping presets
- consider histogram / luminance tools later

Architecture note:

This item is no longer only conceptual on the architecture side; `TonemapPass` is now a real extracted pass and should be treated as the template for future post-processing passes.

## A6 — Image-Based Lighting / Environment Lighting

Priority: **Later**  
Status: **Not Started**

Current sky support is present, but full IBL is not the immediate priority.

Future goals:

- diffuse irradiance
- specular prefiltered environment
- BRDF LUT
- material response validation
- HDR panorama as lighting source

Do this after the pass/resource boundaries are cleaner and the current lighting baseline is better documented.

---

# Track B — Renderer Architecture Stabilization

## B1 — Viewer Shell Separation

Priority: **Soon**  
Status: **Partially Completed**

Problem:

`main.cpp` still owns too much viewer/editor behavior.

What has landed:

```txt
include/viewer/ViewerPanels.h
src/viewer/ViewerPanels.cpp
```

That was the correct first extraction and it is already in the repo.

What remains:

- keep lifecycle/input/bootstrap code distinct from renderer-facing state flow
- avoid letting viewer-only concerns continue to shape the backend API
- possibly introduce a clearer `ViewerApp`-style shell later without over-abstracting now

Non-goal:

Do not delete the C++ viewer.

## B2 — Renderer Settings Structs

Priority: **Soon**  
Status: **Done (Initial Version)**

What landed:

- `DirectionalLightData`
- `ShadowSettings`
- `TonemapSettings`
- `SkySettings`
- `DebugViewSettings`
- `CameraData`

Benefits already realized:

- easier frame-packet design
- clearer UI-to-renderer handoff vocabulary
- better future C ABI mapping

What remains:

- keep extending settings only where they represent coherent state bundles
- avoid falling back into many unrelated setters for new systems

## B3 — Internal Render Frame Packet

Priority: **Before C ABI work**  
Status: **Partially Completed**

What landed:

```cpp
struct RenderFramePacket {
    CameraData camera;
    DirectionalLightData light;
    std::vector<shader_interface::GpuPointLight> pointLights;
    ShadowSettings shadow;
    TonemapSettings tonemap;
    SkySettings sky;
    DebugViewSettings debug;
};
```

Also landed:

- `Renderer::submitFrame(const RenderFramePacket&)`

What remains:

- make this the dominant path for viewer-to-renderer state submission
- reduce duplication between packet submission and legacy setter-based mutation
- decide whether draw submission belongs inside this packet or a sibling submission object

Current recommendation:

Do not expose a public C ABI version yet. First make the C++ internal packet stable.

## B4 — Resource Handle Model

Priority: **Before multi-object / Odin integration**  
Status: **Partially Completed**

What landed:

```cpp
struct MeshHandle { uint32_t index; };
struct MaterialHandle { uint32_t index; };
struct TextureHandle { uint32_t index; };
```

Also landed:

- `DrawCommand`
- internal mesh/material arrays
- renderer-side draw-command rebuilding from imported model data

What remains:

- generation/versioning if handles become externally visible
- clearer renderer resource lifetime APIs
- eventual separation between imported asset data and renderer-owned resource registries

This item is now past the concept stage.

## B5 — Pass Ownership Extraction

Priority: **During AA/post-processing work**
Status: **Partially Completed**

Extracted pass objects already in the repo:

1. `TonemapPass`
2. `ShadowPass`
3. `SkyPass`
4. `ClusteredLightCullingPass`

This is meaningful progress.

What remains:

- keep shrinking pass-specific ownership inside `Renderer`
- define clearer resize and descriptor-update responsibilities for each extracted pass
- keep SMAA as a later pass-oriented comparison or fallback path

Non-goal:

Do not build a large render graph before pass extraction provides real pressure.

## B6 — Minimal Render Graph / Pass Dependency Layer

Priority: **Later**  
Status: **Not Started**

Trigger point:

Implement this only when MSAA, SMAA, VRS, or post-processing chains make manual pass ordering painful.

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
Status: **Partially Completed**

What landed:

- centralized binding/set constants in `ShaderInterface.h`
- centralized shader-facing C++ structs in `ShaderInterface.h`
- static asserts for layout/size validation
- code comments clarifying descriptor ownership boundaries

What remains:

- reflection-based validation in debug builds, or
- codegen/shared schema if the interface surface grows enough to justify it

This item is in a good state for the current project size and does not yet require a heavy framework.

## B8 — Capability-Driven Optional Feature Policy

Priority: **Now / Soon**  
Status: **Started Well**

This is newly important because it underpins future VRS and mesh-shader experiments.

What landed:

- `RendererDeviceFeatures`
- `RendererMeshShaderProperties`
- Vulkan physical-device capability query path
- optional device-extension enabling for supported features

What remains:

- document which optional features are merely detected vs actually used
- define the renderer policy for unsupported platforms
- expose capability information cleanly to debug UI/logging if useful

This item was not explicit enough in the original roadmap and now deserves to remain visible.

---

# Track C — Tooling, Documentation, Portfolio Evidence

## C1 — Build and Run Documentation

Priority: **Immediate**  
Status: **In Progress**

What exists:

- a substantial `README.md`
- architecture analysis docs
- ADR documenting the C++ core / Odin host direction

What still needs work:

- known-good platform matrix
- clearer validated run instructions by platform
- baseline asset / screenshot expectations
- known limitations section that reflects current reality, not only intent

## C2 — Technique Research Notes

Priority: **Ongoing**  
Status: **Mostly Not Started**

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
- screenshots / metrics to capture

## C3 — Architecture Decision Records

Priority: **Ongoing**  
Status: **Started**

Already present:

- ADR for C++ renderer core with future Odin host boundary

Recommended next ADR candidates:

- internal frame-packet boundary before public C ABI
- capability-driven optional feature policy
- pass extraction strategy vs render-graph timing

## C4 — Portfolio / Evidence Capture

Priority: **Medium**  
Status: **Needs More Structure**

Capture evidence for:

- shadow comparisons
- tone-mapping comparisons
- clustered lighting scenes
- GPU timing captures
- platform screenshots
- feature capability logs on different GPUs/platforms

This matters both for project quality and for demonstrating technical identity.

---

# Track D — Future Odin Host Preparation

## D1 — Keep Odin Out of the Core Repo for Now

Priority: **Immediate**  
Status: **Still Correct**

The current architecture work is reducing future integration risk, but the repo is not ready for a stable external ABI yet.

Do now:

- continue shaping C++-internal packet/handle/pass concepts
- let the viewer remain the first client of those concepts

Do not do yet:

- bind Odin directly to shifting renderer internals
- export STL/C++-shaped contracts

## D2 — Public C ABI Draft

Priority: **Later**  
Status: **Not Started**

Preconditions:

- frame-packet shape stops changing frequently
- handle model is clearer
- draw submission boundary is explicit
- pass/resource ownership is more stable

Only then should the project draft public APIs like:

```c
typedef struct RendererFramePacket RendererFramePacket;
typedef uint32_t RendererMeshHandle;
typedef uint32_t RendererMaterialHandle;
```

## D3 — External Host Scene Submission Model

Priority: **Later**  
Status: **Conceptual Foundation Present**

What already exists internally:

- handles
- draw commands
- frame packet

What remains:

- decide how the eventual host submits scene draw lists
- keep asset import concerns out of the future host-facing ABI
- avoid high-frequency FFI mutator chatter

---

# Recommended Next Engineering Slice

The roadmap-informed next slice should be:

1. consolidate `RenderFramePacket` so it becomes the normal internal submission path
2. decide how `DrawCommand` participates in that submission boundary
3. continue pass extraction using the same model for future AA/post-processing work
4. document optional-feature policy and capability logging
5. stabilize screenshot/timing comparison workflow for the current baseline

In practical terms:

- **best next renderer feature:** capability-driven HDR MSAA
- **best next architecture task:** reduce duplication between packet submission and legacy mutator APIs
- **best next documentation task:** baseline comparison notes for shadows, tone mapping, and clustered lighting

# Summary

The roadmap has progressed more than the original document reflected.

The project is no longer only planning:

- viewer separation
- settings structs
- frame packet concepts
- pass extraction
- handle-based draw submission
- capability-driven optional features

It now has first-cut implementations of all of those.

The highest-value next step is not to invent another big abstraction. It is to consolidate the new architecture direction and then stress-test it with capability-driven HDR MSAA, which will naturally reveal what still needs to be extracted or stabilized.
