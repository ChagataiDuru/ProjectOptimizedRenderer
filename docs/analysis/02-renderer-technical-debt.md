# 02 — Renderer Technical Debt Map

Date: 2026-06-13

## Purpose

This document lists current and expected technical debt in ProjectOptimizedRenderer.

The goal is not to shame the codebase. The renderer is still in a productive research/prototype phase. Some debt is acceptable because it helps rapid feature iteration. The goal is to identify which debt still blocks the long-term direction:

- C++ Vulkan renderer remains the core project.
- The current C++ viewer remains useful as a research/debug shell.
- A future Odin engine/editor host may exist as a separate project connected through a C ABI.
- The renderer should eventually become a stable backend, not only a single demo application.

## Important Re-baseline

Several debts originally recorded here have moved from “missing entirely” to “partially addressed.”

Already landed in code:

- viewer panel extraction into `ViewerPanels`
- grouped renderer settings in `RenderSettings.h`
- internal `RenderFramePacket`
- internal `DrawCommand`
- handle types in `RenderHandles.h`
- pass extraction via `TonemapPass`, `ShadowPass`, `SkyPass`, and `ClusteredLightCullingPass`
- centralized shader interface constants/layout structs in `ShaderInterface.h`
- runtime device capability modeling in `VulkanContext`

Because of that, the main technical debt is no longer “these concepts do not exist.” The real debt is that they exist in first-cut form but have not yet become the dominant architecture everywhere.

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

# TD-001 — `Renderer` Is Still a Heavy Coordinator

Severity: **High**  
Timing: **Soon**

## Current State

`Renderer` still owns or coordinates a large portion of the system after Vulkan/swapchain setup:

- model loading/reload flow
- mesh upload
- texture upload
- fallback texture creation
- material descriptor creation
- camera/light buffers
- clustered-light buffers and metadata
- scene pass orchestration
- pass coordination
- resize handling
- screenshot/profiling integration
- render statistics
- remaining renderer state mutators

Pass extraction has started, which is good, but `Renderer` is still the gravity well for too many systems.

## Why It Matters

Future features such as SMAA, perceptual VRS, IBL, transparency, async loading, material editing, and eventual host integration will all naturally try to attach to `Renderer`. Without continued decomposition, every system becomes coupled to every other system.

## Risk

- difficult resource lifetime reasoning
- large resize/reload bugs
- hard-to-test pass interactions
- difficult future C ABI exposure
- backend remains shaped like a demo app

## Recommended Direction

Continue splitting responsibility gradually:

```txt
Renderer
  RenderResourceManager-ish ownership split
  Scene submission boundary
  Pass objects
  Debug/profiling surface
```

Next extraction pressure points:

1. unify packet-based submission vs legacy setters
2. reduce scene/resource rebuilding responsibilities inside `Renderer`
3. make resize responsibilities more explicit per pass/resource

## Not To Do Yet

Do not build a large commercial-style render graph immediately. The project still benefits more from explicit pass ownership than from a large framework.

---

# TD-002 — `main.cpp` Still Owns Too Much Viewer Bootstrap

Severity: **Medium-High**  
Timing: **Soon**

## Current State

This debt is now **partially reduced**, because panel registration was extracted into:

```txt
include/viewer/ViewerPanels.h
src/viewer/ViewerPanels.cpp
```

However, `src/main.cpp` still owns:

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
- deferred file actions
- main render loop

## Why It Matters

The future direction explicitly says the Odin engine/editor host is separate. Therefore, `main.cpp` should be treated as a disposable C++ research viewer shell, not the architectural center of the engine direction.

## Risk

- viewer assumptions continue to leak into renderer-facing APIs
- future host behavior must be reverse-engineered from viewer code
- lifecycle/input and renderer contracts remain too tightly interwoven

## Recommended Direction

Keep the current viewer, but continue treating it as a client of the renderer rather than the renderer's center of gravity.

Low-cost next step:

- reduce direct viewer-driven mutation paths in favor of building a frame/state submission object

## Not To Do Yet

Do not remove the C++ viewer. It is valuable for graphics research and GPU debugging.

---

# TD-003 — Frame Submission Model Exists but Is Not Yet Dominant

Severity: **High**  
Timing: **Now / Soon**

## Current State

This debt is no longer “no explicit frame packet abstraction.”

The repo now has:

- `RenderFramePacket`
- `Renderer::submitFrame(const RenderFramePacket&)`
- grouped render settings/data structs

However, the renderer still also exposes and relies on multiple direct mutators such as camera/light/shadow/tonemap/sky setters.

## Why It Matters

A future host should submit a stable, plain-data frame snapshot. It should not depend on many renderer mutators or partial state updates.

## Risk

- duplicated submission pathways
- ambiguous source of truth for per-frame state
- unstable future host/renderer contract
- state bugs from partial updates

## Recommended Direction

Promote the internal packet model from “available” to “primary.”

Target state:

```cpp
struct RenderFramePacket {
    CameraData camera;
    DirectionalLightData light;
    std::vector<shader_interface::GpuPointLight> pointLights;
    ShadowSettings shadow;
    TonemapSettings tonemap;
    SkySettings sky;
    DebugViewSettings debug;
    // later: draw submission or scene submission reference
};
```

Then:

- build this packet in the viewer
- submit it once per frame
- gradually reduce legacy per-subsystem mutators

## Not To Do Yet

Do not freeze a public C ABI around this yet. First stabilize the internal C++ version.

---

# TD-004 — Scene Submission Is Still Coupled to Imported Model State

Severity: **High**  
Timing: **Soon**

## Current State

This debt is also **partially reduced**.

The repo now has:

- `MeshHandle`, `MaterialHandle`, `TextureHandle`
- `DrawCommand`
- internal mesh/material vectors
- draw-command rebuilding from imported model data

But the renderer still fundamentally assumes a loaded glTF model as the active scene baseline.

## Why It Matters

An engine renderer should render many objects, each potentially referencing shared mesh/material resources with different transforms. The engine/editor host should own object hierarchy and transforms.

## Risk

- imported model remains too close to runtime scene representation
- renderer and asset loader remain coupled
- no clear multi-object submission model yet
- future host integration still lacks a clean scene boundary

## Recommended Direction

Keep the current bridge approach, but continue separating these concepts:

```txt
Asset import model
  - loaded glTF CPU data

Renderer resources
  - mesh handles
  - material handles
  - texture handles

Scene submission
  - draw commands / scene packet referencing handles
```

## Not To Do Yet

Do not design a full asset database inside the renderer. That belongs to the future engine/editor host or external tooling.

---

# TD-005 — Pass Extraction Is Started but Ownership Is Still Mixed

Severity: **High**  
Timing: **Soon**

## Current State

This debt is no longer “no pass ownership extraction.”

The repo already contains extracted passes:

- `TonemapPass`
- `ShadowPass`
- `SkyPass`
- `ClusteredLightCullingPass`

That is real progress.

The remaining debt is that pass-local ownership, resize flow, descriptor update flow, and orchestration responsibilities are still split between pass classes and `Renderer`.

## Why It Matters

SMAA and VRS will increase the number of passes and intermediate resources. If pass-local ownership stays mixed, every new feature will multiply resize/setup complexity.

## Risk

- duplicated descriptor logic
- resize bugs
- mixed lifetime ownership
- hard-to-reason pass interactions

## Recommended Direction

Keep using pass-local extraction as the scaling path.

Best next test:

- implement `SmaaPass` using the same ownership model
- let that reveal whether the existing pass interface is sufficient or needs refinement

## Not To Do Yet

Do not replace pass extraction with a heavy render-graph system prematurely.

---

# TD-006 — No Lightweight Render Graph / Pass Dependency Model Yet

Severity: **Medium Now, High Later**  
Timing: **Later, after SMAA/VRS pressure increases**

## Current State

Pass sequencing is still mostly manual and explicit. Resources such as HDR target, depth, shadow maps, VSM moments, and swapchain images are manually transitioned/used.

## Why It Matters

A renderer with anti-aliasing, shadows, HDR, clustered lighting, post-processing, optional VRS, and debug overlays will eventually benefit from clearer pass dependencies.

## Risk

- barrier/layout transition mistakes
- transient image lifetime confusion
- hard-to-reorder passes
- hard to insert optional debug passes
- hard to reason about performance globally

## Recommended Direction

Do not jump immediately to a full render graph. First define a minimal pass dependency vocabulary only if SMAA/VRS make manual ordering painful.

## Not To Do Yet

Do not copy Unreal-scale render-graph complexity. The project needs a small renderer-owned dependency model, not an engine-scale framework.

---

# TD-007 — Shader Interface Safety Improved, but Validation Is Still Manual-Plus

Severity: **Medium**  
Timing: **Soon / Medium**

## Current State

This debt is **partially reduced**.

The repo now has:

- centralized shader binding constants in `ShaderInterface.h`
- centralized shader-facing structs in `ShaderInterface.h`
- static asserts for layout/size validation
- clearer comments around descriptor ownership

The remaining issue is that validation is still mostly authored manually rather than being reflection-backed or generated.

## Why It Matters

Manual layout sync is manageable now, but becomes riskier as the renderer gains more passes and more shader-visible data structures.

## Risk

- silent shader/C++ mismatch
- broken push constant offsets
- descriptor binding mismatch during refactor

## Recommended Direction

Short-term:

- keep centralizing shader-facing contracts
- keep adding static assertions and clear ownership comments

Medium-term:

- add SPIR-V reflection validation in debug builds, or
- generate shared interface code if the surface area grows enough

## Not To Do Yet

Do not build a full shader compiler framework before SMAA/VRS create a real need.

---

# TD-008 — Resize Handling Is Still Cross-Cutting

Severity: **Medium**  
Timing: **Soon**

## Current State

Resize still affects multiple systems across the renderer:

- swapchain
- depth image
- HDR target
- extracted passes that depend on resize-sensitive resources
- viewport/scissor
- camera aspect ratio
- clustered light metadata/resources
- future SMAA intermediates
- future VRS images/resources

## Why It Matters

As post-processing grows, resize bugs become common and expensive.

## Risk

- stale descriptors after image recreation
- invalid image views
- mismatched viewport size
- wrong camera projection
- swapchain recreation instability

## Recommended Direction

Define and document a consistent resize pathway that clearly states which resources/passes own which reinitialization steps.

## Not To Do Yet

Do not over-abstract resize into a framework. First make ownership explicit.

---

# TD-009 — Capability Policy Exists in Code but Not Yet as a Clear Product Rule

Severity: **Medium**  
Timing: **Soon**

## Current State

The repo now has capability discovery for:

- fragment shading rate
- mesh shader/task shader
- descriptor indexing
- timeline semaphore
- core feature availability

What is still missing is a consistently documented policy for how optional capabilities should be surfaced, logged, debugged, and consumed by future features.

## Why It Matters

Perceptual VRS, mesh-shader experiments, and other optional research features should be capability-driven from day one. Without a clear rule, future features may drift back into ad hoc probing and inconsistent fallback behavior.

## Risk

- inconsistent platform behavior
- confusing logs/debug UX
- optional features accidentally becoming startup requirements

## Recommended Direction

Document a simple policy:

- detect capabilities once
- store them centrally
- log them clearly
- never make optional features mandatory unless intentionally promoted to baseline
- keep unsupported platforms functional

## Not To Do Yet

Do not add a giant feature-management layer. A small, explicit policy is enough.

---

# Recommended Debt Order

1. **TD-003 — Frame submission model exists but is not yet dominant**
2. **TD-001 — `Renderer` is still a heavy coordinator**
3. **TD-005 — Pass extraction is started but ownership is still mixed**
4. **TD-004 — Scene submission is still coupled to imported model state**
5. **TD-008 — Resize handling is still cross-cutting**
6. **TD-009 — Capability policy exists in code but not yet as a clear product rule**
7. **TD-002 — `main.cpp` still owns too much viewer bootstrap**
8. **TD-007 — Shader interface safety improved, but validation is still manual-plus**
9. **TD-006 — No lightweight render graph / pass dependency model yet**

## Why This Order

The highest-value work now is not inventing more architecture layers. It is consolidating the ones that already landed:

- packet-based submission
- pass extraction
- handle-based scene vocabulary
- capability-driven optional feature policy

That consolidation will make native AA work such as HDR MSAA now and SMAA comparisons later much easier to implement without re-fragmenting the architecture.
