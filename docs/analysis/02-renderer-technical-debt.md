# 02 — Renderer Technical Debt Map

Date: 2026-06-17

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
- build split into `por_renderer_core` and `por_viewer`
- `RendererInstance` lifecycle facade
- `RendererResourceManager` batch GPU resource ownership
- C ABI draft header with C11/C++20 mock-backed compile tests

Because of that, the main technical debt is no longer “these concepts do not exist.” The real debt is that they exist in first-cut form and need hardening before they become stable ABI-facing commitments.

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

`Renderer` is no longer responsible for glTF import/reload policy or direct ImGui ownership, but it still coordinates a large portion of backend execution:

- material descriptor creation
- camera/light buffers
- clustered-light buffers and metadata
- scene pass orchestration
- pass coordination
- resize handling
- screenshot/profiling integration
- render statistics
- resize-sensitive pass/resource rebinding

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

1. keep `RendererInstance` as the only lifecycle entry used by the viewer and C wrapper
2. move more resize-sensitive ownership into pass/resource owners when it simplifies code
3. evolve `RendererResourceManager` into explicit resource create/destroy operations

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

However, `src/main.cpp` still owns viewer/application policy:

- app startup
- logger setup
- camera setup
- frame timing
- SDL event handling
- mouse capture
- CPU glTF import/reload and fallback behavior
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

- keep shrinking viewer access to renderer internals; it should continue talking through `RendererInstance`

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

The current public path is primarily `RendererInstance::renderFrame(RenderFramePacket)` over `Renderer::submitFrame()` plus sticky `submitScene()`. Remaining mutators are mostly persistent/debug configuration such as AA settings, screenshot requests, and HDR panorama loading.

## Why It Matters

A future host should submit a stable, plain-data frame snapshot. It should not depend on many renderer mutators or partial state updates.

## Risk

- duplicated submission pathways
- ambiguous source of truth for per-frame state
- unstable future host/renderer contract
- state bugs from partial updates

## Recommended Direction

Keep the packet path dominant. New per-frame features should enter `RenderFramePacket`; persistent or scene-sticky data should enter explicit resource APIs or `RenderScenePacket`, not ad hoc frame mutators.

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
- sticky `RenderScenePacket`
- `RendererResourceManager` owning uploaded GPU resources
- viewer-owned glTF import/reload policy

But stable per-resource create/destroy APIs are still not complete; the current model upload is still a batch bridge from CPU `Model` data.

## Why It Matters

An engine renderer should render many objects, each potentially referencing shared mesh/material resources with different transforms. The engine/editor host should own object hierarchy and transforms.

## Risk

- batch model upload is not yet an engine-facing resource API
- no clear multi-object submission model yet
- deletion/reuse/generation semantics are not settled

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

1. **TD-004 — Resource handles are not yet stable resource APIs**
2. **TD-001 — `Renderer` is still a heavy coordinator**
3. **TD-008 — Resize handling is still cross-cutting**
4. **TD-005 — Pass extraction is started but ownership is still mixed**
5. **TD-003 — Keep packet submission dominant**
6. **TD-009 — Capability policy exists in code but not yet as a clear product rule**
7. **TD-002 — `main.cpp` still owns viewer policy**
8. **TD-007 — Shader interface safety improved, but validation is still manual-plus**
9. **TD-006 — No lightweight render graph / pass dependency model yet**

## Why This Order

The highest-value work now is not inventing more architecture layers. It is consolidating the ones that already landed:

- `RendererInstance` lifecycle boundary
- packet-based submission
- pass extraction
- handle-based scene vocabulary
- batch renderer resource ownership
- capability-driven optional feature policy

That consolidation will make future AA comparisons, VRS experiments, and ABI work much easier to implement without re-fragmenting the architecture.
