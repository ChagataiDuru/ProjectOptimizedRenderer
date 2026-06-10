# ADR-0003 — C++ Renderer Core and Future Odin Host Boundary

Date: 2026-06-10

Status: **Accepted**

## Context

ProjectOptimizedRenderer is a C++ Vulkan renderer focused on modern native rendering techniques, cross-platform Vulkan development, and future engine-readiness.

The project currently includes a C++ standalone viewer/debug shell with SDL3, ImGui, camera controls, runtime model loading, render settings, GPU timing, screenshots, tone mapping controls, shadow controls, and visual debugging features.

A future idea is to build a separate engine/editor host in Odin. The Odin side would own higher-level engine concerns such as object management, scene graph, transform hierarchy, asset metadata, editor panels, animation/audio/game systems, and frame packet production.

The architectural question is whether ProjectOptimizedRenderer should become an Odin-driven engine project now, or remain a C++ renderer core with possible external host integration later.

## Decision

ProjectOptimizedRenderer remains the **core C++ Vulkan renderer repository**.

A future Odin engine/editor may exist as a **separate host project** connected to the renderer through a **narrow C ABI**.

The current C++ viewer/debug shell remains valid and should not be removed. It continues to serve as the renderer research harness for graphics features, shader experiments, screenshots, GPU timing, RenderDoc captures, and visual debugging.

## Rationale

The renderer is still actively evolving. It already owns many low-level Vulkan systems and renderer-specific features:

- Vulkan device/swapchain setup
- GPU resource lifetime
- VMA allocations
- shader and pipeline creation
- PBR rendering
- shadow maps
- VSM blur
- HDR target
- tone mapping
- sky rendering
- GPU timers
- screenshots
- debug overlays

This makes C++ the correct owner of renderer internals.

Odin is still interesting for the future engine/editor host because it is well-suited for:

- explicit allocator control
- data-oriented scene structures
- transform/component arrays
- editor/application code
- C interop
- fast iteration
- no garbage-collected runtime

However, integrating Odin directly into the renderer now would create unnecessary instability. The renderer does not yet have a stable external API surface. The current `Renderer` class is still too broad and implementation-specific to expose directly.

## Consequences

### Positive Consequences

- Renderer development can continue quickly in C++.
- Vulkan internals stay behind C++ ownership and RAII.
- The current C++ viewer remains useful for research and profiling.
- Odin exploration remains possible without destabilizing renderer internals.
- The future engine/editor boundary can be designed deliberately.
- The renderer can eventually support multiple hosts:
  - C++ research viewer
  - external Odin engine/editor
  - possibly other C-compatible hosts later

### Negative Consequences

- A real Odin editor will not exist immediately.
- Some future API design must be done twice: first as internal C++ structures, later as C ABI structs.
- The project must avoid letting the C++ viewer become an accidental engine shell.
- The renderer needs architectural cleanup before clean C ABI integration is possible.

### Neutral Consequences

- The C++ viewer is not the final engine editor.
- Odin is not rejected; it is deferred to the correct layer.
- C ABI design begins as documentation before implementation.

## Rules Implied by This Decision

The future C ABI must expose only:

- opaque renderer handles
- integer resource handles
- POD structs
- arrays with explicit counts
- explicit create/destroy functions
- explicit error codes

The future C ABI must not expose:

- C++ classes
- `std::string`
- `std::vector`
- C++ exceptions
- RAII objects
- Vulkan handles as normal API objects
- VMA allocations
- internal descriptor sets
- internal pipeline objects
- pointers to renderer-owned containers

## Future API Shape

The intended external API shape is packet-based:

```c
por_result por_renderer_create(
    const por_renderer_create_info* info,
    por_renderer_handle* out_renderer);

void por_renderer_destroy(
    por_renderer_handle renderer);

por_result por_renderer_resize(
    por_renderer_handle renderer,
    uint32_t width,
    uint32_t height);

por_result por_renderer_submit_frame(
    por_renderer_handle renderer,
    const por_frame_packet* packet);

por_result por_renderer_get_stats(
    por_renderer_handle renderer,
    por_renderer_stats* out_stats);
```

The Odin host should eventually build a frame packet containing camera data, lights, draw commands, render settings, and debug settings. The C++ renderer consumes the packet and owns all GPU-side execution.

## Implementation Preconditions

Do not implement real Odin integration until the renderer has:

1. grouped renderer settings structs
2. an internal `RenderFramePacket`
3. an explicit renderer resize pathway
4. internal resource handles
5. clearer viewer/backend separation
6. documented frame flow
7. documented resource lifetime rules
8. at least one extracted pass or clearer pass ownership

## Recommended Development Path

1. Continue C++ renderer feature work.
2. Keep the C++ viewer as the research/debug application.
3. Refactor toward cleaner internal renderer boundaries.
4. Draft the C ABI as documentation first.
5. Later create an Odin prototype in a separate repository/project using a mock renderer API.
6. Connect Odin to the real renderer only after the internal renderer API stabilizes.

## Alternatives Considered

### Alternative A — Rewrite/drive the project from Odin now

Rejected.

Reason:

The renderer is not ready for a stable FFI boundary. Starting Odin integration now would couple a new engine/editor layer to unstable renderer internals.

### Alternative B — Keep everything permanently C++ only

Rejected for now.

Reason:

Odin remains valuable as a possible future host language for engine/editor systems. The architecture should not close that door.

### Alternative C — Expose C++ classes directly to Odin

Rejected.

Reason:

This would leak implementation details and make ownership unsafe across language boundaries.

### Alternative D — C++ renderer core with future C ABI and separate Odin host

Accepted.

Reason:

This preserves renderer velocity while keeping the long-term engine/editor direction open.

## Related Documents

- `docs/analysis/00-current-state.md`
- `docs/analysis/01-architecture-boundaries.md`
- `docs/analysis/02-renderer-technical-debt.md`
- `docs/analysis/03-feature-roadmap.md`
- `docs/analysis/04-engine-api-direction.md`
