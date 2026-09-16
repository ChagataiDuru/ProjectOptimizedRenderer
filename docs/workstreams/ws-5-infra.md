# WS-5 — Infrastructure and Robustness

Status: audited; findings recorded. No code changes applied beyond WS-0 hygiene.

## Scope

Descriptor/binding correctness, push-constant and UBO layout agreement, resize paths,
frame synchronization, validation-layer cleanliness, and dead code.

## Owned files

* `include/core/ShaderInterface.h`
* `src/core/Renderer.cpp` — descriptor pools/sets, resize and attachment recreation,
  viewport/scissor, frame submission plumbing.
* `src/core/{Swapchain,FrameSync,CommandBuffer,RendererInstance}.cpp`
* `CMakeLists.txt`

## Method

1. **Validation is the primary detector.** Run any capture with the Debug build (validation
   layers active) and treat every new VUID/message as a finding:

```bash
./build/debug/ProjectOptimizedRenderer --capture screenshots/ws5 --preset all --frames 3 2>&1 \
  | grep -iE "vuid|validation|error"
```

   Known noise at pipeline creation: `Vertex attribute at location N not consumed by vertex
   shader` (ART-INF-001). Known MoltenVK warning at capture: reading a presented drawable
   (ART-INF-002). Anything else is new.
2. Check that `ShaderInterface.h` structs match the std140/std430 layout the shaders
   declare. The static asserts cover standard layout, not offset agreement — compare field
   order against the GLSL blocks by hand when a struct changes.
3. For resize, verify attachments, descriptor rebinds, and viewport/scissor extent are all
   driven from the same extent, and that no pass keeps a stale image view:
   `createResizeDependentResources` / `refreshResizeDependentBindings` in `Renderer.cpp`.
4. Never bind a `VK_NULL_HANDLE` pipeline: the bind helpers compare against the previously
   bound value, so a failed pipeline creation silently produces invalid binds followed by
   draws. Add explicit checks when touching that path.

## Findings

* ART-INF-001 open — unused vertex attributes in the shadow/normals pipelines.
* ART-INF-002 open — screenshot reads the swapchain image after present. Captures succeed on
  MoltenVK, but the readback should happen before present (or through a dedicated readback
  target) to be spec-clean and portable.
* ART-INF-003 fixed — dead `triangle.vert`/`triangle.frag` removed from the build.
* ART-INF-004 fixed — progress-step naming removed from code and docs.

## Acceptance

* A full `--preset all` capture run shows only the two known warnings above.
* `ctest --test-dir build/debug` passes; interactive resize still works.
* Eventual alignment follow-up: cascade math must not compute reverse-Z NDC with a
  near/far pair that differs from the projection matrix it unprojects with (see
  ART-SHD-001) — the same class of bug is worth grepping for wherever NDC is constructed.
