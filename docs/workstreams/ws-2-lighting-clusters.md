# WS-2 — Lighting and Clustered Lights

Status: correctness fixes applied; remaining items open in `inventory.md`.

## Scope

Directional light shading inputs, the ambient term, point lights, the compute cluster
culling pass, cluster metadata, and the per-cluster light index buffer.

## Owned files

* `shaders/cluster_cull.comp`
* `src/renderpasses/ClusteredLightCullingPass.cpp`, `include/renderpasses/ClusteredLightCullingPass.h`
* `shaders/pbr.frag` — **point-light evaluation and cluster indexing sections only**
  (`evaluatePointLight`, `computeClusterIndex`, the cluster loop, `ambient`).
* `src/core/Renderer.cpp` — cluster buffer creation/sizing, cluster metadata upload.
* `include/core/RenderSettings.h` — `DirectionalLightData` only.

## Presets to capture first

`scene-overview`, `clusters-off`, `scene-wall-close`, `sky-off`.

## Method

1. Compare `clusters-off` against default to attribute shading to point lights vs the
   directional term.
2. The highest-risk case is a light near or behind the camera plane: walk the camera past
   the four demo lights and watch for popping at cluster/tile boundaries. The demo lights
   sit at ±0.45×sceneRadius, so a close framing plus a yaw sweep is enough.
3. Verify cluster metadata consistency: `screenSizeDepth.xy` must be the render-target
   extent the PBR pass shades, and `clusterCounts.xy` must match the culling dispatch.

## Findings

* ART-LGT-001 fixed — lights at/behind the camera plane fell back to a conservative
  full-screen footprint instead of a mirrored projection.
* ART-LGT-002 fixed — the screen-space radius now uses `projection[0][0]` for x.
* ART-LGT-003/004 open — see `inventory.md`.

## Acceptance

* No light popping while yawing past a light at close range; before/after captures for a
  framing that previously popped.
* `clusters-off` still renders the directional contribution unchanged.
* No new validation errors; cluster buffer sizes unchanged.
