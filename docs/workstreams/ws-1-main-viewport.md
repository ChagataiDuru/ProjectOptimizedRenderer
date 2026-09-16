# WS-1 — Main Viewport (scene pass)

Status: correctness fixes applied; remaining items open in `inventory.md`.

## Scope

The scene/PBR pass as the user sees it: geometry transform, normal mapping, material
sampling, alpha masking, the MSAA path, and the debug views. Not in scope: shadows
(WS-3), clusters (WS-2), sky/tonemap (WS-4).

## Owned files

* `shaders/pbr.vert`, `shaders/pbr.frag` — **material / normal / alpha / ambient sections
  only**. The shadow-sampling section of `pbr.frag` belongs to WS-3.
* `src/core/Renderer.cpp` — PBR pipeline creation, alpha-masked pipeline selection,
  MSAA attachment/resolve, debug-view pipeline switching.
* `include/core/RenderSettings.h` — `DebugViewSettings`, `AntiAliasingSettings` only.

## Presets to capture first

`scene-overview`, `scene-wall-close`, `scene-floor-close`, `normals-debug`, `msaa-4x`,
`msaa-4x-a2c`, `msaa-sample-shading`, `sky-off`.

## Method

1. Capture the preset set before touching anything; diff presets that *should* differ.
   Byte-identical output between `msaa-4x` and `msaa-4x-a2c` means the control is not
   effective (see ART-VPW-005) — do not assume the control works.
2. Check normal-map correctness with `normals-debug` plus a close view of a wall with a
   mirrored UV island.
3. For MSAA, verify the resolve path, the masked A2C pipeline selection, and that
   `sampleShadingEnabled`/`minSampleShading` reach the pipeline
   (`VkPipelineMultisampleStateCreateInfo`).

## Findings

* ART-VPW-001 fixed — dead clip-space `* 2` in `pbr.vert`.
* ART-VPW-002 fixed — degenerate derivative TBN now falls back to the geometric normal.
* ART-VPW-003/004/005 open — see `inventory.md`.

## Acceptance

* Each fix has before/after PNGs from the identical preset, plus the numeric diff
  (max/mean/changed %) recorded in `inventory.md`.
* `normals-debug` output must contain no NaN/black holes after the TBN guard.
* No new validation errors; existing tests still pass.
