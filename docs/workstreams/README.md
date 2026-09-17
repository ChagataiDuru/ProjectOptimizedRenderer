# Artifact Workstreams

This directory holds the per-subsystem workstreams used to find and fix rendering
artifacts in the demo scene, plus the live inventory of what was found.

## Layout

| File | Purpose |
|---|---|
| `inventory.md` | Live table of every finding: symptom, repro, root cause, status, evidence. |
| `ws-0-foundations.md` | Cleanup groundwork: capture mode, docs unification, naming purge, hygiene. |
| `ws-1-main-viewport.md` | Scene pass: PBR geometry, normal maps, alpha masking, MSAA path, debug views. |
| `ws-2-lighting-clusters.md` | Directional light, clustered forward-plus point lights, cluster culling. |
| `ws-3-shadows-csm.md` | Cascaded shadow maps, PCF/VSM filtering, bias, caster culling. |
| `ws-4-sky-tonemap.md` | Sky/atmosphere, exposure, tone mapping, final output. |
| `ws-5-infra.md` | Bindings, layouts, resize, sync, validation cleanliness, dead code. |

## How a workstream runs

Each brief is self-contained: scope, owned files, required capture presets, method,
and acceptance criteria. They are designed to be handed to one agent session each.

1. Run one workstream at a time, on one topic branch cut from `main` (the original
   `chore/spring-cleanup-2026-09` branch is merged). Do not create parallel branches or
   rebase mid-run; commit on top of the previous workstream.
2. One commit per finding group, with the `ART-*` id in the commit message.
3. Record every finding in `inventory.md` before fixing it, and keep the row updated
   afterwards (status + evidence path).
4. Stay inside the file ownership listed in the brief. Shared files (`pbr.frag`,
   `Renderer.cpp`) are partitioned by section on purpose; cross-workstream edits are a
   merge-conflict source, not a feature.

## Capture matrix

The viewer renders presets to PNGs and exits, so agents never need to drive the UI:

```bash
cmake --build --preset debug
./build/debug/ProjectOptimizedRenderer --list-presets
./build/debug/ProjectOptimizedRenderer --capture screenshots/<run> --preset all --frames 30
```

* One PNG per preset at `screenshots/<run>/<preset>.png`; `screenshots/` is gitignored.
* Preset definitions: `include/viewer/CapturePresets.h`, `src/viewer/CapturePresets.cpp`.
  Add new presets there (never in `main.cpp`). Framings that must stay inside the scene set
  `camera.boundsRelative`, which scales the offset by the scene's AABB half-extent instead
  of its bounding radius; the capture log prints both at startup.
* Presets are deterministic: fixed `deltaTime`, no input, camera set once per preset,
  and the ImGui overlay is not attached.
* Output directories must exist before capturing (`mkdir -p screenshots/<run>`).
* Compare runs numerically rather than by eye, using `tools/compare_captures.py`:

```bash
python3 tools/compare_captures.py screenshots/<run>              # must-differ gate (exit 1 on identical pairs)
python3 tools/compare_captures.py screenshots/<run> screenshots/<baseline>   # per-preset diff between runs
```

A byte-identical PNG between two presets that should differ is itself a finding: it means
a setting is not reaching the shader.
Conversely, an optimization (caster culling) that changes a must-match pair is a
finding too: it means the optimization is not conservative.

## Preset index

| Area | Presets |
|---|---|
| Framing | `scene-overview`, `scene-floor-close`, `scene-wall-close` |
| Shadow filtering | `shadow-hard`, `shadow-pcf`, `shadow-vsm`, `shadow-bias-zero` |
| Shadow parameters | `shadow-distance-short`, `shadow-cull-off`, `cascades-debug`, `shadow-cull-side-sun` / `shadow-cull-side-sun-off` (must be byte-identical) |
| Shadow visibility (inside the atrium) | `shadow-interior`, `shadow-interior-grazing-sun`, `shadow-interior-grazing-sun-hard`, `shadow-interior-grazing-sun-pcf`, `shadow-interior-grazing-sun-vsm`, `shadow-interior-grazing-sun-bias-zero` |
| Anti-aliasing | `msaa-4x`, `msaa-4x-a2c`, `msaa-sample-shading` (exterior); `msaa-4x-interior`, `msaa-4x-a2c-interior`, `msaa-sample-shading-interior` (masked foliage in view) |
| Ambient / occlusion | `ibl-off`, `ibl-off-interior`, `ibl-panorama-interior`, `ao-off-interior`, `ao-debug-interior`, `ao-on-exterior` |
| Depth prepass | `perf-prepass-on-noao` / `perf-prepass-off-noao` and the `-interior` pair (must be byte-identical) |
| Main-pass visibility | `perf-culling-off`, `perf-culling-off-interior` (must match the defaults), `perf-sort-off`, `perf-sort-off-interior` (report only) |
| Tone mapping | `tonemap-reinhard`, `tonemap-agx`, `tonemap-pbr-neutral`, `tonemap-split` |
| Sky / lighting isolation | `sky-procedural`, `sky-off`, `clusters-off` |
| Debug views | `normals-debug` |

## Scope rule

Workstreams fix deterministic correctness bugs. Perceptual or design changes
(normal-offset bias, cascade fade-to-lit, image-based ambient, sun/sky brightness
matching, default-setting retuning, sky ray-march jitter) are recorded in
`inventory.md` as recommendations and are not implemented here.
