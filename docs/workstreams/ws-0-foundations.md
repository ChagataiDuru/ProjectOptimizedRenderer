# WS-0 — Foundations

Status: **done** (implemented in the spring-cleanup branch).

## Goal

Give the artifact workstreams a deterministic way to reproduce rendering output, then
remove process naming from the repo and collapse the duplicated documentation.

## Deliverables

1. **Scripted capture mode** (`src/main.cpp`, `include/viewer/CapturePresets.h`,
   `src/viewer/CapturePresets.cpp`, `Camera::setView`).
   `--capture <dir> --preset <id|all> --scene <gltf> --frames <n> --width/--height
   --list-presets`. Capture mode skips the ImGui overlay, renders a fixed number of
   frames with a fixed `deltaTime`, optionally re-submits the scene without point lights,
   writes one PNG per preset, and exits non-zero on failure. Without `--capture` the
   interactive viewer is unchanged.
2. **Naming purge.** No `Phase N` markers remain in `include/`, `src/`, `shaders/`, or
   `docs/`. `docs/analysis/04-engine-api-direction.md` uses descriptive section titles
   instead of `Stage 0..6`.
3. **Docs unification.** `docs/analysis/03-feature-roadmap.md` owns the status table;
   `00-current-state.md`, `02-renderer-technical-debt.md`, and `README.md` link to it
   instead of restating it. All analysis docs re-baselined to 2026-09-16. Trailing
   "next document" pointers removed. `docs/archive/documentation.md` deleted.
4. **Hygiene.** `imgui.ini` untracked and ignored, `screenshots/` ignored, the stray
   `documentation.md` ignore rule removed, dead `triangle.vert`/`triangle.frag` deleted
   with their `CMakeLists.txt` entries.

## Acceptance

* `cmake --build --preset debug` succeeds and `ctest --test-dir build/debug` passes.
* `--list-presets` prints 25 ids; `--capture <dir> --preset all` writes 25 non-empty PNGs.
* Running without arguments still opens the interactive viewer with panels, F2 screenshot,
  and resize working.

## Notes for the following workstreams

* Add presets only in `CapturePresets.cpp`; never in `main.cpp`.
* Presets for a workstream must be appended so existing capture filenames keep meaning.
