# WS-3 — Shadows and CSM

Status: one critical correctness fix applied; remaining items open in `inventory.md`.

## Scope

Cascaded shadow map construction (splits, cascade boxes, light matrices, texel snapping),
depth bias, the PCF/hard/VSM sampling paths, VSM moments and blur, caster culling, and the
cascade debug overlay.

## Owned files

* `src/renderpasses/ShadowPass.cpp`, `include/renderpasses/ShadowPass.h`
* `shaders/shadow.vert`, `shadow_alpha.vert`, `shadow_alpha.frag`, `shadow_vsm.frag`,
  `shadow_vsm_alpha.frag`, `shadow_blur.comp`
* `shaders/pbr.frag` — **shadow sampling section only** (`projectToCascade`,
  `sampleShadowHard`, `sampleShadowPCF`, `sampleShadowVSM`, `sampleCascade`,
  `computeShadow`, the `Lo *= shadowFactor` application, cascade debug overlay).
* `include/core/RenderSettings.h` — `ShadowSettings`, `ShadowDebugInfo`.

## Presets to capture first

`shadow-hard`, `shadow-pcf`, `shadow-vsm`, `shadow-bias-zero`, `shadow-distance-short`,
`shadow-cull-off`, `cascades-debug`, `shadow-interior`, `shadow-interior-grazing-sun`,
`shadow-interior-grazing-sun-pcf`, `shadow-interior-grazing-sun-bias-zero`.

## Method

1. **Never trust a shadow setting that changes nothing.** Diff every shadow preset pair.
   `shadow-hard` vs `shadow-pcf` was byte-identical before this workstream, which is the
   signature of the cascade-placement defect (ART-SHD-001). Re-run this check after any
   change: a legitimate reason for identical output is that no visible receiver is
   occluded in that framing, which must be *demonstrated*, not assumed.
2. Exterior Sponza framings contain almost no shadowed receiver. Use the
   `shadow-interior-*` presets and add new interior framings in `CapturePresets.cpp` when
   a fix needs a visible shadow to verify.
3. To inspect the shadow term, temporarily set
   `outColor = vec4(vec3(shadowFactor), 1.0);` after `computeShadow`, capture, then revert.
   Remember the HDR target is tone mapped afterwards, so decode before reading values.
4. To inspect cascade geometry, temporarily log the intended slice depths against the
   reconstructed corner view depths in `updateMatrices()` (see `inventory.md`).
5. Any change to cascade matrices must be checked for the reverse-Z convention: the
   cascade corners are unprojected with the **camera** projection, so the NDC depth must
   be derived from that projection's near/far, not from the shadow-distance ladder.

## Findings

* ART-SHD-001 fixed and numerically verified — see `inventory.md`.
* ART-SHD-003/004 fixed — PCF border clamping; sanitized shadow settings now feed the
  lighting UBO.
* ART-SHD-002/005/006/007/008 open — see `inventory.md`.

## Acceptance

* Cascade slice reconstruction matches the intended view depth range for every cascade.
* `shadow-hard` vs `shadow-pcf` differ where a shadow edge is visible, and caster-culling
  and shadow-distance changes produce measurable differences in a framing with shadows.
* `shadow-vsm` differs materially from `shadow-pcf` in such a framing.
* No new validation errors; existing tests still pass.
