# Artifact Inventory

One row per finding. Status values: `fixed` (implemented + verified), `open` (found,
not implemented — see the recommendation), `rejected` (investigated, not a defect).

Evidence paths are relative to the repo root. Capture runs are gitignored, so evidence
is referenced by run directory name rather than by committed file.

| ID | Subsystem | Symptom | Repro preset | Root cause | Status |
|---|---|---|---|---|---|
| ART-SHD-001 | Shadows / CSM | Shadows had no effect anywhere; filter mode, distance, and caster-culling changes produced 0–5 differing pixels | `shadow-hard` vs `shadow-pcf` vs `shadow-distance-short` vs `shadow-cull-off` | `ShadowPass::updateMatrices()` computed cascade-slice reverse-Z NDC with the shadow-distance ladder near/far (`n`, `f`) but unprojected the corners with the camera projection matrix (near 0.01 / far 1000). Cascade 0 landed at view depth ≈[0.01, 1.39] instead of [0.05, 6.51], and the far cascade spanned out toward 1000 units, so `projectToCascade()` rejected the entire scene | **fixed** |
| ART-SHD-002 | Shadows / CSM | Receiver bias is a fixed shader constant (0.005) applied on top of `vkCmdSetDepthBias()`; no normal-offset bias, so acne/peter-panning cannot be tuned from one place | `shadow-interior-grazing-sun-bias-zero` vs `shadow-pcf` | Two independent bias mechanisms; the shader constant is unreachable from `ShadowSettings` | open — recommendation |
| ART-SHD-003 | Shadows / CSM | PCF taps pushed outside the cascade were clamped to the border, duplicating edge texels and smearing/streaking cascade boundaries | `shadow-pcf` | `sampleShadowPCF()` used `clamp(sampleUV, 0, 1)` before `texelFetch` | **fixed** |
| ART-SHD-004 | Shadows / CSM | Shader could divide by zero in `sampleShadowVSM()`; lighting UBO parameters could disagree with the shadow pass actually running | `shadow-vsm` | `Renderer::buildCurrentLightUBO()` used the raw packet `ShadowSettings` while `ShadowPass` sanitized its own copy (`vsmBleedReduction` clamped to 0.95, `pcfSpreadRadius` to ≥0.25). `vsmBleedReduction = 1.0` gives `(pMax - 1) / 0` | **fixed** |
| ART-SHD-005 | Shadows / CSM | Hard shadow cutoff line at the shadow distance; cascade blend zone can double-darken | `shadow-distance-short`, `cascades-debug` | No fade-to-lit beyond `maxDistance`; the last 20% of each cascade blends two cascades with `smoothstep` | open — recommendation |
| ART-SHD-006 | Shadows / CSM | First rendered frame builds cascade matrices from a stale (identity) camera view | any preset, frame 0 | `updateMatrices()` runs during command recording before a camera has been submitted for that frame | open — low severity |
| ART-SHD-007 | Shadows / CSM | `shadow_blur.comp` indexes a fixed 7-tap weight table with `blur.radius`; any radius other than 3 reads out of bounds | `shadow-vsm` | Blur radius is hardcoded to 3 in `ShadowPass::record()`, so it holds today, but the shader has no guard | open — latent |
| ART-SHD-008 | Shadows / CSM | Possible shadow edge crawl when moving | — | `updateMatrices()` derives `lightViewSnap` for texel snapping and then re-derives `lightView` from `snappedCenter`; needs a moving-camera check | open — unverified |
| ART-VPW-001 | Main viewport | Dead clip-space scaling in the scene vertex shader | any preset | `pbr.vert` ended with `... * worldPos * 2`, which scales `w` as well and is a no-op for NDC | **fixed** |
| ART-VPW-002 | Main viewport | Degenerate normals / invalid shading at UV seams and zero-area triangles | `scene-wall-close`, `normals-debug` | Derivative TBN normalized a tangent that can be exactly zero | **fixed** |
| ART-VPW-003 | Main viewport | Mirrored-UV surfaces can shade with an inverted normal | `scene-wall-close` | Bitangent handedness is not applied and the vertex format has no tangent `w` component | open — recommendation |
| ART-VPW-004 | Main viewport | Normal map silently disabled per-pixel where the sampled texel is exactly white | `scene-wall-close` | Normal-map presence is inferred from `normalSample != vec3(1.0)` instead of a material flag | open — recommendation |
| ART-VPW-005 | Main viewport | A2C and sample shading produced byte-identical output to plain MSAA 4x | `msaa-4x` vs `msaa-4x-a2c` vs `msaa-sample-shading` | Not resolved: the capture framings may simply contain no masked geometry or high-frequency edges. Needs a masked-material framing before calling it a defect | open — unverified |
| ART-LGT-001 | Lighting / clusters | Lights whose centre is at or behind the camera project to mirrored screen positions and were assigned to the wrong clusters | `clusters-off`, `scene-wall-close` | `cluster_cull.comp` divided by `clipCenter.w` without checking its sign (the projection maps visible points to positive `w`) | **fixed** |
| ART-LGT-002 | Lighting / clusters | Cluster screen-radius used the vertical projection term for both axes | `scene-wall-close` | `radiusPixels` was a scalar derived from `projection[1][1]` | **fixed** |
| ART-LGT-003 | Lighting / clusters | Fireflies near a point-light centre; hard falloff ring at the radius | `clusters-off` vs default | `evaluatePointLight()` divides by `distSq` clamped at 0.0001 and cuts off hard at the light radius | open — recommendation |
| ART-LGT-004 | Lighting / clusters | Interiors stay lit in shadow (flat ambient glow) | `sky-off` | Ambient is a constant `baseColor * ambientIntensity` with no occlusion or IBL | open — recommendation |
| ART-SKY-001 | Sky / output | Dense speckled/noisy band along the horizon in the default exterior view — the most visible demo artifact | `scene-overview` (band disappears in `sky-off`) | Procedural atmosphere marches 16 fixed steps with an unjittered start and a hard Earth intersection; `tMax - tMin` collapses to near zero for downward rays, so the integration aliases. Also `EARTH_R + 0.0002` is 0.2 m while the comment claims 200 m | open — recommendation |
| ART-SKY-002 | Sky / output | Sun disc brightness is unrelated to the directional light intensity | `scene-overview`, `sky-procedural` | `sunDisc` is scaled by an arbitrary `sunPower * 0.2` | open — recommendation |
| ART-SKY-003 | Sky / output | Negative HDR values from the BRDF are only guarded in the PBR Neutral operator | `tonemap-agx`, `tonemap-pbr-neutral` | `max(color, 0)` appears only in `tonemapPBRNeutral()`; AgX guards its own log input, Reinhard does not need one | rejected — no artifact observed (70.6% of pixels differ between Reinhard and AgX as expected) |
| ART-INF-001 | Infrastructure | Validation warnings at pipeline creation: vertex attributes 1/2/3 declared but not consumed | any preset, pipeline creation | Shadow/normals pipelines bind the full vertex layout while their shaders read only position (shadow) or position/uv (shadow alpha) | open — low severity |
| ART-INF-002 | Infrastructure | MoltenVK warning: reading the swapchain image after present ("should not be called after already presenting this drawable") | any capture | `Renderer::endFrame()` presents first and then captures the just-presented swapchain image | open — medium severity |
| ART-INF-003 | Infrastructure | Unused `triangle.vert` / `triangle.frag` still compiled by the shader target | build | Dead since the PBR pipeline replaced the triangle sample | **fixed** |
| ART-INF-004 | Infrastructure | Docs/naming drift: `Phase N` markers across headers/shaders and a stale archived phase log | — | Progress-step naming leaked into permanent comments and documentation | **fixed** |

## Verification notes

**ART-SHD-001** — temporary instrumentation logged the intended cascade slice view
depths against the reconstructed corner depths:

```
cascade 0: intended [0.050, 6.510] -> reconstructed [0.050, 6.510]
cascade 1: intended [6.510, 14.183] -> reconstructed [6.510, 14.183]
```

Before the fix the same reconstruction landed outside the scene. A temporary shader probe
(`outColor = vec4(vec3(shadowFactor), 1.0)`) confirmed the shadow term was a constant 1.0
before the fix and that `projectToCascade()` reported in-range UVs afterwards. Aggregate
effect on the capture matrix (`screenshots/baseline` → `screenshots/after`):

| Preset | max abs diff | mean abs diff | changed pixels |
|---|---|---|---|
| `scene-wall-close` | 116 | 0.2318 | 15.886% |
| `scene-overview` / `shadow-pcf` / `shadow-hard` | 134 | 0.0036 | 0.111% |
| `cascades-debug` | 76 | 0.0011 | 0.073% |
| `normals-debug` | 0 | 0.0000 | 0.000% |

Mean image brightness is unchanged (`scene-wall-close` 125.24 → 125.27), so the
difference is localised to shadow-dependent shading rather than a global exposure shift.

**Byte-identical preset pairs are findings, not noise.** Before the fixes,
`scene-overview`, `shadow-hard`, `shadow-pcf`, `shadow-cull-off`, `shadow-distance-short`,
`sky-procedural`, and `tonemap-reinhard` produced identical PNGs, and
`msaa-4x` / `msaa-4x-a2c` / `msaa-sample-shading` were identical to each other. That is
what exposed ART-SHD-001 and is the reason ART-VPW-005 stays open rather than rejected.
