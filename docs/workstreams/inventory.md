# Artifact Inventory

One row per finding. Status values: `fixed` (implemented + verified), `open` (found,
not implemented — see the recommendation), `rejected` (investigated, not a defect).

Evidence paths are relative to the repo root. Capture runs are gitignored, so evidence
is referenced by run directory name rather than by committed file.

| ID | Subsystem | Symptom | Repro preset | Root cause | Status |
|---|---|---|---|---|---|
| ART-SHD-001 | Shadows / CSM | Shadows had no effect anywhere; filter mode, distance, and caster-culling changes produced 0–5 differing pixels | `shadow-hard` vs `shadow-pcf` vs `shadow-distance-short` vs `shadow-cull-off` | `ShadowPass::updateMatrices()` computed cascade-slice reverse-Z NDC with the shadow-distance ladder near/far (`n`, `f`) but unprojected the corners with the camera projection matrix (near 0.01 / far 1000). Cascade 0 landed at view depth ≈[0.01, 1.39] instead of [0.05, 6.51], and the far cascade spanned out toward 1000 units, so `projectToCascade()` rejected the entire scene | **fixed** |
| ART-SHD-002 | Shadows / CSM | Receiver bias is a fixed shader constant (0.005) applied on top of `vkCmdSetDepthBias()`; no normal-offset bias, so acne/peter-panning cannot be tuned from one place | `shadow-interior-grazing-sun-bias-zero` vs `shadow-pcf` | Two independent bias mechanisms; the shader constant is unreachable from `ShadowSettings`. With the interior framing, zero rasterizer bias darkens only 0.1% of pixels and produces no acne on the sunlit floor, so this is a tunability issue, not a visible defect | open — recommendation (checked in `screenshots/cull-fix2`) |
| ART-SHD-003 | Shadows / CSM | PCF taps pushed outside the cascade were clamped to the border, duplicating edge texels and smearing/streaking cascade boundaries | `shadow-pcf` | `sampleShadowPCF()` used `clamp(sampleUV, 0, 1)` before `texelFetch` | **fixed** |
| ART-SHD-004 | Shadows / CSM | Shader could divide by zero in `sampleShadowVSM()`; lighting UBO parameters could disagree with the shadow pass actually running | `shadow-vsm` | `Renderer::buildCurrentLightUBO()` used the raw packet `ShadowSettings` while `ShadowPass` sanitized its own copy (`vsmBleedReduction` clamped to 0.95, `pcfSpreadRadius` to ≥0.25). `vsmBleedReduction = 1.0` gives `(pMax - 1) / 0` | **fixed** |
| ART-SHD-005 | Shadows / CSM | Hard shadow cutoff line at the shadow distance; cascade blend zone can double-darken | `shadow-distance-short`, `cascades-debug` (not observable in the interior framings, which stay inside cascade 0–1) | No fade-to-lit beyond `maxDistance`; the last 20% of each cascade blends two cascades with `smoothstep` | open — recommendation |
| ART-SHD-006 | Shadows / CSM | First rendered frame builds cascade matrices from a stale (identity) camera view | any preset, frame 0 | `updateMatrices()` runs during command recording before a camera has been submitted for that frame | open — low severity |
| ART-SHD-007 | Shadows / CSM | `shadow_blur.comp` indexes a fixed 7-tap weight table with `blur.radius`; any radius other than 3 reads out of bounds | `shadow-vsm` | Blur radius is hardcoded to 3 in `ShadowPass::record()`, so it holds today, but the shader has no guard | open — latent |
| ART-SHD-008 | Shadows / CSM | Possible shadow edge crawl when moving | — | `updateMatrices()` derives `lightViewSnap` for texel snapping and then re-derives `lightView` from `snappedCenter`; needs a moving-camera check (capture presets are static, so this cannot be verified with the current tooling) | open — unverified |
| ART-VPW-001 | Main viewport | Dead clip-space scaling in the scene vertex shader | any preset | `pbr.vert` ended with `... * worldPos * 2`, which scales `w` as well and is a no-op for NDC | **fixed** |
| ART-VPW-002 | Main viewport | Degenerate normals / invalid shading at UV seams and zero-area triangles | `scene-wall-close`, `normals-debug` | Derivative TBN normalized a tangent that can be exactly zero | **fixed** |
| ART-VPW-003 | Main viewport | Mirrored-UV surfaces can shade with an inverted normal | `scene-wall-close` | Bitangent handedness is not applied and the vertex format has no tangent `w` component | open — recommendation |
| ART-VPW-004 | Main viewport | Normal map silently disabled per-pixel where the sampled texel is exactly white | `scene-wall-close` | Normal-map presence is inferred from `normalSample != vec3(1.0)` instead of a material flag | open — recommendation |
| ART-VPW-005 | Main viewport | A2C and sample shading produced byte-identical output to plain MSAA 4x | `msaa-4x-interior` vs `msaa-4x-a2c-interior` vs `msaa-sample-shading-interior` | Two separate causes. **A2C:** not a defect; the exterior framings contain no masked materials, and inside the atrium A2C changes 1.8% of pixels. **Sample shading:** a real defect. MoltenVK (driver 0.2.2209) reports `sampleRateShading` and accepts `sampleShadingEnable`, but still shades once per pixel: output stayed byte-identical even with `minSampleShading = 1.0`, while `sample`-qualified inputs changed 56.9% of pixels. Fixed with a `pbr_sample.frag.spv` variant (`POR_PER_SAMPLE_SHADING`) selected by `Renderer::pbrFragmentShaderName()`. Metal has no fractional mode, so every sample is shaded regardless of `minSampleShading`. `sky.frag` has no such variant; it only shades the background | **fixed** (sample shading); A2C rejected — framing |
| ART-VPW-006 | Main viewport | Sun-facing surfaces received no direct light; surfaces facing away from the sun were lit; interior point lights leaked onto exterior walls | `shadow-interior-grazing-sun-hard` (N·L probe) | PBR pipelines used `VK_FRONT_FACE_CLOCKWISE`, but glTF front faces are CCW and the Y-flipped camera projection keeps them CCW in framebuffer space. `gl_FrontFacing` was therefore false for every front face, and `pbr.frag` negated N everywhere. A temporary probe (`outColor = (shadowFactor, N·L, cascade/3)`) showed N·L = 0 on the atrium floor while the lit shadow term was correct; after switching both PBR pipeline copies to `COUNTER_CLOCKWISE`, the floor gets N·L > 0 and the −X lion wall gets 0, as expected for the sun direction | **fixed** |
| ART-VPW-007 | Main viewport | Distant textures shimmered and aliased; minification read full-resolution texels | `scene-overview`, `shadow-interior` | `Image::create()` always used `mipLevels = 1`, so the trilinear/anisotropic sampler had a single level. Textures now get a full mip chain built on the GPU with linear blits (`Image::generateMipmaps()`, format support checked once). High-frequency (Laplacian) energy in distant regions dropped 66% (`scene-overview` roof) and 38–42% (interior far floor); close-up `scene-floor-close` is unchanged; texture memory 272 → 363 MB. Watch item: mip-averaged alpha can thin alpha-tested foliage at large distances (no thinning visible at interior distances) | **fixed** (`screenshots/mips`) |
| ART-VPW-008 | Main viewport | Every PBR draw rasterized back faces (overdraw), ignoring `Material::doubleSided` | `shadow-interior`, `scene-overview` | PBR pipelines used a static `VK_CULL_MODE_NONE`. The cull mode is now dynamic (`VK_DYNAMIC_STATE_CULL_MODE`, Vulkan 1.3 core) and set per draw: `BACK` for single-sided materials, `NONE` for the 3/25 double-sided Sponza materials and for wireframe/normals debug views. 29/32 presets are byte-identical; the three interior MSAA presets differ in ~37 silhouette pixels (0.001%), where back faces used to cover edge samples. Scene-pass GPU time over the matrix: 176.9 → 142.3 ms total | **fixed** (`screenshots/cull-back`) |
| ART-LGT-001 | Lighting / clusters | Lights whose centre is at or behind the camera project to mirrored screen positions and were assigned to the wrong clusters | `clusters-off`, `scene-wall-close` | `cluster_cull.comp` divided by `clipCenter.w` without checking its sign (the projection maps visible points to positive `w`) | **fixed** |
| ART-LGT-002 | Lighting / clusters | Cluster screen-radius used the vertical projection term for both axes | `scene-wall-close` | `radiusPixels` was a scalar derived from `projection[1][1]` | **fixed** |
| ART-LGT-003 | Lighting / clusters | Fireflies near a point-light centre; hard falloff ring at the radius | `clusters-off` vs default | `evaluatePointLight()` divides by `distSq` clamped at 0.0001 and cuts off hard at the light radius | open — recommendation |
| ART-LGT-004 | Lighting / clusters | Interiors stay lit in shadow (flat ambient glow) | `ibl-off`, `ibl-off-interior`, `ibl-panorama-interior` | Ambient was a constant `baseColor * ambientIntensity`. `IblPass` now bakes the current sky on the GPU into an equirectangular environment (256×128 + mips), a 32×16 cosine-convolved irradiance map and a 6-mip GGX prefilter, plus a 128×128 split-sum BRDF LUT at init; `pbr.frag` evaluates the split-sum ambient (scene bindings 9–11). Re-bakes only when the sky source changes (mode, panorama, or a >0.5° sun move). Deviation from the plan: diffuse uses an irradiance map instead of SH9 in the light UBO, which avoids a GPU→CPU readback. Panorama textures are now RGBA16F with mips (seam-safe `textureGrad` in `sky.frag`). Captures: interior differs from flat ambient on ~100% of pixels (mean 12.45) and the synthetic-panorama preset differs again (mean 7.9); mean brightness barely moves (57.9 → 58.9) | **fixed** (`screenshots/ibl`) |
| ART-SKY-001 | Sky / output | Dense speckled/noisy band along the horizon in the default exterior view — the most visible demo artifact | `scene-overview` (band disappears in `sky-off`) | Float32 cancellation in the ray–sphere test, not ray-march aliasing: `c = dot(ro, ro) - r*r` subtracts ~4e7-sized values whose true difference near the ground (~2.5 km²) is below float32 resolution, so near-horizon rays flipped between Earth hit and miss per pixel; `-b - sqrt(disc)` also cancelled for downward rays. `raySphereFromAltitude()` now forms `c` from the altitude and uses the stable quadratic roots. Horizontal pixel-to-pixel variation in the horizon strip dropped from 13.9 to 0.002. The atmosphere moved to `shaders/include/atmosphere.glsl` (shader include support added to the build). The observer altitude stays 0.2 m (the comment said 200 m; 200 m turns the unlit ground into blue haze and would tint the IBL lower hemisphere) | **fixed** (`screenshots/sky-fix`) |
| ART-SKY-004 | Sky / output | Horizon tilted ~122 px across the frame with zero camera roll (the old speckle band was tilted the same way) | `scene-overview` | `sky.vert` normalized the per-vertex ray before interpolation. Far-plane positions are affine in screen space, but normalized directions are not, and the oversized fullscreen triangle is asymmetric, so the interpolated rays were skewed. The vertex shader now passes the unnormalized direction (`sky.frag` already normalizes). Horizon rows at the left/right frame edges: 533/411 → 419/418 (expected ≈414 for a −10° pitch). Also affects HDR panorama mode | **fixed** (`screenshots/sky-fix`) |
| ART-SKY-002 | Sky / output | Sun disc brightness is unrelated to the directional light intensity | `scene-overview`, `sky-procedural` | `sunDisc` is scaled by an arbitrary `sunPower * 0.2` | open — recommendation |
| ART-SKY-003 | Sky / output | Negative HDR values from the BRDF are only guarded in the PBR Neutral operator | `tonemap-agx`, `tonemap-pbr-neutral` | `max(color, 0)` appears only in `tonemapPBRNeutral()`; AgX guards its own log input, Reinhard does not need one | rejected — no artifact observed (70.6% of pixels differ between Reinhard and AgX as expected) |
| ART-INF-001 | Infrastructure | Validation warnings at pipeline creation: vertex attributes 1/2/3 declared but not consumed | any preset, pipeline creation | Shadow/normals pipelines bind the full vertex layout while their shaders read only position (shadow) or position/uv (shadow alpha) | open — low severity |
| ART-INF-002 | Infrastructure | MoltenVK warning: reading the swapchain image after present ("should not be called after already presenting this drawable") | any capture | `Renderer::endFrame()` presents first and then captures the just-presented swapchain image | open — medium severity |
| ART-INF-003 | Infrastructure | Unused `triangle.vert` / `triangle.frag` still compiled by the shader target | build | Dead since the PBR pipeline replaced the triangle sample | **fixed** |
| ART-PRF-001 | Performance | Main pass drew every mesh (no frustum culling), in submission order, with the sky shaded under all geometry | `perf-culling-off*`, `perf-sort-off*` | World AABBs existed only for shadow culling. The main pass now frustum-culls against them (`culling::` helpers shared with `ShadowPass`), orders draws opaque → alpha-tested → A2C and front-to-back, and records the sky after the geometry so early-Z skips covered pixels. Interior framings cull 46/103 draws, `scene-wall-close` 78/103. All previous presets and the culling/sorting on-off pairs are byte-identical. GPU scene time change is within timing noise (142.3 → 135.6 ms over the matrix): the culled meshes were off-screen anyway, so the saving is mostly CPU/driver draw-call work, which is not measured yet | **fixed** (`screenshots/frustum`) |
| ART-INF-005 | Infrastructure | Every "interior" capture preset was actually outside Sponza, so shadow filters, sun direction, and A2C looked like no-ops | `shadow-interior*` | Offsets were scaled by the bounding-sphere radius (6.23), but Sponza's half-depth is only 3.08, so `z = 0.85 × radius` placed the camera outside the wall. `CaptureCamera::boundsRelative` now scales by `SceneInfo::normalizedHalfExtent`; interior presets use it | **fixed** |
| ART-SHD-010 | Shadows / CSM | Shadows disappeared or reappeared with sub-degree camera rotations; interiors rendered fully sunlit for many sun directions | `shadow-cull-side-sun` vs `shadow-cull-side-sun-off` (must be byte-identical) | Three faults in shadow-caster culling (`computeShadowCullPlanes()` / `updateMatrices()`): (1) it was given the direction toward the sun, while the plane classification assumes the direction light travels. (2) Silhouette extrusion planes were oriented by the dropped face's normal, which is wrong for some edge/light configurations; in random tests, 183 of 400 cases were non-conservative, and the planes rejected all 103 meshes whenever the sun had a zero X or Z component. (3) Cascades c>0 were culled against their own slice only, but `pbr.frag` also samples them in the last 20% of the previous cascade. Fixed by passing `-lightDir`, orienting each extrusion plane so the frustum centroid is on its inner side (0/400 non-conservative), and starting the cull slice at 0.8 × the previous split. Across a 48-direction × 2-view sweep, culling on and off are now byte-identical (previously the floor was up to 58 levels brighter with culling on), while culling still removes meshes (e.g. 30/92/103 in the interior cascades) | **fixed** |
| ART-LGT-006 | Lighting / clusters | Image-based ambient has no occlusion: interior surfaces are lit as if they saw the whole sky, and dark rough materials (foliage, curtains) pick up a visible sky-coloured sheen | `shadow-interior` with IBL on vs off | Not a maths error — probes confirmed prefiltered radiance ≈1.1 and BRDF LUT (scale 0.43, bias 0.05) are in range; for dark materials the specular term is simply comparable to the diffuse one. Distant-environment IBL cannot know about the walls. Screen-space occlusion (ART-LGT-005) is the intended mitigation, applied to the ambient and, approximately, to the specular term | open — mitigated by GTAO |
| ART-SHD-009 | Shadows / CSM | Shadow edges on the atrium floor show large stair-steps in hard mode | `shadow-interior-grazing-sun-hard` | Texel lookup (`textureSize`/`texelFetch`) is consistent; likely the default shadow quality (1024 px) against distant roof-edge casters. Not investigated further | open — observation |
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

**Interior validation (`screenshots/cull-fix2`, 2026-09-17)** — `tools/compare_captures.py`
gate after ART-INF-005, ART-VPW-005, ART-VPW-006, and ART-SHD-010:

| Pair | max | mean | changed |
|---|---|---|---|
| `shadow-interior` vs `-grazing-sun` | 205 | 21.221 | 37.418% |
| `-grazing-sun-hard` vs `-pcf` | 103 | 1.233 | 8.673% |
| `-grazing-sun-pcf` vs `-vsm` | 86 | 0.604 | 9.082% |
| `-grazing-sun-pcf` vs `-bias-zero` | 36 | 0.017 | 0.575% |
| `msaa-4x-interior` vs `-a2c-interior` | 73 | 0.172 | 1.837% |
| `msaa-4x-interior` vs `msaa-sample-shading-interior` | 104 | 0.889 | 56.918% |
| `shadow-cull-side-sun` == `-off` (must match) | 0 | 0 | 0% |

**Ground truth for the interior shadows.** A throwaway CPU ray caster (numpy
Möller–Trumbore over all 262k Sponza triangles, normalized like `computeSceneInfo()`)
checked sun visibility at 288 floor points of the `-grazing-sun` framing. The render agrees
with it at 274/288 points for hard shadows and 270/288 for PCF (lit median 129 vs shadow
median 75). The same ray caster showed that the first interior sun direction,
`(0.6, 0.75, 0.25)`, leaves the whole atrium floor in shadow. Its "sunlit" capture was
therefore an ART-SHD-010 artifact, so the presets now use `(0.55, 0.82, 0.10)`, which lights
about 17% of the floor.

The `screenshots/interior` numbers recorded earlier were measured with ART-SHD-010 still
present, and `screenshots/final` predates ART-VPW-006. Neither is a valid baseline; use
`screenshots/cull-fix2`.
