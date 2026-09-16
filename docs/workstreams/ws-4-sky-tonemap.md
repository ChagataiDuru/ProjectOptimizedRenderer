# WS-4 — Sky, Tone Mapping, and Final Output

Status: audited; no correctness code changes applied. The main finding (ART-SKY-001) is a
recommendation because it is a quality/under-sampling issue rather than a hard defect.

## Scope

Procedural atmosphere, HDR panorama path, sun representation, exposure, the three tone map
operators, split-screen comparison, and the sRGB output transfer.

## Owned files

* `src/renderpasses/SkyPass.cpp`, `include/renderpasses/SkyPass.h`
* `src/renderpasses/TonemapPass.cpp`, `include/renderpasses/TonemapPass.h`
* `shaders/sky.vert`, `shaders/sky.frag`, `shaders/tonemap.vert`, `shaders/tonemap.frag`
* `include/core/RenderSettings.h` — `SkySettings`, `TonemapSettings`.

## Presets to capture first

`sky-procedural`, `sky-off`, `scene-overview`, `tonemap-reinhard`, `tonemap-agx`,
`tonemap-pbr-neutral`, `tonemap-split`, `cascades-debug`.

## Method

1. **Confirm what is sky and what is geometry.** `sky-off` removes the sky; anything that
   changes between `scene-overview` and `sky-off` is sky, not shadow or lighting. This is
   how ART-SKY-001 was attributed (a speckled horizon band, not shadow acne).
2. Tone map operators are expected to differ substantially — `tonemap-reinhard` vs
   `tonemap-agx` changes ~70% of pixels. A small difference means the operator is not
   being selected; check the `TonemapPC` push constants and the operator dispatch.
3. Verify the output transfer once: the swapchain prefers `B8G8R8A8_SRGB` /
   `R8G8B8A8_SRGB`, and the tone map pass writes to it expecting hardware linear→sRGB
   conversion. If the surface exposes no sRGB format, `Swapchain` logs a warning and falls
   back — that fallback is the one case where output would look washed out.
4. For the sky, remember the capture PNG is post-tone-map: decode (`sRGB⁻¹`, then invert
   Reinhard) before reading numeric values from a debug capture.

## Findings

* ART-SKY-001 open, highest visual impact — the 16-step unjittered march over a hard Earth
  intersection aliases into a speckled band along the horizon.
  Recommended fix: jitter the first sample per pixel with a stable hash, and fix the
  observer altitude comment/code mismatch (`EARTH_R + 0.0002` km is 0.2 m, the comment says
  200 m).
* ART-SKY-002 open — arbitrary sun-disc scale factor.
* ART-SKY-003 rejected — no artifact observed.

## Acceptance for any fix here

* Before/after `scene-overview` and `sky-off` captures; the horizon band must be measurably
  reduced without changing mean scene brightness outside the sky region.
* All three tone map operators still produce distinct output.
* No new validation errors; existing tests still pass.
