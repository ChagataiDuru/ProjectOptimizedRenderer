# ADR-0004: ShadowPass ownership and CSM corrections

## Decision

Shadow rendering ownership now lives in `ShadowPass`. `Renderer` keeps frame orchestration and the existing scene/material descriptor layout, while `ShadowPass` owns shadow images, samplers, UBO, pipelines, matrix updates, VSM blur resources, alpha-mask shadow variants, and debug data.

## Direction convention

`light.direction` means the world-space direction from a shaded surface toward the light source. PBR uses it directly as `L`, the debug sun overlay places the icon along that vector from the camera, and the shadow camera is placed on the light-source side of the cascade center.

## Reverse-Z split fix

CSM slice depths are positive view-space distances, but the camera projection is reverse-Z. Cascade slice unprojection now maps view depth to reverse-Z NDC with:

```cpp
(nearZ * (farZ - d)) / (d * (farZ - nearZ))
```

This maps near depths toward 1 and far depths toward 0 before unprojecting cascade corners.

## Shadow distance

Split calculation uses `ShadowSettings::maxDistance` instead of the camera far plane. The default is 100 units, so Sponza-sized scenes get meaningful cascade redistribution across the full lambda slider range instead of most split precision being spent out to 1000 units.

## Performance changes

Medium quality defaults to 1024 cascades. VSM moment, temp, blur descriptor, and blur pipeline resources are allocated lazily when VSM is first selected and blur work only records in VSM mode.

## Remaining TODOs

- Validate shadow caster culling against more light directions and asset scales; the runtime toggle remains available for correctness debugging.
- Consider separate per-cascade resolution or update frequency once the current ownership boundary is stable.
