#version 460 core
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in  vec3 inRayDir;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform SkyPC {
    uint skyMode;   // 0 = Procedural Rayleigh+Mie,  1 = HDR equirectangular panorama
} params;

// ── Set 0: shared scene descriptors (same layout as PBR pipeline) ─────────────
// Binding 0 is camera — declared in sky.vert, not needed here.
// Binding 1: directional light — provides sun direction and intensity.
// ShaderInterface.h: scene set/binding 1 owns LightData.
layout(binding = 1, set = 0) uniform LightData {
    vec3  lightDirection;   // world-space direction from the surface toward the sun
    float lightIntensity;
} light;

// ── Set 1: equirectangular HDR panorama ───────────────────────────────────────
// In procedural mode (skyMode=0) a 1×1 white dummy texture is bound here.
// ShaderInterface.h: sky panorama set/binding 0 owns panorama.
layout(binding = 0, set = 1) uniform sampler2D panorama;

#include "atmosphere.glsl"
#include "equirect.glsl"

// ── Equirectangular panorama sampling ─────────────────────────────────────────
// The panorama has a mip chain, so the u = 0/1 seam needs care: implicit derivatives
// jump there and would select the smallest mip along a visible line. Use whichever of
// u and fract(u + 0.5) is continuous at this pixel to form the gradients (Tarini).
vec3 samplePanorama(vec3 rd)
{
    vec2 uv = dirToEquirect(rd);
    vec2 uvShifted = vec2(fract(uv.x + 0.5), uv.y);
    vec2 dx = dFdx(uv);
    vec2 dy = dFdy(uv);
    vec2 dxShifted = dFdx(uvShifted);
    vec2 dyShifted = dFdy(uvShifted);
    if (abs(dxShifted.x) + abs(dyShifted.x) < abs(dx.x) + abs(dy.x)) {
        dx = dxShifted;
        dy = dyShifted;
    }
    return textureGrad(panorama, uv, dx, dy).rgb;
}

// ── Main ──────────────────────────────────────────────────────────────────────
void main()
{
    vec3 rd    = normalize(inRayDir);
    vec3 color;

    if (params.skyMode == 1u) {
        color = samplePanorama(rd);
    } else {
        color = atmosphericScattering(rd, light.lightDirection, light.lightIntensity, true);
    }

    outColor = vec4(color, 1.0);
}
