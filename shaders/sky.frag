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

// ── Equirectangular panorama sampling ─────────────────────────────────────────
// Standard equirectangular (latitude-longitude) layout:
//   u ∈ [0,1] → azimuth [-π, π],   u=0 → west,  u=0.5 → east
//   v ∈ [0,1] → elevation [+π/2, -π/2],  v=0 → zenith, v=1 → nadir
vec3 samplePanorama(vec3 rd)
{
    float azimuth   = atan(rd.z, rd.x);                // [-π, π]
    float elevation = asin(clamp(rd.y, -1.0, 1.0));   // [-π/2, π/2]
    vec2  uv = vec2(azimuth / (2.0 * PI) + 0.5,
                    0.5 - elevation / PI);
    return texture(panorama, uv).rgb;
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
