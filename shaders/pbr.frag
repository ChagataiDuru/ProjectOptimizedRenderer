#version 460 core

// ── Inputs from vertex shader ─────────────────────────────────────────────────
layout(location = 0) in VS_OUT {
    vec3 worldPos;
    vec3 normal;
    vec2 uv;
} fs_in;

// ── Uniform bindings (set 0) ──────────────────────────────────────────────────
layout(binding = 1, set = 0) uniform LightData {
    vec3  lightDirection;   // world-space, pointing TOWARD the scene (like the sun)
    float lightIntensity;
    vec3  lightColor;
    float ambientIntensity;
} light;

// ── Output ────────────────────────────────────────────────────────────────────
layout(location = 0) out vec4 outColor;

// ── Physical constants ────────────────────────────────────────────────────────
const float PI = 3.14159265359;

// ── GGX Normal Distribution Function ─────────────────────────────────────────
// Measures how many microfacets are aligned with the half-vector H.
// High roughness = broad highlight, low roughness = sharp specular peak.
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float NdotH  = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;
    float denom  = (NdotH2 * (a2 - 1.0) + 1.0);
    return a2 / max(PI * denom * denom, 0.0001);
}

// ── Fresnel-Schlick Approximation ─────────────────────────────────────────────
// F0 = base reflectivity at normal incidence (0.04 for dielectrics, tinted for metals).
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ── Smith Geometry Function ───────────────────────────────────────────────────
// Models self-shadowing and masking of microfacets (both view and light side).
float geometrySchlickGGX(float NdotV, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;   // Direct lighting k (differs from IBL variant)
    return NdotV / max(NdotV * (1.0 - k) + k, 0.0001);
}

float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    return geometrySchlickGGX(NdotV, roughness) *
           geometrySchlickGGX(NdotL, roughness);
}

// ── Cook-Torrance BRDF ────────────────────────────────────────────────────────
// Returns outgoing radiance for one directional light hit.
vec3 cookTorrance(vec3 N, vec3 L, vec3 V,
                  vec3 baseColor, float metallic, float roughness) {
    vec3 H = normalize(L + V);

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);

    // Base reflectivity: 0.04 for dielectrics, baseColor for metals
    vec3 F0 = mix(vec3(0.04), baseColor, metallic);

    float NDF = distributionGGX(N, H, roughness);
    float G   = geometrySmith(N, V, L, roughness);
    vec3  F   = fresnelSchlick(max(dot(H, V), 0.0), F0);

    // kS = specular fraction (Fresnel), kD = diffuse fraction (energy conservation)
    vec3 kS = F;
    vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);

    vec3 specular = (NDF * G * F) / max(4.0 * NdotV * NdotL + 0.0001, 0.0001);

    vec3 radiance = light.lightColor * light.lightIntensity;
    return (kD * baseColor / PI + specular) * radiance * NdotL;
}

// ── Reinhard Tone Mapping ─────────────────────────────────────────────────────
vec3 tonemapReinhard(vec3 hdr) {
    return hdr / (hdr + vec3(1.0));
}

// ── Main ──────────────────────────────────────────────────────────────────────
void main() {
    vec3 N = normalize(fs_in.normal);
    // View vector: direction from surface toward camera.
    // NOTE: assumes camera is near world origin — will be improved when
    // camera position is added to the CameraUBO in a later phase.
    vec3 V = normalize(-fs_in.worldPos);
    // Light vector: direction from surface toward the light source
    vec3 L = normalize(light.lightDirection);

    // Placeholder material — will be driven by glTF textures in Phase 1.6
    vec3  baseColor = vec3(0.8);
    float metallic  = 0.0;
    float roughness = 0.5;

    // Direct lighting
    vec3 Lo = cookTorrance(N, L, V, baseColor, metallic, roughness);

    // Simple ambient (constant indirect approximation)
    vec3 ambient = baseColor * light.ambientIntensity;

    vec3 color = ambient + Lo;

    // Tone map HDR -> LDR, then gamma correct
    color = tonemapReinhard(color);
    color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, 1.0);
}
