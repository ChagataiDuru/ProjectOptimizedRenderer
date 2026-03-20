#version 460 core

// ── Inputs from vertex shader ─────────────────────────────────────────────────
layout(location = 0) in VS_OUT {
    vec3 worldPos;
    vec3 normal;
    vec2 uv;
} fs_in;

// ── Per-frame uniform bindings (set 0) ───────────────────────────────────────
layout(binding = 0, set = 0) uniform CameraData {
    mat4 view;
    mat4 projection;
    vec3 cameraPos;
} camera;

layout(binding = 1, set = 0) uniform LightData {
    vec3  lightDirection;   // world-space, pointing TOWARD the scene (like the sun)
    float lightIntensity;
    vec3  lightColor;
    float ambientIntensity;
} light;

// ── Material textures (set 1) ─────────────────────────────────────────────────
layout(binding = 0, set = 1) uniform sampler2D texAlbedo;
layout(binding = 1, set = 1) uniform sampler2D texNormal;
layout(binding = 2, set = 1) uniform sampler2D texMetallicRoughness;

// ── Material factors via push constants ──────────────────────────────────────
// offset = 64: this block starts after the 64-byte model matrix (vertex stage).
layout(push_constant) uniform MaterialPC {
    layout(offset = 64) vec4  baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    float alphaCutoff;
} material;

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

// ── Fresnel-Schlick Approximation ────────────────────────────────────────────
// F0 = base reflectivity at normal incidence (0.04 for dielectrics, tinted for metals).
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ── Smith Geometry Function ──────────────────────────────────────────────────
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

// ── Cook-Torrance BRDF ───────────────────────────────────────────────────────
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

// ── Reinhard Tone Mapping ────────────────────────────────────────────────────
vec3 tonemapReinhard(vec3 hdr) {
    return hdr / (hdr + vec3(1.0));
}

// ── Main ──────────────────────────────────────────────────────────────────────
void main() {
    // ── Sample material textures ─────────────────────────────────────────
    // Albedo: texture is SRGB, hardware decodes to linear on sample.
    // Multiply by baseColorFactor (glTF spec: factor × texture).
    vec4  albedoSample = texture(texAlbedo, fs_in.uv) * material.baseColorFactor;
    vec3  baseColor    = albedoSample.rgb;
    float alpha        = albedoSample.a;

    // Alpha masking: discard fragments below the cutoff.
    // Without this, alpha-masked materials (vegetation, chains, curtain edges)
    // write black pixels from their texture's masked regions, occluding geometry behind.
    if (material.alphaCutoff > 0.0 && alpha < material.alphaCutoff)
        discard;

    // Metallic-roughness: green = roughness, blue = metallic (glTF convention).
    // Texture is UNORM (linear data). Multiply by material factors.
    vec4  mrSample  = texture(texMetallicRoughness, fs_in.uv);
    float metallic  = mrSample.b * material.metallicFactor;
    float roughness = mrSample.g * material.roughnessFactor;

    // Clamp roughness to avoid division-by-zero in GGX (perfectly smooth = NaN)
    roughness = clamp(roughness, 0.04, 1.0);

    // ── Normal mapping ───────────────────────────────────────────────────
    vec3 N = normalize(fs_in.normal);
    if (!gl_FrontFacing) N = -N;
    vec3 normalSample = texture(texNormal, fs_in.uv).rgb;
    // Only apply normal map if the texture is not the 1x1 white fallback
    if (normalSample != vec3(1.0)) {
        // Decode from [0,1] to [-1,1]
        vec3 tangentNormal = normalSample * 2.0 - 1.0;

        // Reconstruct TBN from screen-space derivatives — avoids dependence on
        // vertex tangent quality (some glTF exporters produce unreliable tangents).
        vec3 Q1  = dFdx(fs_in.worldPos);
        vec3 Q2  = dFdy(fs_in.worldPos);
        vec2 st1 = dFdx(fs_in.uv);
        vec2 st2 = dFdy(fs_in.uv);

        vec3 T   = normalize(Q1 * st2.t - Q2 * st1.t);
        vec3 B   = normalize(cross(N, T));
        mat3 TBN = mat3(T, B, N);

        N = normalize(TBN * tangentNormal);
    }

    // ── View vector (from surface toward camera) ─────────────────────────
    // Uses actual camera world position — fixes specular for moving cameras.
    vec3 V = normalize(camera.cameraPos - fs_in.worldPos);

    // ── Light vector (from surface toward light source) ──────────────────
    vec3 L = normalize(light.lightDirection);

    // ── Direct lighting ──────────────────────────────────────────────────
    vec3 Lo = cookTorrance(N, L, V, baseColor, metallic, roughness);

    // ── Ambient ──────────────────────────────────────────────────────────
    vec3 ambient = baseColor * light.ambientIntensity;

    vec3 color = ambient + Lo;

    // ── Tone mapping (HDR → LDR) ─────────────────────────────────────────
    color = tonemapReinhard(color);

    // NOTE: No manual gamma correction here.
    // The swapchain is VK_FORMAT_B8G8R8A8_SRGB — the hardware automatically applies
    // the linear→sRGB transfer function on framebuffer write.
    // The previous pow(color, vec3(1.0/2.2)) was double-gamma-correcting.

    outColor = vec4(color, alpha);
}
