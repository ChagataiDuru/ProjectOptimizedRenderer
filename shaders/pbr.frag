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
    uint  debugCascades;    // 0 = off, 1 = false-color cascade overlay
    uint  shadowFilterMode; // 0=None, 1=PCF, 2=VSM
    float pcfSpreadRadius;  // PCF kernel spread in texels
    float vsmBleedReduction; // VSM light-bleeding reduction
} light;

layout(binding = 2, set = 0) uniform ShadowData {
    mat4 lightViewProj[4];
    vec4 splitDepths;       // view-space |Z| at cascade far planes: x=c0, y=c1, z=c2, w=c3
} shadow;

// Depth shadow map (D32_SFLOAT array, 4 layers) — used for None and PCF modes.
layout(binding = 3, set = 0) uniform sampler2DArray shadowMap;

// VSM moment map (RG32_SFLOAT array, 4 layers) — used for VSM mode.
layout(binding = 4, set = 0) uniform sampler2DArray shadowMoments;

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

// ── 16-tap Poisson disk offsets (unit disk, golden-ratio rotated) ─────────────
const vec2 POISSON_DISK[16] = vec2[16](
    vec2(-0.94201624, -0.39906216),
    vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870),
    vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432),
    vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845),
    vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554),
    vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023),
    vec2( 0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507),
    vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367),
    vec2( 0.14383161, -0.14100790)
);

// ── Project worldPos into a cascade; return UV, currentDepth, and whether in range ──
bool projectToCascade(vec3 worldPos, int cascadeIdx, out vec2 shadowUV, out float currentDepth) {
    vec4 lightClip = shadow.lightViewProj[cascadeIdx] * vec4(worldPos, 1.0);
    vec3 ndc = lightClip.xyz / lightClip.w;
    shadowUV = ndc.xy * 0.5 + 0.5;
    currentDepth = ndc.z;
    return (shadowUV.x >= 0.0 && shadowUV.x <= 1.0 &&
            shadowUV.y >= 0.0 && shadowUV.y <= 1.0 &&
            currentDepth >= 0.0 && currentDepth <= 1.0);
}

// ── Hard shadow sample (single depth compare) ────────────────────────────────
float sampleShadowHard(vec3 worldPos, int cascadeIdx) {
    vec2 shadowUV;
    float currentDepth;
    if (!projectToCascade(worldPos, cascadeIdx, shadowUV, currentDepth))
        return 1.0;  // outside cascade bounds → fully lit

    ivec3 mapSize = textureSize(shadowMap, 0);
    ivec2 texel   = clamp(ivec2(shadowUV * vec2(mapSize.xy)), ivec2(0), mapSize.xy - 1);
    float storedDepth = texelFetch(shadowMap, ivec3(texel, cascadeIdx), 0).r;

    const float bias = 0.005;
    return (currentDepth - bias) <= storedDepth ? 1.0 : 0.0;
}

// ── PCF: 16-tap Poisson disk (soft shadow) ───────────────────────────────────
float sampleShadowPCF(vec3 worldPos, int cascadeIdx) {
    vec2 shadowUV;
    float currentDepth;
    if (!projectToCascade(worldPos, cascadeIdx, shadowUV, currentDepth))
        return 1.0;

    ivec3 mapSize = textureSize(shadowMap, 0);
    vec2  texelSz = 1.0 / vec2(mapSize.xy);
    float spread  = light.pcfSpreadRadius * texelSz.x;  // spread in UV space

    const float bias = 0.005;
    float lit = 0.0;
    for (int i = 0; i < 16; ++i) {
        vec2 sampleUV = shadowUV + POISSON_DISK[i] * spread;
        sampleUV = clamp(sampleUV, vec2(0.0), vec2(1.0));
        ivec2 texel = ivec2(sampleUV * vec2(mapSize.xy));
        texel = clamp(texel, ivec2(0), mapSize.xy - 1);
        float storedDepth = texelFetch(shadowMap, ivec3(texel, cascadeIdx), 0).r;
        lit += (currentDepth - bias) <= storedDepth ? 1.0 : 0.0;
    }
    return lit / 16.0;
}

// ── VSM: Chebyshev upper bound on probability of being lit ───────────────────
float sampleShadowVSM(vec3 worldPos, int cascadeIdx) {
    vec2 shadowUV;
    float currentDepth;
    if (!projectToCascade(worldPos, cascadeIdx, shadowUV, currentDepth))
        return 1.0;

    // Bilinear sample of blurred moments (hardware LINEAR filter on RG32F)
    vec2 moments = texture(shadowMoments, vec3(shadowUV, float(cascadeIdx))).rg;
    float mean    = moments.x;
    float mean2   = moments.y;

    // Chebyshev: if receiver is clearly in front of all occluders, fully lit
    if (currentDepth <= mean)
        return 1.0;

    // Variance (clamped to prevent numerical artifacts)
    float variance = mean2 - mean * mean;
    const float minVariance = 0.00002;
    variance = max(variance, minVariance);

    // Chebyshev upper bound
    float d   = currentDepth - mean;
    float pMax = variance / (variance + d * d);

    // Light-bleeding reduction: remap pMax so regions deep in shadow stay dark.
    // Subtract threshold and rescale — reduces bleeding at the cost of slightly
    // harder shadow edges at grazing angles.
    float bleed = light.vsmBleedReduction;
    pMax = clamp((pMax - bleed) / (1.0 - bleed), 0.0, 1.0);

    return pMax;
}

// ── Select cascade and sample shadow with optional cascade blending ──────────
// The blend zone is the last 20% of each cascade's depth range.
// In that zone we smoothly interpolate with the next cascade to hide seams.
float sampleCascade(vec3 worldPos, int cascadeIdx) {
    if (light.shadowFilterMode == 0u)
        return sampleShadowHard(worldPos, cascadeIdx);
    else if (light.shadowFilterMode == 1u)
        return sampleShadowPCF(worldPos, cascadeIdx);
    else
        return sampleShadowVSM(worldPos, cascadeIdx);
}

float computeShadow(vec3 worldPos, out int cascadeIdx) {
    // Select cascade by view-space Z magnitude
    float viewZ = abs((camera.view * vec4(worldPos, 1.0)).z);
    cascadeIdx = 3;
    if      (viewZ < shadow.splitDepths.x) cascadeIdx = 0;
    else if (viewZ < shadow.splitDepths.y) cascadeIdx = 1;
    else if (viewZ < shadow.splitDepths.z) cascadeIdx = 2;

    float shadow0 = sampleCascade(worldPos, cascadeIdx);

    // Cascade blending: in the last 20% of this cascade, blend toward the next cascade
    // to hide the hard seam where matrix changes discontinuously.
    if (cascadeIdx < 3) {
        float splitFar = (cascadeIdx == 0) ? shadow.splitDepths.x :
                         (cascadeIdx == 1) ? shadow.splitDepths.y :
                                              shadow.splitDepths.z;
        float blendStart = splitFar * 0.8;
        float blendT = smoothstep(blendStart, splitFar, viewZ);
        if (blendT > 0.0) {
            float shadow1 = sampleCascade(worldPos, cascadeIdx + 1);
            return mix(shadow0, shadow1, blendT);
        }
    }

    return shadow0;
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

    // ── Shadow ───────────────────────────────────────────────────────────
    int cascadeIdx;
    float shadowFactor = computeShadow(fs_in.worldPos, cascadeIdx);
    Lo *= shadowFactor;

    // ── Ambient ──────────────────────────────────────────────────────────
    vec3 ambient = baseColor * light.ambientIntensity;

    vec3 color = ambient + Lo;

    // ── Tone mapping (HDR → LDR) ─────────────────────────────────────────
    color = tonemapReinhard(color);

    // NOTE: No manual gamma correction here.
    // The swapchain is VK_FORMAT_B8G8R8A8_SRGB — the hardware automatically applies
    // the linear→sRGB transfer function on framebuffer write.
    // The previous pow(color, vec3(1.0/2.2)) was double-gamma-correcting.

    // ── Cascade debug overlay ─────────────────────────────────────────────
    if (light.debugCascades != 0u) {
        const vec3 cascadeColors[4] = vec3[4](
            vec3(1.0, 0.3, 0.3),   // cascade 0 = red
            vec3(0.3, 1.0, 0.3),   // cascade 1 = green
            vec3(0.3, 0.3, 1.0),   // cascade 2 = blue
            vec3(1.0, 1.0, 0.3)    // cascade 3 = yellow
        );
        // Show blend zone by brightening toward cascade+1 color at zone boundary
        float viewZ    = abs((camera.view * vec4(fs_in.worldPos, 1.0)).z);
        float splitFar = (cascadeIdx == 0) ? shadow.splitDepths.x :
                         (cascadeIdx == 1) ? shadow.splitDepths.y :
                         (cascadeIdx == 2) ? shadow.splitDepths.z :
                                              shadow.splitDepths.w;
        float blendT = (cascadeIdx < 3)
            ? smoothstep(splitFar * 0.8, splitFar, viewZ)
            : 0.0;
        vec3 baseOverlay = cascadeColors[cascadeIdx];
        vec3 nextOverlay = (cascadeIdx < 3) ? cascadeColors[cascadeIdx + 1] : baseOverlay;
        vec3 overlay = mix(baseOverlay, nextOverlay, blendT);
        color = mix(color, overlay, 0.4);
    }

    outColor = vec4(color, alpha);
}
