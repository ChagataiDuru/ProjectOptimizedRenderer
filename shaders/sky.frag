#version 460 core

layout(location = 0) in  vec3 inRayDir;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform SkyPC {
    uint skyMode;   // 0 = Procedural Rayleigh+Mie,  1 = HDR equirectangular panorama
} params;

// ── Set 0: shared scene descriptors (same layout as PBR pipeline) ─────────────
// Binding 0 is camera — declared in sky.vert, not needed here.
// Binding 1: directional light — provides sun direction and intensity.
layout(binding = 1, set = 0) uniform LightData {
    vec3  lightDirection;   // world-space direction pointing TOWARD the sun
    float lightIntensity;
} light;

// ── Set 1: equirectangular HDR panorama ───────────────────────────────────────
// In procedural mode (skyMode=0) a 1×1 white dummy texture is bound here.
layout(binding = 0, set = 1) uniform sampler2D panorama;

// ── Constants ─────────────────────────────────────────────────────────────────
const float PI = 3.14159265358979323846;

// Atmosphere geometry — all distances in kilometres.
const float EARTH_R = 6371.0;   // Earth radius (km)
const float ATMOS_R = 6471.0;   // Atmosphere outer boundary (100 km thick)

// Scattering coefficients at sea level (km⁻¹).
// Rayleigh: shorter wavelengths scatter more → blue sky.
// BETA_R[0]=red, BETA_R[1]=green, BETA_R[2]=blue  (22.4e-3 ≫ 5.5e-3 → Rayleigh prefers blue).
const vec3  BETA_R = vec3(5.5e-3, 13.0e-3, 22.4e-3);  // Rayleigh (km⁻¹)
const float BETA_M = 21.0e-3;                          // Mie (km⁻¹, wavelength-independent)
const float G_MIE  = 0.758;   // Henyey-Greenstein asymmetry: >0 = forward-scattering

// Scale heights (density falls off exponentially with altitude)
const float HR = 8.0;   // Rayleigh scale height (km)
const float HM = 1.2;   // Mie scale height (km)

const int STEPS = 16;   // Ray-march samples

// ── Helpers ───────────────────────────────────────────────────────────────────

// Ray-sphere intersection. Returns (t_near, t_far). Both negative on miss.
vec2 raySphere(vec3 ro, vec3 rd, float r)
{
    float b = dot(ro, rd);
    float c = dot(ro, ro) - r * r;
    float h = b * b - c;
    if (h < 0.0) return vec2(-1.0);
    h = sqrt(h);
    return vec2(-b - h, -b + h);
}

// Henyey-Greenstein Mie phase function.
// G_MIE > 0 concentrates energy in the forward (sun) direction → halo/corona.
float miePhase(float cosTheta)
{
    float g2 = G_MIE * G_MIE;
    return (1.0 - g2) / (pow(max(1.0 + g2 - 2.0 * G_MIE * cosTheta, 1e-4), 1.5) * 4.0 * PI);
}

// Rayleigh phase: symmetric dipole pattern, stronger at forward/backward angles.
float rayleighPhase(float cosTheta)
{
    return (3.0 / (16.0 * PI)) * (1.0 + cosTheta * cosTheta);
}

// ── Rayleigh+Mie single-scattering integral ───────────────────────────────────
// Integrates in-scattered sunlight along the view ray through the atmosphere.
// Returns linear-HDR RGB sky colour (not tone-mapped — the tone map pass handles that).
vec3 atmosphericScattering(vec3 rd)
{
    // Observer 200 m above surface, centred on Earth's Y axis.
    // Small offset avoids numerical artefacts at exactly sea level.
    const vec3 origin = vec3(0.0, EARTH_R + 0.0002, 0.0);

    // Find ray segment inside the atmosphere.
    vec2 atmoHit = raySphere(origin, rd, ATMOS_R);
    if (atmoHit.y < 0.0) return vec3(0.0);    // ray misses atmosphere

    float tMin = max(atmoHit.x, 0.0);          // start at observer (inside atmosphere)
    float tMax = atmoHit.y;

    // Clip against Earth surface for downward-looking rays.
    vec2 earthHit = raySphere(origin, rd, EARTH_R);
    if (earthHit.x > 0.0) tMax = min(tMax, earthHit.x);

    float dt = (tMax - tMin) / float(STEPS);

    float cosTheta = dot(rd, light.lightDirection);
    float phaseR   = rayleighPhase(cosTheta);
    float phaseM   = miePhase(cosTheta);

    vec3  sumR = vec3(0.0);   // Rayleigh in-scatter accumulator
    vec3  sumM = vec3(0.0);   // Mie in-scatter accumulator
    float optR = 0.0;         // Rayleigh optical depth along view ray (km)
    float optM = 0.0;         // Mie optical depth along view ray (km)

    for (int i = 0; i < STEPS; ++i) {
        float t   = tMin + (float(i) + 0.5) * dt;
        vec3  pos = origin + t * rd;
        float h   = length(pos) - EARTH_R;     // altitude above surface (km)

        // Density × step length (contributes to optical depth accumulator)
        float sR  = exp(-h / HR) * dt;
        float sM  = exp(-h / HM) * dt;
        optR += sR;
        optM += sM;

        // Approximate sun-path optical depth from sample point upward toward the sun.
        // Uses a vertical integral exp(-h/H)*H as a fast closed-form estimate
        // (accurate when the sun is not at the horizon).
        float sunOptR = exp(-h / HR) * HR;
        float sunOptM = exp(-h / HM) * HM;

        // Total extinction: view-ray path + sun path.
        // Factor 1.1 on Mie accounts for absorption (Mie extinction ≈ 1.1 × scattering).
        vec3 tau  = BETA_R * (optR + sunOptR) + (BETA_M * 1.1) * (optM + sunOptM);
        vec3 attn = exp(-tau);

        sumR += sR * attn;
        sumM += sM * attn;
    }

    // Sun intensity drives overall sky brightness (matches scene light intensity).
    float sunPower = max(light.lightIntensity, 0.0) * 22.0;

    vec3 color = sunPower * (phaseR * BETA_R * sumR + phaseM * (BETA_M * 1.1) * sumM);

    // ── Sun disc ──────────────────────────────────────────────────────────────
    // A narrow angular spike near the sun direction (cosTheta ≈ 1).
    // Attenuated by atmospheric extinction along the direct view ray toward the sun.
    float sunDisc = smoothstep(0.9993, 0.9999, cosTheta);
    vec3  discExt = exp(-(BETA_R * optR + BETA_M * optM));
    color += sunDisc * sunPower * 0.2 * discExt;

    return max(color, vec3(0.0));
}

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
        color = atmosphericScattering(rd);
    }

    outColor = vec4(color, 1.0);
}
