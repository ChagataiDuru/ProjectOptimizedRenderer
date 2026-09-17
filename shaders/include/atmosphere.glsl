// Procedural Rayleigh + Mie single-scattering sky, shared by sky.frag and the IBL bake.
// Requires GL_GOOGLE_include_directive in the including shader.
#ifndef POR_ATMOSPHERE_GLSL
#define POR_ATMOSPHERE_GLSL

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

// Ray-sphere intersection for a ray that starts at altitude h (km) above the Earth's
// surface, i.e. at (0, EARTH_R + h, 0), against a sphere of radius r centred on the
// Earth. Returns (t_near, t_far); both negative on a miss.
//
// Both coefficients are formed without cancellation. The textbook
// c = dot(ro, ro) - r * r subtracts two ~4e7 float32 values whose difference near the
// ground is ~2.5 km^2, below float32 resolution at that magnitude, so rays near the
// horizon flipped between hit and miss per pixel (ART-SKY-001). The roots use the
// stable q = -(b + sign(b) * sqrt(disc)) form for the same reason.
vec2 raySphereFromAltitude(float h, vec3 rd, float r)
{
    float ro   = EARTH_R + h;
    float b    = ro * rd.y;
    float c    = (h - (r - EARTH_R)) * (ro + r);
    float disc = b * b - c;
    if (disc < 0.0) return vec2(-1.0);
    float q = -(b + (b >= 0.0 ? 1.0 : -1.0) * sqrt(disc));
    if (abs(q) < 1e-12) return vec2(0.0);
    float t0 = q;
    float t1 = c / q;
    return vec2(min(t0, t1), max(t0, t1));
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
vec3 atmosphericScattering(vec3 rd, vec3 sunDir, float sunIntensity, bool includeSunDisc)
{
    // Observer 0.2 m (0.0002 km) above the surface, centred on Earth's Y axis. Rays
    // below the horizon hit the ground within ~1.6 km and stay dark, which also keeps
    // the lower hemisphere of the IBL bake dark. (At 200 m those rays would integrate
    // ~50 km of air and turn the ground into bright blue haze; ground albedo is not
    // modelled, so that is not more correct.)
    const float OBSERVER_ALT = 0.0002;
    const vec3 origin = vec3(0.0, EARTH_R + OBSERVER_ALT, 0.0);

    // Find ray segment inside the atmosphere.
    vec2 atmoHit = raySphereFromAltitude(OBSERVER_ALT, rd, ATMOS_R);
    if (atmoHit.y < 0.0) return vec3(0.0);    // ray misses atmosphere

    float tMin = max(atmoHit.x, 0.0);          // start at observer (inside atmosphere)
    float tMax = atmoHit.y;

    // Clip against Earth surface for downward-looking rays.
    vec2 earthHit = raySphereFromAltitude(OBSERVER_ALT, rd, EARTH_R);
    if (earthHit.x > 0.0) tMax = min(tMax, earthHit.x);

    float dt = (tMax - tMin) / float(STEPS);

    float cosTheta = dot(rd, sunDir);
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
    float sunPower = max(sunIntensity, 0.0) * 22.0;

    vec3 color = sunPower * (phaseR * BETA_R * sumR + phaseM * (BETA_M * 1.1) * sumM);

    // ── Sun disc ──────────────────────────────────────────────────────────────
    // A narrow angular spike near the sun direction (cosTheta ≈ 1).
    // Attenuated by atmospheric extinction along the direct view ray toward the sun.
    if (includeSunDisc) {
        float sunDisc = smoothstep(0.9993, 0.9999, cosTheta);
        vec3  discExt = exp(-(BETA_R * optR + BETA_M * optM));
        color += sunDisc * sunPower * 0.2 * discExt;
    }

    return max(color, vec3(0.0));
}

#endif // POR_ATMOSPHERE_GLSL
