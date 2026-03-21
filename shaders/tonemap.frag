#version 460 core

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(binding = 0, set = 0) uniform sampler2D hdrInput;

layout(push_constant) uniform TonemapPC {
    uint  tonemapMode;      // 0=Reinhard, 1=AgX, 2=PBR Neutral
    float exposure;         // EV adjustment applied before tone mapping
    uint  splitScreenMode;  // 0=off, 1=on (left=tonemapMode, right=splitRightMode)
    uint  splitRightMode;   // tone map operator for the right half
} params;

// ── Reinhard ──────────────────────────────────────────────────────────────────
vec3 tonemapReinhard(vec3 hdr) {
    return hdr / (hdr + vec3(1.0));
}

// ── AgX ───────────────────────────────────────────────────────────────────────
// Reference: Blender/Filament AgX implementation by Troy Sobotka.
// Four steps: inset rotation → log2 encoding → sigmoid curve → outset rotation.

// Inset matrix: compresses Rec.709 into AgX working space.
// Prevents out-of-gamut clipping and controls hue flight toward white.
// Stored in GLSL column-major order (transposed from the row-major whitepaper notation).
//   result.r = 0.8566*in.r + 0.1373*in.g + 0.1118*in.b
//   result.g = 0.0951*in.r + 0.7612*in.g + 0.0767*in.b
//   result.b = 0.0482*in.r + 0.1014*in.g + 0.8113*in.b
const mat3 AGX_INSET = mat3(
    0.8566, 0.0951, 0.0482,   // column 0
    0.1373, 0.7612, 0.1014,   // column 1
    0.1118, 0.0767, 0.8113    // column 2
);

// Outset matrix: approximate inverse of AGX_INSET; restores gamut volume.
const mat3 AGX_OUTSET = mat3(
     1.1271, -0.1106, -0.0164,   // column 0
    -0.1413,  1.1578, -0.0164,   // column 1
    -0.1413, -0.1106,  1.2519    // column 2
);

// Log2 encoding range in EV stops; maps the HDR working space into [0, 1].
const float AGX_LOG_MIN = -12.47393;
const float AGX_LOG_MAX =   4.026069;

// 6th-order polynomial fit to the AgX sigmoid (Blender community approximation).
// Closely matches the reference 1D LUT across the full [0, 1] normalized log range.
vec3 agxSigmoid(vec3 x) {
    vec3 x2 = x * x;
    vec3 x4 = x2 * x2;
    return  15.5     * x4 * x2
          - 40.14    * x4 * x
          + 31.96    * x4
          -  6.868   * x2 * x
          +  0.4298  * x2
          +  0.1191  * x
          -  0.00232;
}

vec3 tonemapAgX(vec3 color) {
    // Step 1: Rotate into AgX working space
    color = AGX_INSET * color;

    // Step 2: Log2 encode — clamp before log to avoid log2(0) = -inf
    color = max(color, 1e-10);
    color = (log2(color) - AGX_LOG_MIN) / (AGX_LOG_MAX - AGX_LOG_MIN);
    color = clamp(color, 0.0, 1.0);

    // Step 3: Apply sigmoid curve per-channel.
    // Hue shifts are suppressed because the inset matrix pre-mixed channels.
    color = agxSigmoid(color);

    // Step 4: Rotate back to display gamut
    color = AGX_OUTSET * color;

    return clamp(color, 0.0, 1.0);
}

// ── Khronos PBR Neutral ───────────────────────────────────────────────────────
// Reference: Khronos glTF Sample Viewer (MIT licence).
// Properties: exact hue preservation below 0.76 peak, zero modification below toe,
// gradual desaturation toward white at extreme luminance only.

vec3 tonemapPBRNeutral(vec3 color) {
    // Guard against negative values from BRDF precision loss at grazing angles
    color = max(color, vec3(0.0));

    const float startCompression = 0.8 - 0.04;   // = 0.76
    const float desaturation     = 0.15;

    // Step 1: Fresnel toe correction.
    // PBR reflectance adds ~4% floor (F0=0.04). This offsets base colors so they
    // map 1:1 to display values under neutral lighting. Applied only below 0.08.
    float x = min(color.r, min(color.g, color.b));
    float offset = (x < 0.08) ? (x - 6.25 * x * x) : 0.04;
    color -= offset;

    // Step 2: Find peak channel value
    float peak = max(color.r, max(color.g, color.b));

    // Step 3: Pass-through below compression threshold — hue and saturation exact
    if (peak < startCompression) {
        return color;
    }

    // Step 4: Rational highlight compression.
    // Maps [startCompression, ∞) → [startCompression, 1.0) with a smooth curve.
    const float d = 1.0 - startCompression;
    float newPeak = 1.0 - d * d / (peak + d - startCompression);

    // Step 5: Uniform vector rescale — preserves R:G:B hue exactly
    color *= newPeak / peak;

    // Step 6: Targeted desaturation for extreme overexposure.
    // g → 0 for mild highlights, g → 1 for very bright peaks.
    float g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    color = mix(color, vec3(newPeak), g);

    return color;
}

// ── Dispatch ──────────────────────────────────────────────────────────────────
vec3 applyTonemap(vec3 hdr, uint mode) {
    switch (mode) {
        case 1u: return tonemapAgX(hdr);
        case 2u: return tonemapPBRNeutral(hdr);
        default: return tonemapReinhard(hdr);
    }
}

// ── Main ──────────────────────────────────────────────────────────────────────
void main() {
    vec3 hdr = texture(hdrInput, uv).rgb;

    // Exposure: exp2() maps directly to a hardware instruction; avoids pow() overhead
    hdr *= exp2(params.exposure);

    vec3 ldr;

    if (params.splitScreenMode == 1u) {
        // Split-screen: left half = primary operator, right half = comparison operator.
        // Sharp vertical divider drawn at the exact center pixel.
        if (uv.x < 0.5) {
            ldr = applyTonemap(hdr, params.tonemapMode);
        } else {
            ldr = applyTonemap(hdr, params.splitRightMode);
        }

        // 1-pixel white divider line at center.
        // textureSize returns integer texel dimensions; multiplying UV distance
        // by texel width converts fractional coordinates to a pixel-space distance.
        float dividerDist = abs(uv.x - 0.5) * float(textureSize(hdrInput, 0).x);
        if (dividerDist < 1.0) {
            ldr = vec3(1.0);
        }
    } else {
        ldr = applyTonemap(hdr, params.tonemapMode);
    }

    // Write to SRGB swapchain — hardware applies linear→sRGB on write
    outColor = vec4(ldr, 1.0);
}
