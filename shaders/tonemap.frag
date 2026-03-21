#version 460 core

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(binding = 0, set = 0) uniform sampler2D hdrInput;

layout(push_constant) uniform TonemapPC {
    uint  tonemapMode;   // 0=Reinhard (Phase 6 adds AgX, PBR Neutral)
    float exposure;      // EV adjustment applied before tone mapping
} params;

vec3 tonemapReinhard(vec3 hdr) {
    return hdr / (hdr + vec3(1.0));
}

void main() {
    vec3 hdr = texture(hdrInput, uv).rgb;

    // Exposure: multiply by 2^EV before tone mapping
    hdr *= pow(2.0, params.exposure);

    // Tone map (mode switch — Phase 6 adds more operators)
    vec3 ldr = tonemapReinhard(hdr);  // default / only mode for now

    // Write to SRGB swapchain — hardware applies linear→sRGB on write
    outColor = vec4(ldr, 1.0);
}
