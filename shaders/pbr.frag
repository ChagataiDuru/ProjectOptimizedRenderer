#version 460 core

// ── Input interface block (must match VS_OUT in pbr.vert) ────────────────────
layout(location = 0) in VS_OUT {
    vec3 worldPos;
    vec3 normal;
    vec2 uv;
} fs_in;

// ── Output ───────────────────────────────────────────────────────────────────
layout(location = 0) out vec4 outColor;

void main() {
    // Re-normalize after rasterizer interpolation (interpolated normals drift
    // slightly off unit length across a triangle).
    vec3 normal = normalize(fs_in.normal);

    // Remap [-1, 1] → [0, 1] for debug visualization.
    // Face toward +X → red, +Y → green, +Z → blue.
    vec3 color = normal * 0.5 + 0.5;

    outColor = vec4(color, 1.0);
}
