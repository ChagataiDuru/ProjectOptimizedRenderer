#version 460 core

// ── Input interface block — must match pbr.vert VS_OUT exactly ───────────────
layout(location = 0) in VS_OUT {
    vec3 worldPos;
    vec3 normal;
    vec2 uv;
} fs_in;

// ── Output ───────────────────────────────────────────────────────────────────
layout(location = 0) out vec4 outColor;

// ── Main ─────────────────────────────────────────────────────────────────────
void main() {
    vec3 N = normalize(fs_in.normal);

    // Flip normal for back faces (same as pbr.frag)
    if (!gl_FrontFacing) N = -N;

    // Map from [-1, 1] to [0, 1] for visualization:
    //   +X (right)  → red
    //   +Y (up)     → green
    //   +Z (toward) → blue
    outColor = vec4(N * 0.5 + 0.5, 1.0);
}
