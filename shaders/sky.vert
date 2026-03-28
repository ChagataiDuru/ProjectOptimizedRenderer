#version 460 core

// ── Camera data (CameraUBO) — binding 0, set 0 ────────────────────────────────
// Provides inverseVP to reconstruct a world-space ray direction from clip-space position.
layout(binding = 0, set = 0) uniform CameraData {
    mat4 view;
    mat4 projection;
    mat4 inverseVP;    // inverse(projection * view): NDC → world-space
    vec3 cameraPos;    // world-space camera origin
} camera;

layout(location = 0) out vec3 outRayDir;

void main()
{
    // Fullscreen triangle (no vertex buffer).
    //   vertex 0 → uv=(0,0) → NDC=(-1,-1)
    //   vertex 1 → uv=(2,0) → NDC=( 3,-1)
    //   vertex 2 → uv=(0,2) → NDC=(-1, 3)
    // The triangle completely covers the NDC square [-1,1]^2.
    vec2 uv      = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vec4 clipPos = vec4(uv * 2.0 - 1.0, 0.0, 1.0);

    // Unproject: inverse(P*V) maps homogeneous clip-space → world-space.
    vec4 worldPos = camera.inverseVP * clipPos;

    // Perspective-correct world position, then subtract camera origin for ray direction.
    outRayDir = normalize(worldPos.xyz / worldPos.w - camera.cameraPos);

    // z = 0.0 → reverse-Z far plane (depth buffer cleared to 0.0).
    // depthCompareOp = GREATER_OR_EQUAL passes where stored depth == 0.0 (no geometry).
    gl_Position = vec4(clipPos.xy, 0.0, 1.0);
}
