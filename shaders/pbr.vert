#version 460 core

// ── Vertex inputs (must match Vertex struct layout) ─────────────────────────
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inTangent;

// ── Per-object data via push constant (updated every draw call) ──────────────
layout(push_constant) uniform PushConstants {
    mat4 model;
} pc;

// ── Per-frame camera data via uniform buffer ─────────────────────────────────
layout(binding = 0, set = 0) uniform CameraData {
    mat4 view;
    mat4 projection;
    mat4 inverseVP;     // inverse(projection * view) — used by sky shader; declared here for layout compatibility
    vec3 cameraPos;     // World-space camera position
} camera;

// ── Output interface block to fragment shader ────────────────────────────────
layout(location = 0) out VS_OUT {
    vec3 worldPos;
    vec3 normal;
    vec2 uv;
} vs_out;

void main() {
    // World-space position (used for lighting in fragment shader)
    vec4 worldPos    = pc.model * vec4(inPosition, 1.0);
    vs_out.worldPos  = worldPos.xyz;

    // World-space normal: mat3(model) is correct for uniform scale + rotation.
    // Replace with transpose(inverse(mat3(model))) if non-uniform scaling is needed.
    vs_out.normal = normalize(mat3(pc.model) * inNormal);

    vs_out.uv = inUV;

    // Clip-space output
    gl_Position = camera.projection * camera.view * worldPos * 2;
}
