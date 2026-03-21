#version 460 core

// Fullscreen triangle — no vertex buffer, no vertex input state.
// gl_VertexIndex drives UV generation:
//   index 0 → uv=(0,0), pos=(-1,-1)
//   index 1 → uv=(2,0), pos=( 3,-1)
//   index 2 → uv=(0,2), pos=(-1, 3)
// The oversized triangle is clipped to the viewport; every screen pixel is covered exactly once.

layout(location = 0) out vec2 uv;

void main() {
    uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
