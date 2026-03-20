#version 460 core

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec3 inTangent;

layout(push_constant) uniform PushConstants { mat4 model; } pc;

layout(binding = 2, set = 0) uniform ShadowData { mat4 lightViewProj; } shadow;

void main() {
    gl_Position = shadow.lightViewProj * pc.model * vec4(inPosition, 1.0);
}
