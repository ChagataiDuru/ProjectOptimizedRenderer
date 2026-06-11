#version 460 core

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec2 outMoments;

layout(binding = 0, set = 1) uniform sampler2D texAlbedo;

layout(push_constant) uniform MaterialPC {
    layout(offset = 64) vec4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    float alphaCutoff;
} material;

void main() {
    float alpha = texture(texAlbedo, inUV).a * material.baseColorFactor.a;
    if (material.alphaCutoff > 0.0 && alpha < material.alphaCutoff) {
        discard;
    }

    float depth = gl_FragCoord.z;
    outMoments = vec2(depth, depth * depth);
}
