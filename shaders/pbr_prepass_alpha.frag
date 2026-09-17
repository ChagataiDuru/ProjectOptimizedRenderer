#version 460 core

// Depth prepass for alpha-tested materials: no colour output, only the same
// alpha cutout as pbr.frag so the prepass depth matches the main pass exactly.

layout(location = 0) in VS_OUT {
    vec3 worldPos;
    vec3 normal;
    vec2 uv;
} fs_in;

layout(binding = 0, set = 1) uniform sampler2D texAlbedo;

layout(push_constant) uniform MaterialPC {
    layout(offset = 64) vec4  baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    float alphaCutoff;
    float alphaCoverageMode;
} material;

void main()
{
    float alpha = texture(texAlbedo, fs_in.uv).a * material.baseColorFactor.a;
    if (material.alphaCutoff > 0.0 && alpha < material.alphaCutoff) {
        discard;
    }
}
