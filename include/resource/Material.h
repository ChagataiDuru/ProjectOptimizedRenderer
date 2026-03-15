#pragma once

#include <glm/glm.hpp>
#include <string>
#include <cstdint>

// Indices into Model::textures / Renderer::m_textures.
// -1 means "no texture assigned — use fallback."
struct Material {
    std::string name;

    // PBR metallic-roughness workflow
    int32_t albedoTextureIndex            = -1;  // base_color_texture
    int32_t normalTextureIndex            = -1;  // normal_texture
    int32_t metallicRoughnessTextureIndex = -1;  // metallic_roughness_texture

    // Factors (multiplied with texture samples, or used directly if no texture)
    glm::vec4 baseColorFactor = glm::vec4(1.0f);  // RGBA
    float     metallicFactor  = 1.0f;
    float     roughnessFactor = 1.0f;

    // Alpha mode (for future use — transparency, cutout)
    enum class AlphaMode { Opaque, Mask, Blend };
    AlphaMode alphaMode   = AlphaMode::Opaque;
    float     alphaCutoff = 0.5f;

    // Double-sided flag from glTF (already handled via VK_CULL_MODE_NONE,
    // but stored for future per-material culling)
    bool doubleSided = false;
};
