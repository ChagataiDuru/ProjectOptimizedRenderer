#pragma once
#include "resource/Mesh.h"
#include "resource/Material.h"
#include <vector>
#include <string>
#include <glm/glm.hpp>

// Describes what type of data a texture contains (determines VkFormat at upload time).
enum class TextureType {
    Color,   // sRGB — albedo, emissive
    Linear,  // UNORM — normal maps, metallic-roughness, occlusion
};

struct TextureEntry {
    std::string path;  // Absolute file path resolved from glTF URI
    TextureType type;  // Determines VK_FORMAT at load time
};

struct Model {
    std::string               name;
    std::vector<Mesh>         meshes;
    std::vector<Material>     materials;
    std::vector<TextureEntry> textures;  // Deduplicated list of texture file paths

    // Bounding box (computed after loading)
    glm::vec3 boundsMin = glm::vec3(0);
    glm::vec3 boundsMax = glm::vec3(1);

    Model() = default;
    explicit Model(const std::string& modelName) : name(modelName) {}

    size_t getTotalVertexCount() const {
        size_t count = 0;
        for (const auto& mesh : meshes) count += mesh.vertices.size();
        return count;
    }

    size_t getTotalIndexCount() const {
        size_t count = 0;
        for (const auto& mesh : meshes) count += mesh.indices.size();
        return count;
    }
};
