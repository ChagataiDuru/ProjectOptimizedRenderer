#pragma once
#include "resource/Mesh.h"
#include <vector>
#include <string>
#include <glm/glm.hpp>

struct Model {
    std::string       name;
    std::vector<Mesh> meshes;

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
