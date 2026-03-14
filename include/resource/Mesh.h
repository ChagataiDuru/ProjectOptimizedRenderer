#pragma once
#include "resource/Vertex.h"
#include <vector>
#include <string>
#include <glm/glm.hpp>

struct Mesh {
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    // Material parameters (basic, will expand in Phase 1.4)
    glm::vec3 baseColor  = glm::vec3(0.8f);
    float     metallic   = 0.0f;
    float     roughness  = 0.5f;

    Mesh() = default;
    Mesh(const std::string& meshName, const std::vector<Vertex>& verts, const std::vector<uint32_t>& inds)
        : name(meshName), vertices(verts), indices(inds) {}
};
