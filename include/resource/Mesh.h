#pragma once
#include "resource/Vertex.h"
#include <glm/glm.hpp>
#include <vector>
#include <string>

struct Mesh {
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    // Index into Model::materials (-1 = no material)
    int32_t materialIndex = -1;

    // Per-mesh axis-aligned bounding box (model space, computed at load time)
    glm::vec3 boundsMin = glm::vec3(0.0f);
    glm::vec3 boundsMax = glm::vec3(0.0f);

    Mesh() = default;
};
