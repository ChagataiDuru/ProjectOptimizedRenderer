#pragma once

#include <glm/glm.hpp>
#include <cstdint>

struct MeshRenderData {
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    glm::vec3 boundsMin = glm::vec3(0.0f);
    glm::vec3 boundsMax = glm::vec3(0.0f);
};
