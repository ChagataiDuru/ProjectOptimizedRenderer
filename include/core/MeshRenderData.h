#pragma once

#include <cstdint>
#include <glm/glm.hpp>

struct MeshRenderData {
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    glm::vec3 worldBoundsMin = glm::vec3(0.0f);
    glm::vec3 worldBoundsMax = glm::vec3(0.0f);
};
