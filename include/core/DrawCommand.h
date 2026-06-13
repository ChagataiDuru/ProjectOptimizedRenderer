#pragma once

#include "resource/RenderHandles.h"

#include <glm/glm.hpp>

struct DrawCommand {
    glm::mat4      transform = glm::mat4(1.0f);
    MeshHandle     mesh;
    MaterialHandle material;
};

// Renderer-derived world-space bounds aligned 1:1 with active draw submission.
// These bounds are rebuilt when submitScene() activates a sticky scene packet.
struct SceneDrawBounds {
    glm::vec3 worldMin = glm::vec3(0.0f);
    glm::vec3 worldMax = glm::vec3(0.0f);
};
