#pragma once

#include "resource/RenderHandles.h"

#include <glm/glm.hpp>

struct DrawCommand {
    glm::mat4      transform = glm::mat4(1.0f);
    MeshHandle     mesh;
    MaterialHandle material;
};
