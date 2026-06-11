#pragma once

#include <cstdint>
#include <glm/glm.hpp>

struct DirectionalLightData {
    glm::vec3 direction = glm::normalize(glm::vec3(0.577f, 0.577f, 0.577f));
    glm::vec3 color     = glm::vec3(1.0f);
    float intensity     = 3.5f;
    float ambient       = 0.3f;
};

struct ShadowSettings {
    float csmLambda        = 0.5f;
    bool  debugCascades    = false;
    int32_t filterMode     = 0;
    float pcfSpreadRadius  = 2.0f;
    float vsmBleedReduction = 0.2f;
};

struct TonemapSettings {
    int32_t mode           = 0;
    float exposure         = 0.0f;
    bool splitScreen       = false;
    int32_t splitRightMode = 1;
};

struct SkySettings {
    bool enabled = true;
    int32_t mode = 0;
};

struct DebugViewSettings {
    bool wireframe   = false;
    bool showNormals = false;
};

struct CameraData {
    glm::mat4 view       = glm::mat4(1.0f);
    glm::mat4 projection = glm::mat4(1.0f);
    glm::vec3 position   = glm::vec3(0.0f);
    float nearZ          = 0.01f;
    float farZ           = 1000.0f;
};
