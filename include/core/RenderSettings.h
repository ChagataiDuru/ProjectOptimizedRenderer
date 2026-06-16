#pragma once

#include <cstdint>
#include <array>
#include <glm/glm.hpp>

struct DirectionalLightData {
    glm::vec3 direction = glm::normalize(glm::vec3(0.577f, 0.577f, 0.577f));
    glm::vec3 color     = glm::vec3(1.0f);
    float intensity     = 3.5f;
    float ambient       = 0.3f;
};

struct ShadowSettings {
    float csmLambda         = 0.75f;
    float maxDistance       = 100.0f;
    bool  debugCascades     = false;
    int32_t filterMode      = 0;
    float pcfSpreadRadius   = 2.0f;
    float vsmBleedReduction = 0.2f;
    float depthBiasConstant = 1.5f;
    float depthBiasSlope    = 2.0f;
    bool enableCasterCulling = true;
    int32_t qualityPreset   = 1;
    int32_t cullMode        = 0; // 0 none, 1 front, 2 back
};

struct ShadowDebugInfo {
    glm::vec4 splitDepths = glm::vec4(0.0f);
    std::array<uint32_t, 4> culledMeshes = {};
    std::array<uint32_t, 4> totalMeshes = {};
    std::array<uint32_t, 4> drawnMeshes = {};
    uint32_t cascadeResolution = 0;
    uint32_t cascadeCount = 4;
    bool vsmAllocated = false;
    float estimatedMemoryMB = 0.0f;
    float maxDistance = 0.0f;
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

enum class AntiAliasingMode : uint8_t {
    None,
    MSAA,
};

enum class MsaaSampleCount : uint8_t {
    X1 = 1,
    X2 = 2,
    X4 = 4,
    X8 = 8,
};

struct AntiAliasingSettings {
    AntiAliasingMode mode = AntiAliasingMode::None;
    MsaaSampleCount requestedSampleCount = MsaaSampleCount::X1;

    bool sampleShadingEnabled = false;
    float minSampleShading = 0.0f;

    bool alphaToCoverageEnabled = false;
};

struct AntiAliasingStatus {
    AntiAliasingMode mode = AntiAliasingMode::None;
    MsaaSampleCount requestedSampleCount = MsaaSampleCount::X1;
    MsaaSampleCount activeSampleCount = MsaaSampleCount::X1;
    MsaaSampleCount fallbackSampleCount = MsaaSampleCount::X1;
    std::array<bool, 4> supportedSampleCounts = { true, false, false, false };

    bool sampleShadingEnabled = false;
    float minSampleShading = 0.0f;

    bool alphaToCoverageEnabled = false;
};

struct CameraData {
    glm::mat4 view       = glm::mat4(1.0f);
    glm::mat4 projection = glm::mat4(1.0f);
    glm::vec3 position   = glm::vec3(0.0f);
    float nearZ          = 0.01f;
    float farZ           = 1000.0f;
};
