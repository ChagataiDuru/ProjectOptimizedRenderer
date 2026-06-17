#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <glm/glm.hpp>

namespace shader_interface {

inline constexpr uint32_t kSceneSet = 0;
inline constexpr uint32_t kMaterialSet = 1;
inline constexpr uint32_t kTonemapSet = 0;
inline constexpr uint32_t kSkyPanoramaSet = 1;
inline constexpr uint32_t kShadowBlurSet = 0;

namespace scene_binding {
inline constexpr uint32_t kCamera = 0;
inline constexpr uint32_t kLight = 1;
inline constexpr uint32_t kShadow = 2;
inline constexpr uint32_t kShadowMap = 3;
inline constexpr uint32_t kShadowMoments = 4;
inline constexpr uint32_t kPointLights = 5;
inline constexpr uint32_t kClusterGrid = 6;
inline constexpr uint32_t kClusterLightIndices = 7;
inline constexpr uint32_t kClusterMetadata = 8;
}

namespace material_binding {
inline constexpr uint32_t kAlbedo = 0;
inline constexpr uint32_t kNormal = 1;
inline constexpr uint32_t kMetallicRoughness = 2;
}

namespace tonemap_binding {
inline constexpr uint32_t kHdrInput = 0;
}

namespace sky_binding {
inline constexpr uint32_t kPanorama = 0;
}

namespace shadow_blur_binding {
inline constexpr uint32_t kInputImage = 0;
inline constexpr uint32_t kOutputImage = 1;
}

inline constexpr uint32_t kCascadeCount = 4;
inline constexpr uint32_t kModelMatrixPcOffset = 0;
inline constexpr uint32_t kMaterialPcOffset = 64;
inline constexpr uint32_t kCascadeIndexPcOffset = 96;
inline constexpr uint32_t kSkyPcOffset = 0;
inline constexpr uint32_t kShadowBlurPcOffset = 0;
inline constexpr uint32_t kTonemapPcOffset = 0;
inline constexpr uint32_t kClusterTileSizeX = 16;
inline constexpr uint32_t kClusterTileSizeY = 16;
inline constexpr uint32_t kClusterZSlices = 24;
inline constexpr uint32_t kMaxLightsPerCluster = 128;

struct GpuPointLight {
    glm::vec4 positionRadius;   // xyz, radius
    glm::vec4 colorIntensity;   // rgb, intensity
};

struct GpuSpotLight {
    glm::vec4 positionRadius;
    glm::vec4 directionInnerCos;
    glm::vec4 colorOuterCosIntensity;
};

struct ClusterLightRange {
    uint32_t offset;
    uint32_t count;
};

struct ClusterMetadataUBO {
    glm::uvec4 clusterCounts;    // x=countX, y=countY, z=countZ, w=maxLightsPerCluster
    glm::uvec4 lightCounts;      // x=pointLightCount, y=debugMode
    glm::vec4  screenSizeDepth;  // x=width, y=height, z=nearZ, w=farZ
};

struct MaterialPushConstants {
    glm::vec4 baseColorFactor = glm::vec4(1.0f);
    float     metallicFactor  = 1.0f;
    float     roughnessFactor = 1.0f;
    float     alphaCutoff     = 0.0f;
    float     alphaCoverageMode = 0.0f;
};

struct CameraUBO {
    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 inverseVP;
    glm::vec3 cameraPos;
    float     _pad = 0.0f;
};

struct LightUBO {
    glm::vec3 lightDirection;
    float     lightIntensity;
    glm::vec3 lightColor;
    float     ambientIntensity;
    uint32_t  debugCascades;
    uint32_t  shadowFilterMode;
    float     pcfSpreadRadius;
    float     vsmBleedReduction;
};

struct ShadowCascadeUBO {
    glm::mat4 lightViewProj[kCascadeCount];
    glm::vec4 splitDepths;
};

struct TonemapPC {
    uint32_t mode;
    float    exposure;
    uint32_t splitScreenMode;
    uint32_t splitRightMode;
};

static_assert(std::is_standard_layout_v<MaterialPushConstants>);
static_assert(std::is_standard_layout_v<GpuPointLight>);
static_assert(std::is_standard_layout_v<GpuSpotLight>);
static_assert(std::is_standard_layout_v<ClusterLightRange>);
static_assert(std::is_standard_layout_v<ClusterMetadataUBO>);
static_assert(std::is_standard_layout_v<CameraUBO>);
static_assert(std::is_standard_layout_v<LightUBO>);
static_assert(std::is_standard_layout_v<ShadowCascadeUBO>);
static_assert(std::is_standard_layout_v<TonemapPC>);

static_assert(sizeof(MaterialPushConstants) == 32);
static_assert(sizeof(GpuPointLight) == 32);
static_assert(sizeof(GpuSpotLight) == 48);
static_assert(sizeof(ClusterLightRange) == 8);
static_assert(sizeof(ClusterMetadataUBO) == 48);
static_assert(sizeof(CameraUBO) == 208);
static_assert(sizeof(LightUBO) == 48);
static_assert(sizeof(ShadowCascadeUBO) == 272);
static_assert(sizeof(TonemapPC) == 16);

static_assert(offsetof(MaterialPushConstants, baseColorFactor) == 0);
static_assert(offsetof(MaterialPushConstants, metallicFactor) == 16);
static_assert(offsetof(MaterialPushConstants, roughnessFactor) == 20);
static_assert(offsetof(MaterialPushConstants, alphaCutoff) == 24);
static_assert(offsetof(MaterialPushConstants, alphaCoverageMode) == 28);

static_assert(offsetof(CameraUBO, view) == 0);
static_assert(offsetof(CameraUBO, projection) == 64);
static_assert(offsetof(CameraUBO, inverseVP) == 128);
static_assert(offsetof(CameraUBO, cameraPos) == 192);

static_assert(offsetof(LightUBO, lightDirection) == 0);
static_assert(offsetof(LightUBO, lightIntensity) == 12);
static_assert(offsetof(LightUBO, lightColor) == 16);
static_assert(offsetof(LightUBO, ambientIntensity) == 28);
static_assert(offsetof(LightUBO, debugCascades) == 32);
static_assert(offsetof(LightUBO, shadowFilterMode) == 36);
static_assert(offsetof(LightUBO, pcfSpreadRadius) == 40);
static_assert(offsetof(LightUBO, vsmBleedReduction) == 44);

static_assert(offsetof(ShadowCascadeUBO, lightViewProj) == 0);
static_assert(offsetof(ShadowCascadeUBO, splitDepths) == 256);

static_assert(offsetof(TonemapPC, mode) == 0);
static_assert(offsetof(TonemapPC, exposure) == 4);
static_assert(offsetof(TonemapPC, splitScreenMode) == 8);
static_assert(offsetof(TonemapPC, splitRightMode) == 12);

} // namespace shader_interface
