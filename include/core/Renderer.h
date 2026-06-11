#pragma once

#include "core/VulkanContext.h"
#include "core/Swapchain.h"
#include "core/CommandBuffer.h"
#include "core/DrawCommand.h"
#include "core/FrameSync.h"
#include "core/MeshRenderData.h"
#include "core/RenderFramePacket.h"
#include "core/ShaderInterface.h"
#include "debug/GPUTimer.h"
#include "debug/Screenshot.h"
#include "renderpasses/ShadowPass.h"
#include "renderpasses/SkyPass.h"
#include "renderpasses/TonemapPass.h"
#include "renderpasses/ClusteredLightCullingPass.h"
#include "resource/Buffer.h"
#include "resource/Image.h"
#include "resource/Model.h"
#include "resource/SceneInfo.h"
#include "resource/Texture.h"
#include "resource/SamplerCache.h"
#include <glm/glm.hpp>
#include <array>
#include <string>
#include <vector>

class ImGuiManager;  // forward declare — full header pulled in by Renderer.cpp

class Renderer {
public:
    Renderer(VulkanContext& ctx, Swapchain& swapchain);
    ~Renderer();

    Renderer(const Renderer&)            = delete;
    Renderer& operator=(const Renderer&) = delete;

    void init();
    void shutdown();

    // Wait for frame slot, collect GPU timer results, reset command buffer, acquire image.
    // Returns false if the frame should be skipped (e.g. window minimized).
    // Caller must NOT call endFrame() when beginFrame() returns false.
    bool beginFrame();

    // Record commands, submit, present, advance frame indices.
    void endFrame();

    void submitFrame(const RenderFramePacket& packet);

    void setCameraMatrices(const glm::mat4& view, const glm::mat4& projection,
                           const glm::vec3& cameraPos);

    // Phase 4.3: expose camera frustum planes so CSM splits stay in sync
    void setCameraFrustum(float nearZ, float farZ);

    void setLightParameters(const glm::vec3& direction, const glm::vec3& color,
                            float intensity, float ambient);
    void setPointLights(const std::vector<shader_interface::GpuPointLight>& pointLights);

    // Phase 4.3: CSM controls
    void setCsmLambda(float lambda)            { m_shadowSettings.csmLambda = lambda; }
    void setCascadeDebugEnabled(bool enabled);
    bool isCascadeDebugEnabled() const         { return m_shadowSettings.debugCascades; }

    // Phase 4.4-4.6: shadow filter mode and per-mode settings
    void setShadowFilterMode(int32_t mode);    // 0=None, 1=PCF, 2=VSM
    void setPcfSpreadRadius(float radius);
    void setVsmBleedReduction(float reduction);
    int32_t getShadowFilterMode() const        { return m_shadowSettings.filterMode; }

    // Unload current model and load a new glTF at the given path.
    // Destroys model-dependent GPU resources, loads new model, recreates descriptors.
    // Must be called on the main thread while no frames are in flight.
    void reloadModel(const std::string& modelPath);

    // Phase 2.5: wire an ImGuiManager to receive an overlay render pass each frame
    void setImGuiManager(ImGuiManager* mgr) { m_imguiManager = mgr; }

    // Read-only access for UI panels
    const Model&     getModel()     const { return m_model; }
    const SceneInfo& getSceneInfo() const { return m_sceneInfo; }

    // Rendering toggles — stored here, applied when pipeline variants are added
    void setWireframeEnabled(bool enabled) { m_wireframe   = enabled; }
    void setNormalVisualization(bool enabled) { m_showNormals = enabled; }
    bool isWireframeEnabled()     const { return m_wireframe; }
    bool isNormalVisualization()  const { return m_showNormals; }

    // Phase 4.3: cascade shadow map constants
    static constexpr uint32_t CASCADE_COUNT = shader_interface::kCascadeCount;
    static constexpr uint32_t CASCADE_SIZE  = 1024;

    // Phase 4.2.5: shadow culling stats
    const std::array<uint32_t, CASCADE_COUNT>& getShadowCulledMeshes() const { return m_shadowPass.getDebugInfo().culledMeshes; }
    const std::array<uint32_t, CASCADE_COUNT>& getShadowTotalMeshes()  const { return m_shadowPass.getDebugInfo().totalMeshes; }
    const ShadowDebugInfo& getShadowDebugInfo() const { return m_shadowPass.getDebugInfo(); }

    // Phase 5/6: tone mapping controls
    void setExposure(float ev)           { m_exposure = ev; }
    float getExposure()   const          { return m_exposure; }
    int32_t getTonemapMode() const       { return m_tonemapMode; }

    // Phase 6: unified setter for all tonemap params (pushed as one PC struct each frame)
    void setTonemapParams(int32_t mode, float exposure, bool splitScreen, int32_t splitRightMode) {
        m_tonemapMode      = mode;
        m_exposure         = exposure;
        m_splitScreenMode  = splitScreen ? 1 : 0;
        m_splitRightMode   = splitRightMode;
    }

    // Phase 6.5: sky controls
    void setSkyEnabled(bool enabled)     { m_skyEnabled = enabled; }
    bool isSkyEnabled()     const        { return m_skyEnabled; }
    void setSkyMode(int32_t mode)        { m_skyMode = mode; }
    int32_t getSkyMode()    const        { return m_skyMode; }
    void loadHdrPanorama(const std::string& path);

    // Phase 2.6: render statistics — populated each frame in render()
    struct RenderStats {
        uint32_t drawCalls          = 0;
        uint32_t triangles          = 0;
        uint32_t meshCount          = 0;
        uint32_t materialCount      = 0;
        uint32_t textureCount       = 0;
        size_t   textureMemoryBytes = 0;   // approximate GPU texture memory usage
    };
    const RenderStats& getRenderStats() const { return m_renderStats; }

    // Phase 2.6: GPU timer — read timing results from the previous completed frame
    const GPUTimer& getGPUTimer() const { return m_gpuTimer; }

    // Phase 2.6: queue a screenshot capture at the end of the current frame
    void requestScreenshot(const std::string& filename = "");

private:
    void render();
    void handleResize();
    void createResizeDependentResources();
    void destroyResizeDependentResources();
    void recreateResizeDependentResources();
    void createPbrPipeline();
    void loadModel(const std::string& modelPath);
    void rebuildDrawCommands();
    void destroyPipeline();
    void refreshRenderStats();

    static std::vector<uint32_t> loadSpv(const std::string& path);

    VulkanContext& m_ctx;
    Swapchain&     m_swapchain;
    CommandBuffer  m_commandBuffer;
    FrameSync      m_frameSync;

    // Geometry buffers
    Buffer m_vertexBuffer;
    Buffer m_indexBuffer;

    // Phase 1.3: model and per-mesh draw data
    Model                       m_model;
    SceneInfo                   m_sceneInfo;    // Phase 3.7: normalization transform
    std::vector<MeshRenderData> m_meshes;
    std::vector<Material>       m_materials;
    std::vector<DrawCommand>    m_drawCommands;

    // Pipeline resources
    VkPipeline            m_pipeline           = VK_NULL_HANDLE;
    VkPipeline            m_wireframePipeline  = VK_NULL_HANDLE;
    VkPipeline            m_normalsPipeline    = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout     = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_cameraSetLayout    = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_materialSetLayout  = VK_NULL_HANDLE;
    VkShaderModule        m_vertModule         = VK_NULL_HANDLE;
    VkShaderModule        m_fragModule         = VK_NULL_HANDLE;
    VkShaderModule        m_normalsFragModule  = VK_NULL_HANDLE;

    // Material descriptor sets (one per Model::materials entry)
    std::vector<VkDescriptorSet> m_materialSets;

    // Push constant: bytes 64-95 in the pipeline layout (after 64-byte model matrix)
    using MaterialPushConstants = shader_interface::MaterialPushConstants;

    // Camera UBO (host-visible, updated every frame)
    // Phase 6.5: added inverseVP between projection and cameraPos for sky ray reconstruction.
    using CameraUBO = shader_interface::CameraUBO;

    // Directional light UBO (host-visible, std140: 48 bytes = 3×vec4)
    using LightUBO = shader_interface::LightUBO;
    using ShadowCascadeUBO = shader_interface::ShadowCascadeUBO;
    using ClusterMetadataUBO = shader_interface::ClusterMetadataUBO;

    Buffer           m_cameraUBOBuffer;
    Buffer           m_lightUBOBuffer;
    Buffer           m_pointLightBuffer;
    Buffer           m_clusterGridBuffer;
    Buffer           m_clusterLightIndexBuffer;
    Buffer           m_clusterMetadataBuffer;
    std::vector<shader_interface::GpuPointLight> m_pointLights;
    uint32_t         m_maxPointLights = 1024;
    uint32_t         m_clusterCountX = 0;
    uint32_t         m_clusterCountY = 0;
    uint32_t         m_clusterCountZ = shader_interface::kClusterZSlices;
    uint32_t         m_clusterCount = 0;

    // Texture infrastructure
    SamplerCache         m_samplerCache;
    std::vector<Texture> m_textures;
    Texture              m_fallbackWhite;

    // Resize-dependent depth buffer (D32_SFLOAT, reverse-Z)
    Image            m_depthImage;

    // Phase 5: resize-dependent HDR offscreen target (R16G16B16A16_SFLOAT, swapchain-sized)
    Image            m_hdrTarget;
    // Image-independent sampler reused when the HDR target is recreated.
    VkSampler        m_hdrSampler           = VK_NULL_HANDLE;

    TonemapPass      m_tonemapPass;

    // Phase 5/6: tone map state (uploaded as TonemapPC push constant each frame)
    float   m_exposure         = 0.0f;  // EV offset; 0 = no change
    int32_t m_tonemapMode      = 0;     // 0=Reinhard, 1=AgX, 2=PBR Neutral
    int32_t m_splitScreenMode  = 0;     // 0=off, 1=on
    int32_t m_splitRightMode   = 1;     // comparison operator for right half (default AgX)

    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet  m_descriptorSet  = VK_NULL_HANDLE;

    // Phase 2.5: optional ImGui overlay
    ImGuiManager*    m_imguiManager   = nullptr;

    // Phase 2.6: profiling and stats
    GPUTimer     m_gpuTimer;
    Screenshot   m_screenshot;
    RenderStats  m_renderStats;

    ShadowPass      m_shadowPass;
    ClusteredLightCullingPass m_clusteredLightCullingPass;
    ShadowSettings  m_shadowSettings;
    glm::vec3       m_lightDirection   = glm::vec3(1.0f, 1.0f, 1.0f);

    // Stored to re-upload LightUBO when debug toggle changes
    glm::vec3 m_lightColor        = glm::vec3(1.0f);
    float     m_lightIntensity    = 1.0f;
    float     m_ambientIntensity  = 0.1f;

    // Camera frustum — updated by setCameraFrustum() for CSM split computation
    glm::mat4 m_viewMatrix  = glm::mat4(1.0f);
    glm::mat4 m_projMatrix  = glm::mat4(1.0f);
    float     m_cameraNearZ = 0.01f;
    float     m_cameraFarZ  = 1000.0f;

    SkyPass m_skyPass;
    bool    m_skyEnabled = true;   // sky drawn by default
    int32_t m_skyMode    = 0;      // 0 = Procedural (Rayleigh+Mie),  1 = HDR Panorama

    bool        m_screenshotRequested = false;
    std::string m_screenshotFilename;

    // Rendering toggles (UI-driven; applied when pipeline variants are added)
    bool m_wireframe   = false;
    bool m_showNormals = false;

    glm::vec3 m_cameraPos = glm::vec3(0.0f);

    void createCameraUBO();
    void createLightUBO();
    void uploadLightUBO();   // re-upload m_lightUBOBuffer from stored params
    void createDefaultPointLights();
    void createClusteredLightResources();
    void destroyClusteredLightResources();
    void resizeClusteredLightResources();
    void uploadPointLightBuffer();
    void uploadClusterMetadata();
    void updateClusterDescriptors();
    void createDepthImage();
    void createHdrTarget();
    void createDescriptorPool();
    void createDescriptorSet();
    void updateShadowDescriptors();
    void createMaterialDescriptorSets();
};
