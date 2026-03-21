#pragma once

#include "core/VulkanContext.h"
#include "core/Swapchain.h"
#include "core/CommandBuffer.h"
#include "core/FrameSync.h"
#include "debug/GPUTimer.h"
#include "debug/Screenshot.h"
#include "resource/Buffer.h"
#include "resource/Image.h"
#include "resource/Model.h"
#include "resource/SceneInfo.h"
#include "resource/Texture.h"
#include "resource/SamplerCache.h"
#include <glm/glm.hpp>
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

    void setCameraMatrices(const glm::mat4& view, const glm::mat4& projection,
                           const glm::vec3& cameraPos);

    void setLightParameters(const glm::vec3& direction, const glm::vec3& color,
                            float intensity, float ambient);

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

    // Phase 4.1: shadow map resolution (power-of-two for clean texel mapping)
    static constexpr uint32_t SHADOW_MAP_SIZE = 2048;

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
    void createPbrPipeline();
    void loadModel(const std::string& modelPath);
    void destroyPipeline();

    static std::vector<uint32_t> loadSpv(const std::string& path);

    VulkanContext& m_ctx;
    Swapchain&     m_swapchain;
    CommandBuffer  m_commandBuffer;
    FrameSync      m_frameSync;

    // Geometry buffers
    Buffer m_vertexBuffer;
    Buffer m_indexBuffer;

    // Phase 1.3: model and per-mesh draw data
    struct MeshRenderData {
        uint32_t firstIndex;
        uint32_t indexCount;
        int32_t  materialIndex = -1;
    };
    Model                       m_model;
    SceneInfo                   m_sceneInfo;    // Phase 3.7: normalization transform
    std::vector<MeshRenderData> m_meshRenderData;

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
    struct MaterialPushConstants {
        glm::vec4 baseColorFactor = glm::vec4(1.0f);
        float     metallicFactor  = 1.0f;
        float     roughnessFactor = 1.0f;
        float     alphaCutoff     = 0.0f;   // 0 = no cutoff (opaque); >0 = discard below this
        float     _pad            = 0.0f;
    };

    // Camera UBO (host-visible, updated every frame)
    struct CameraUBO {
        glm::mat4 view;
        glm::mat4 projection;
        glm::vec3 cameraPos;
        float     _pad = 0.0f;  // std140: vec3 pads to 16 bytes
    };

    // Directional light UBO (host-visible)
    struct LightUBO {
        glm::vec3 lightDirection;
        float     lightIntensity;
        glm::vec3 lightColor;
        float     ambientIntensity;
    };

    Buffer           m_cameraUBOBuffer;
    Buffer           m_lightUBOBuffer;

    // Texture infrastructure
    SamplerCache         m_samplerCache;
    std::vector<Texture> m_textures;
    Texture              m_fallbackWhite;

    // Depth buffer (D32_SFLOAT, reverse-Z)
    Image            m_depthImage;

    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet  m_descriptorSet  = VK_NULL_HANDLE;

    // Phase 2.5: optional ImGui overlay
    ImGuiManager*    m_imguiManager   = nullptr;

    // Phase 2.6: profiling and stats
    GPUTimer     m_gpuTimer;
    Screenshot   m_screenshot;
    RenderStats  m_renderStats;

    // Phase 4.1: shadow map resources
    struct ShadowUBO {
        glm::mat4 lightViewProj;
    };
    Image            m_shadowMap;           // D32_SFLOAT depth image, SHADOW_MAP_SIZE²
    VkSampler        m_shadowSampler = VK_NULL_HANDLE;  // comparison sampler
    Buffer           m_shadowUBOBuffer;     // host-visible ShadowUBO
    VkPipeline       m_shadowPipeline   = VK_NULL_HANDLE;
    VkShaderModule   m_shadowVertModule = VK_NULL_HANDLE;
    glm::vec3        m_lightDirection   = glm::vec3(1.0f, 1.0f, 1.0f);

    bool        m_screenshotRequested = false;
    std::string m_screenshotFilename;

    // Rendering toggles (UI-driven; applied when pipeline variants are added)
    bool m_wireframe   = false;
    bool m_showNormals = false;

    glm::vec3 m_cameraPos = glm::vec3(0.0f);

    void createCameraUBO();
    void createLightUBO();
    void createDepthImage();
    void createDescriptorPool();
    void createDescriptorSet();
    void createMaterialDescriptorSets();
    void createShadowResources();
    void updateShadowMatrices();
};
