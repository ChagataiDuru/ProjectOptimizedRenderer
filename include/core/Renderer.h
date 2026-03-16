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
    void beginFrame();

    // Record commands, submit, present, advance frame indices.
    void endFrame();

    void setCameraMatrices(const glm::mat4& view, const glm::mat4& projection,
                           const glm::vec3& cameraPos);

    void setLightParameters(const glm::vec3& direction, const glm::vec3& color,
                            float intensity, float ambient);

    // Phase 2.5: wire an ImGuiManager to receive an overlay render pass each frame
    void setImGuiManager(ImGuiManager* mgr) { m_imguiManager = mgr; }

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
    std::vector<MeshRenderData> m_meshRenderData;

    // Pipeline resources
    VkPipeline            m_pipeline           = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout     = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_cameraSetLayout    = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_materialSetLayout  = VK_NULL_HANDLE;
    VkShaderModule        m_vertModule         = VK_NULL_HANDLE;
    VkShaderModule        m_fragModule         = VK_NULL_HANDLE;

    // Material descriptor sets (one per Model::materials entry)
    std::vector<VkDescriptorSet> m_materialSets;

    // Push constant: bytes 64-95 in the pipeline layout (after 64-byte model matrix)
    struct MaterialPushConstants {
        glm::vec4 baseColorFactor = glm::vec4(1.0f);
        float     metallicFactor  = 1.0f;
        float     roughnessFactor = 1.0f;
        float     _pad[2]         = {};
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

    bool        m_screenshotRequested = false;
    std::string m_screenshotFilename;

    void createCameraUBO();
    void createLightUBO();
    void createDepthImage();
    void createDescriptorPool();
    void createDescriptorSet();
    void createMaterialDescriptorSets();
};
