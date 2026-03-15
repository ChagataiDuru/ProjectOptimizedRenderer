#pragma once

#include "core/VulkanContext.h"
#include "core/Swapchain.h"
#include "core/CommandBuffer.h"
#include "core/FrameSync.h"
#include "resource/Buffer.h"
#include "resource/Image.h"
#include "resource/Model.h"
#include "resource/Texture.h"
#include "resource/SamplerCache.h"
#include <glm/glm.hpp>
#include <vector>
#include <string>

class Renderer {
public:
    Renderer(VulkanContext& ctx, Swapchain& swapchain);
    ~Renderer();

    Renderer(const Renderer&)            = delete;
    Renderer& operator=(const Renderer&) = delete;

    void init();
    void shutdown();

    // Wait for frame slot, reset command buffer, acquire swapchain image.
    void beginFrame();

    // Record commands, submit, present, advance frame indices.
    void endFrame();

    // Placeholder: Phase 1.2 will wire this to a per-frame UBO upload.
    void setCameraMatrices(const glm::mat4& view, const glm::mat4& projection);

    // Phase 1.4: update the directional light parameters at any time
    void setLightParameters(const glm::vec3& direction, const glm::vec3& color,
                            float intensity, float ambient);

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

    // Phase 1.1/1.3: geometry buffers (initialized in constructor list — Buffer has no default ctor)
    Buffer m_vertexBuffer;
    Buffer m_indexBuffer;

    // Phase 1.3: model and per-mesh draw data
    struct MeshRenderData {
        uint32_t firstIndex;              // starting element in the index buffer
        uint32_t indexCount;              // number of indices for this mesh
        int32_t  materialIndex = -1;      // index into Model::materials (Phase 2.2)
    };
    Model                     m_model;
    std::vector<MeshRenderData> m_meshRenderData;

    // Phase 1.1: pipeline resources
    VkPipeline            m_pipeline           = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout     = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_cameraSetLayout    = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_materialSetLayout  = VK_NULL_HANDLE;  // Phase 2.3: set=1 textures
    // Shader modules are destroyed after pipeline creation; kept as VK_NULL_HANDLE thereafter.
    VkShaderModule        m_vertModule         = VK_NULL_HANDLE;
    VkShaderModule        m_fragModule         = VK_NULL_HANDLE;

    // Phase 2.3: material descriptor sets (one per Model::materials entry)
    std::vector<VkDescriptorSet> m_materialSets;

    // Push constant data for material factors sent to the fragment shader.
    // Lives at offset 64 (after the 64-byte model matrix at offset 0).
    struct MaterialPushConstants {
        glm::vec4 baseColorFactor = glm::vec4(1.0f);
        float     metallicFactor  = 1.0f;
        float     roughnessFactor = 1.0f;
        float     _pad[2]         = {};  // pad to 32 bytes
    };

    // Phase 1.2: camera UBO — host-visible so we can memcpy each frame without staging
    struct CameraUBO {
        glm::mat4 view;
        glm::mat4 projection;
    };

    // Phase 1.4: directional light UBO — host-visible, updated via setLightParameters()
    struct LightUBO {
        glm::vec3 lightDirection;   // world-space direction (normalized, pointing toward scene)
        float     lightIntensity;   // brightness multiplier
        glm::vec3 lightColor;       // RGB color
        float     ambientIntensity; // ambient light level
    };

    Buffer           m_cameraUBOBuffer;
    Buffer           m_lightUBOBuffer;

    // Phase 2.1/2.2: texture infrastructure
    SamplerCache         m_samplerCache;
    std::vector<Texture> m_textures;     // Indexed parallel to Model::textures
    Texture              m_fallbackWhite; // 1x1 white texture for missing slots

    // Phase 1.5: depth buffer — D32_SFLOAT, reverse-Z (clear=0.0, compare=GREATER)
    Image            m_depthImage;

    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet  m_descriptorSet  = VK_NULL_HANDLE;

    void createCameraUBO();
    void createLightUBO();
    void createDepthImage();
    void createDescriptorPool();
    void createDescriptorSet();
    void createMaterialDescriptorSets();  // Phase 2.3
};
