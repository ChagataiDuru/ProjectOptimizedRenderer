#pragma once

#include "core/DrawCommand.h"
#include "core/MeshRenderData.h"
#include "core/RenderSettings.h"
#include "core/ShaderInterface.h"
#include "core/VulkanContext.h"
#include "debug/GPUTimer.h"
#include "resource/Buffer.h"
#include "resource/Image.h"
#include "resource/Material.h"

#include <array>
#include <string>
#include <vector>

class ShadowPass {
public:
    static constexpr uint32_t CASCADE_COUNT = shader_interface::kCascadeCount;

    struct CullingPlane {
        glm::vec3 normal = glm::vec3(0.0f);
        float d = 0.0f;
    };

    explicit ShadowPass(VulkanContext& ctx);
    ~ShadowPass();

    ShadowPass(const ShadowPass&) = delete;
    ShadowPass& operator=(const ShadowPass&) = delete;

    void init(VkPipelineLayout sharedPipelineLayout,
              uint32_t framesInFlight,
              const std::string& shaderDir);
    void shutdown(VkDevice device);

    // Returns true when image/sampler descriptors should be rewritten.
    bool updateSettings(const ShadowSettings& settings);
    void updateLightDirection(glm::vec3 lightDirection);
    void updateCamera(const glm::mat4& view,
                      const glm::mat4& projection,
                      float nearZ,
                      float farZ,
                      const glm::vec3& cameraPos);

    void record(VkCommandBuffer cmd,
                VkDescriptorSet sceneDescriptorSet,
                uint32_t frameIndex,
                const Buffer& vertexBuffer,
                const Buffer& indexBuffer,
                const std::vector<DrawCommand>& drawCommands,
                const std::vector<MeshRenderData>& meshes,
                const std::vector<Material>& materials,
                const std::vector<VkDescriptorSet>& materialSets,
                GPUTimer& gpuTimer);

    const shader_interface::ShadowCascadeUBO& getLastShadowUBO() const { return m_lastShadowUBO; }
    const ShadowDebugInfo& getDebugInfo() const { return m_debugInfo; }

    VkBuffer getShadowUBOBuffer(uint32_t frameIndex) const;
    VkSampler getDepthSampler() const { return m_shadowSampler; }
    VkImageView getDepthImageView() const { return m_shadowMap.getImageView(); }
    VkImageLayout getDepthSampleLayout() const { return VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL; }
    VkSampler getMomentsSampler() const { return m_momentsSampler; }
    VkImageView getMomentsImageView() const;
    VkImageLayout getMomentsSampleLayout() const { return VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; }
    uint32_t getCascadeResolution() const { return m_cascadeSize; }
    bool consumeDescriptorDirty();

private:
    static std::vector<uint32_t> loadSpv(const std::string& path);
    static glm::vec3 safeNormalizeDirection(glm::vec3 v, glm::vec3 fallback);

    void createDepthResources();
    void createFallbackMoments();
    void createPipelines();
    void createVsmResources();
    void destroyVsmResources(VkDevice device);
    void destroyDepthResources(VkDevice device);
    void recreateForResolution(uint32_t newSize);
    void updateDebugMemoryEstimate();
    void updateMatrices(uint32_t frameIndex);

    struct FrameResources {
        explicit FrameResources(VulkanContext& ctx)
            : shadowUBO(ctx)
        {
        }

        FrameResources(FrameResources&&) noexcept = default;
        FrameResources& operator=(FrameResources&&) noexcept = default;
        FrameResources(const FrameResources&) = delete;
        FrameResources& operator=(const FrameResources&) = delete;

        Buffer shadowUBO;
    };

    VulkanContext& m_ctx;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    std::string m_shaderDir;

    Image m_shadowMap;
    std::array<VkImageView, CASCADE_COUNT> m_shadowLayerViews = {};
    VkSampler m_shadowSampler = VK_NULL_HANDLE;
    std::vector<FrameResources> m_frameResources;

    Image m_fallbackMoments;
    VkSampler m_momentsSampler = VK_NULL_HANDLE;
    Image m_shadowMoments;
    Image m_shadowMomentsTemp;
    std::array<VkImageView, CASCADE_COUNT> m_momentsLayerViews = {};
    VkPipeline m_shadowPipeline = VK_NULL_HANDLE;
    VkPipeline m_shadowAlphaPipeline = VK_NULL_HANDLE;
    VkPipeline m_shadowVsmPipeline = VK_NULL_HANDLE;
    VkPipeline m_shadowVsmAlphaPipeline = VK_NULL_HANDLE;

    VkPipeline m_blurPipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_blurPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_blurSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_blurDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet m_blurSetHorizontal = VK_NULL_HANDLE;
    VkDescriptorSet m_blurSetVertical = VK_NULL_HANDLE;

    ShadowSettings m_settings;
    ShadowDebugInfo m_debugInfo;
    shader_interface::ShadowCascadeUBO m_lastShadowUBO{};
    std::array<std::vector<CullingPlane>, CASCADE_COUNT> m_shadowCullPlanes;

    glm::vec3 m_lightDirection = glm::normalize(glm::vec3(0.577f));
    glm::mat4 m_viewMatrix = glm::mat4(1.0f);
    glm::mat4 m_projMatrix = glm::mat4(1.0f);
    glm::vec3 m_cameraPos = glm::vec3(0.0f);
    float m_cameraNearZ = 0.01f;
    float m_cameraFarZ = 1000.0f;
    uint32_t m_cascadeSize = 1024;
    bool m_vsmAllocated = false;
    bool m_descriptorDirty = false;
};
