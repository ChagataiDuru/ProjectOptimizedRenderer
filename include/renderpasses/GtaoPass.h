#pragma once

#include "core/VulkanContext.h"
#include "resource/Image.h"

#include <glm/glm.hpp>

#include <string>
#include <vector>

class GPUTimer;

// Screen-space ambient occlusion from the depth prepass (ART-LGT-005).
// Runs at half resolution and writes a single-channel AO image that pbr.frag
// multiplies into the ambient/IBL term.
class GtaoPass {
public:
    explicit GtaoPass(VulkanContext& ctx);
    ~GtaoPass();

    GtaoPass(const GtaoPass&) = delete;
    GtaoPass& operator=(const GtaoPass&) = delete;

    void init(const std::string& shaderDir);
    void shutdown(VkDevice device);

    // (Re)creates the half-resolution AO targets for a new scene depth view.
    void resize(uint32_t fullWidth, uint32_t fullHeight, VkImageView sceneDepthView,
                VkSampler sceneDepthSampler);

    // Records AO + blur. sceneDepth must be in SHADER_READ_ONLY_OPTIMAL.
    void record(VkCommandBuffer cmd, const glm::mat4& inverseProjection,
                float worldRadius, float intensity, GPUTimer* timer);

    VkImageView aoView() const { return m_aoBlurred.getImageView(); }
    VkSampler aoSampler() const { return m_aoSampler; }
    bool isReady() const { return m_aoBlurred.getImageView() != VK_NULL_HANDLE; }

private:
    void destroyTargets();
    void writeSets(VkImageView sceneDepthView, VkSampler sceneDepthSampler);
    static std::vector<uint32_t> loadSpv(const std::string& path);

    VulkanContext& m_ctx;
    std::string m_shaderDir;

    Image m_aoRaw;
    Image m_aoBlurred;
    Image m_aoTemp;
    VkSampler m_aoSampler = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_gtaoSet = VK_NULL_HANDLE;
    VkDescriptorSet m_blurHorizontalSet = VK_NULL_HANDLE;
    VkDescriptorSet m_blurVerticalSet = VK_NULL_HANDLE;

    VkPipelineLayout m_gtaoLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_blurLayout = VK_NULL_HANDLE;
    VkPipeline m_gtaoPipeline = VK_NULL_HANDLE;
    VkPipeline m_blurPipeline = VK_NULL_HANDLE;

    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_fullWidth = 0;
    uint32_t m_fullHeight = 0;
};
