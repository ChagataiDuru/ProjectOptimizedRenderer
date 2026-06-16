#pragma once

#include "core/VulkanContext.h"
#include "resource/Image.h"

#include <string>

class SkyPass {
public:
    explicit SkyPass(VulkanContext& ctx);

    void init(VulkanContext& ctx,
              VkDescriptorSetLayout sceneSetLayout,
              VkFormat colorFormat,
              VkFormat depthFormat,
              VkSampleCountFlagBits sceneSamples,
              const std::string& shaderDir);

    void recreatePipeline(VulkanContext& ctx,
                          VkSampleCountFlagBits sceneSamples);

    void record(VkCommandBuffer cmd,
                VkDescriptorSet sceneSet,
                VkExtent2D extent,
                bool enabled,
                int32_t mode) const;

    void loadHdrPanorama(VulkanContext& ctx,
                         const std::string& path);

    void shutdown(VkDevice device);

private:
    Image                 m_panorama;
    VkSampler             m_panoramaSampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_panoramaSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool      m_panoramaPool = VK_NULL_HANDLE;
    VkDescriptorSet       m_panoramaSet = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline            m_pipeline = VK_NULL_HANDLE;
    VkFormat              m_colorFormat = VK_FORMAT_UNDEFINED;
    VkFormat              m_depthFormat = VK_FORMAT_UNDEFINED;
    VkSampleCountFlagBits m_sceneSamples = VK_SAMPLE_COUNT_1_BIT;
    std::string           m_shaderDir;
};
