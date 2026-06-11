#pragma once

#include "core/ShaderInterface.h"
#include "core/VulkanContext.h"

#include <string>

class TonemapPass {
public:
    void init(VkDevice device, VkFormat outputFormat, const std::string& shaderDir);
    void resize(VkDevice device, VkSampler hdrSampler, VkImageView hdrImageView);
    void record(VkCommandBuffer cmd, VkImageView outputView, VkExtent2D extent,
                const shader_interface::TonemapPC& params) const;
    void shutdown(VkDevice device);

private:
    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkDescriptorPool      m_pool = VK_NULL_HANDLE;
    VkDescriptorSet       m_set = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline            m_pipeline = VK_NULL_HANDLE;
};
