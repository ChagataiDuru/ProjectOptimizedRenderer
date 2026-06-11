#pragma once

#include "core/VulkanContext.h"

#include <string>
#include <vector>

class ClusteredLightCullingPass {
public:
    explicit ClusteredLightCullingPass(VulkanContext& ctx);
    ~ClusteredLightCullingPass();

    ClusteredLightCullingPass(const ClusteredLightCullingPass&) = delete;
    ClusteredLightCullingPass& operator=(const ClusteredLightCullingPass&) = delete;

    void init(VkDescriptorSetLayout sceneSetLayout, const std::string& shaderDir);
    void record(VkCommandBuffer cmd,
                VkDescriptorSet sceneDescriptorSet,
                uint32_t clusterCountX,
                uint32_t clusterCountY,
                uint32_t clusterCountZ) const;
    void shutdown(VkDevice device);

private:
    static std::vector<uint32_t> loadSpv(const std::string& path);

    VulkanContext& m_ctx;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
};
