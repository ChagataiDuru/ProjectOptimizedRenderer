#pragma once

#include "core/VulkanContext.h"
#include "resource/Descriptor.h"
#include <vector>

class PipelineLayout {
public:
    explicit PipelineLayout(VulkanContext& ctx);
    ~PipelineLayout();

    PipelineLayout(const PipelineLayout&)            = delete;
    PipelineLayout& operator=(const PipelineLayout&) = delete;

    void addDescriptorSetLayout(const DescriptorSetLayout& dsl);

    // Registers a push constant range. offset defaults to 0 for the common single-range case.
    void setPushConstants(VkShaderStageFlags stages, uint32_t size, uint32_t offset = 0);

    // Compiles the layout. Must be called before getLayout().
    void finalize();

    VkPipelineLayout getLayout()              const { return m_layout; }
    bool             hasPushDescriptors()     const { return m_ctx.hasFeature_PushDescriptor(); }

private:
    VulkanContext&                      m_ctx;
    VkPipelineLayout                    m_layout = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayout>  m_setLayouts;
    std::vector<VkPushConstantRange>    m_pushConstantRanges;
};
