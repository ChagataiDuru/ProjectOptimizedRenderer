#include "resource/PipelineLayout.h"

#include <spdlog/spdlog.h>

PipelineLayout::PipelineLayout(VulkanContext& ctx)
    : m_ctx(ctx)
{
}

PipelineLayout::~PipelineLayout()
{
    if (m_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(m_ctx.getDevice(), m_layout, nullptr);
}

void PipelineLayout::addDescriptorSetLayout(const DescriptorSetLayout& dsl)
{
    m_setLayouts.push_back(dsl.getLayout());
}

void PipelineLayout::setPushConstants(VkShaderStageFlags stages, uint32_t size, uint32_t offset)
{
    m_pushConstantRanges.push_back({ stages, offset, size });
}

void PipelineLayout::finalize()
{
    const VkPipelineLayoutCreateInfo ci{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = static_cast<uint32_t>(m_setLayouts.size()),
        .pSetLayouts            = m_setLayouts.empty() ? nullptr : m_setLayouts.data(),
        .pushConstantRangeCount = static_cast<uint32_t>(m_pushConstantRanges.size()),
        .pPushConstantRanges    = m_pushConstantRanges.empty() ? nullptr : m_pushConstantRanges.data(),
    };
    VK_CHECK(vkCreatePipelineLayout(m_ctx.getDevice(), &ci, nullptr, &m_layout));
    spdlog::debug("PipelineLayout finalized (sets={}, pushConstants={}, pushDescriptors={})",
        m_setLayouts.size(), m_pushConstantRanges.size(), hasPushDescriptors());
}
