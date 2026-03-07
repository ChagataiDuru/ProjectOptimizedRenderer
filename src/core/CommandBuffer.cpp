#include "core/CommandBuffer.h"

#include <spdlog/spdlog.h>
#include <stdexcept>

CommandBuffer::CommandBuffer(VulkanContext& ctx)
    : m_ctx(ctx)
{
}

CommandBuffer::~CommandBuffer()
{
    shutdown();
}

void CommandBuffer::init(uint32_t queueFamily, uint32_t framesInFlight)
{
    m_framesInFlight = framesInFlight;

    const VkCommandPoolCreateInfo poolInfo{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        // RESET_COMMAND_BUFFER_BIT allows individual buffer reset without resetting the whole pool
        .flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queueFamily,
    };
    VK_CHECK(vkCreateCommandPool(m_ctx.getDevice(), &poolInfo, nullptr, &m_pool));

    m_commandBuffers.resize(framesInFlight);
    const VkCommandBufferAllocateInfo allocInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = m_pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = framesInFlight,
    };
    VK_CHECK(vkAllocateCommandBuffers(m_ctx.getDevice(), &allocInfo, m_commandBuffers.data()));

    spdlog::info("Command pool created with {} buffers", framesInFlight);
}

void CommandBuffer::shutdown()
{
    if (m_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_ctx.getDevice(), m_pool, nullptr);
        m_pool = VK_NULL_HANDLE;
        m_commandBuffers.clear();
    }
}

void CommandBuffer::beginFrame()
{
    m_frameIndex = (m_frameIndex + 1) % m_framesInFlight;
}

void CommandBuffer::resetFrame()
{
    // RELEASE_RESOURCES_BIT returns memory to the pool; correct for per-frame reuse
    VK_CHECK(vkResetCommandBuffer(
        m_commandBuffers[m_frameIndex],
        VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT));
}

void CommandBuffer::submit(VkQueue queue, VkSemaphore waitSem, VkSemaphore signalSem, VkFence fence)
{
    // VkSubmitInfo2 is core in Vulkan 1.3 (synchronization2). Provides explicit pipeline stage
    // granularity on the semaphore wait, replacing the implicit VkPipelineStageFlags in SubmitInfo.
    const VkSemaphoreSubmitInfo waitInfo{
        .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = waitSem,
        // Wait before writing to the color attachment
        .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
    };
    const VkSemaphoreSubmitInfo signalInfo{
        .sType     = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
        .semaphore = signalSem,
        .stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
    };
    const VkCommandBufferSubmitInfo cmdInfo{
        .sType         = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
        .commandBuffer = m_commandBuffers[m_frameIndex],
    };
    const VkSubmitInfo2 submitInfo{
        .sType                    = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
        .waitSemaphoreInfoCount   = 1,
        .pWaitSemaphoreInfos      = &waitInfo,
        .commandBufferInfoCount   = 1,
        .pCommandBufferInfos      = &cmdInfo,
        .signalSemaphoreInfoCount = 1,
        .pSignalSemaphoreInfos    = &signalInfo,
    };
    VK_CHECK(vkQueueSubmit2(queue, 1, &submitInfo, fence));
}
