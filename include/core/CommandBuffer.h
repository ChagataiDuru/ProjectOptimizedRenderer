#pragma once

#include "core/VulkanContext.h"
#include <vector>

class CommandBuffer {
public:
    explicit CommandBuffer(VulkanContext& ctx);
    ~CommandBuffer();

    CommandBuffer(const CommandBuffer&)            = delete;
    CommandBuffer& operator=(const CommandBuffer&) = delete;

    void init(uint32_t queueFamily, uint32_t framesInFlight);
    void shutdown();

    VkCommandBuffer getFrameCommandBuffer() const { return m_commandBuffers[m_frameIndex]; }

    // Advances frame slot — call at end of endFrame alongside FrameSync::advanceFrame().
    void beginFrame();

    // Resets the current frame's command buffer — call at start of beginFrame.
    void resetFrame();

    void submit(VkQueue queue, VkSemaphore waitSem, VkSemaphore signalSem, VkFence fence);

private:
    VulkanContext&               m_ctx;
    VkCommandPool                m_pool = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> m_commandBuffers;
    uint32_t                     m_frameIndex     = 0;
    uint32_t                     m_framesInFlight = 3;
};
