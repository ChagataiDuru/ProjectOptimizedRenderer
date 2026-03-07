#include "core/FrameSync.h"

#include <spdlog/spdlog.h>

FrameSync::FrameSync(VulkanContext& ctx)
    : m_ctx(ctx)
{
}

FrameSync::~FrameSync()
{
    shutdown();
}

void FrameSync::init(uint32_t framesInFlight)
{
    m_framesInFlight = framesInFlight;
    m_imageAvailable.resize(framesInFlight, VK_NULL_HANDLE);
    m_renderFinished.resize(framesInFlight, VK_NULL_HANDLE);
    m_inFlight.resize(framesInFlight, VK_NULL_HANDLE);

    const VkSemaphoreCreateInfo semaphoreInfo{
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    // Fences start signaled so the first frame doesn't block on an unsignaled fence.
    const VkFenceCreateInfo fenceInfo{
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };

    for (uint32_t i = 0; i < framesInFlight; ++i) {
        VK_CHECK(vkCreateSemaphore(m_ctx.getDevice(), &semaphoreInfo, nullptr, &m_imageAvailable[i]));
        VK_CHECK(vkCreateSemaphore(m_ctx.getDevice(), &semaphoreInfo, nullptr, &m_renderFinished[i]));
        VK_CHECK(vkCreateFence(m_ctx.getDevice(), &fenceInfo, nullptr, &m_inFlight[i]));
    }

    spdlog::info("Frame synchronization initialized ({} frames in flight)", framesInFlight);
}

void FrameSync::shutdown()
{
    const VkDevice dev = m_ctx.getDevice();
    if (dev == VK_NULL_HANDLE) return;

    for (uint32_t i = 0; i < m_framesInFlight; ++i) {
        if (m_imageAvailable[i] != VK_NULL_HANDLE)
            vkDestroySemaphore(dev, m_imageAvailable[i], nullptr);
        if (m_renderFinished[i] != VK_NULL_HANDLE)
            vkDestroySemaphore(dev, m_renderFinished[i], nullptr);
        if (m_inFlight[i] != VK_NULL_HANDLE)
            vkDestroyFence(dev, m_inFlight[i], nullptr);
    }
    m_imageAvailable.clear();
    m_renderFinished.clear();
    m_inFlight.clear();
}

void FrameSync::waitForFrame()
{
    VK_CHECK(vkWaitForFences(m_ctx.getDevice(), 1, &m_inFlight[m_currentFrame], VK_TRUE, UINT64_MAX));
}

void FrameSync::resetFence()
{
    VK_CHECK(vkResetFences(m_ctx.getDevice(), 1, &m_inFlight[m_currentFrame]));
}

void FrameSync::advanceFrame()
{
    m_currentFrame = (m_currentFrame + 1) % m_framesInFlight;
}
