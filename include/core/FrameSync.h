#pragma once

#include "core/VulkanContext.h"
#include <vector>

class FrameSync {
public:
    explicit FrameSync(VulkanContext& ctx);
    ~FrameSync();

    FrameSync(const FrameSync&)            = delete;
    FrameSync& operator=(const FrameSync&) = delete;

    void init(uint32_t framesInFlight = 3);
    void shutdown();

    void waitForFrame();
    void resetFence();
    void advanceFrame();

    VkSemaphore getImageAvailableSemaphore() const { return m_imageAvailable[m_currentFrame]; }
    VkSemaphore getRenderFinishedSemaphore() const { return m_renderFinished[m_currentFrame]; }
    VkFence     getInFlightFence()           const { return m_inFlight[m_currentFrame]; }
    uint32_t    getCurrentFrame()            const { return m_currentFrame; }

private:
    VulkanContext&           m_ctx;
    std::vector<VkSemaphore> m_imageAvailable;
    std::vector<VkSemaphore> m_renderFinished;
    std::vector<VkFence>     m_inFlight;
    uint32_t                 m_currentFrame   = 0;
    uint32_t                 m_framesInFlight = 3;
};
