#pragma once

#include "core/VulkanContext.h"
#include <string>

class Swapchain;

// Captures the rendered swapchain image to a PNG file on disk.
// Each capture creates and destroys its own staging buffer and transient
// command pool — intentionally heavyweight since this is a manual trigger (F2).
class Screenshot {
public:
    explicit Screenshot(VulkanContext& ctx);
    ~Screenshot() = default;

    Screenshot(const Screenshot&)            = delete;
    Screenshot& operator=(const Screenshot&) = delete;

    // Capture the image at the current swapchain index to a PNG file.
    // Call after the frame fence has signaled (e.g. after vkDeviceWaitIdle).
    // The swapchain image must be in PRESENT_SRC_KHR layout at the time of the call.
    //
    // If filename is empty, a timestamped name in screenshots/ is used automatically.
    void capture(const Swapchain& swapchain, const std::string& filename = "");

private:
    VulkanContext& m_ctx;
};
