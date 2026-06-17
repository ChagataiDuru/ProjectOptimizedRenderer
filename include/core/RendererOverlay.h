#pragma once

#include "core/VulkanContext.h"
#include "core/Swapchain.h"

// Internal viewer/debug extension point. This is intentionally not part of the
// future public C ABI; it lets the C++ viewer record overlays without making
// Renderer depend on ImGui types.
struct RendererOverlayInitInfo {
    VulkanContext* context = nullptr;
    Swapchain*     swapchain = nullptr;
    void*          platformWindow = nullptr;
};

struct RendererOverlayRenderInfo {
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkImageView     swapchainView = VK_NULL_HANDLE;
    VkExtent2D      extent = {};
};

class RendererOverlay {
public:
    virtual ~RendererOverlay() = default;

    virtual void init(const RendererOverlayInitInfo& info) = 0;
    virtual void shutdown() = 0;
    virtual void record(const RendererOverlayRenderInfo& info) = 0;
};
