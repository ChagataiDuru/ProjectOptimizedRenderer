#include "core/Swapchain.h"

#include <SDL3/SDL_vulkan.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <stdexcept>
#include <string>

Swapchain::Swapchain(VulkanContext& ctx)
    : m_ctx(ctx)
{
}

Swapchain::~Swapchain()
{
    shutdown();
}

// ── Public interface ──────────────────────────────────────────────────────────

void Swapchain::init(void* platformWindow, uint32_t width, uint32_t height)
{
    m_requestedWidth  = width;
    m_requestedHeight = height;

    createSurface(platformWindow);
    selectFormat();
    createSwapchain();
    createImageViews();

    spdlog::info("Swapchain created: {}x{}, {} images, format {}",
        m_extent.width, m_extent.height,
        m_images.size(),
        static_cast<int>(m_imageFormat));
}

void Swapchain::recreate(uint32_t newWidth, uint32_t newHeight)
{
    vkDeviceWaitIdle(m_ctx.getDevice());

    destroyImageViews();

    VkSwapchainKHR old = m_swapchain;
    m_swapchain = VK_NULL_HANDLE;

    m_requestedWidth  = newWidth;
    m_requestedHeight = newHeight;

    createSwapchain();
    // Old swapchain destroyed inside createSwapchain() via oldSwapchain field
    if (old != VK_NULL_HANDLE)
        vkDestroySwapchainKHR(m_ctx.getDevice(), old, nullptr);

    createImageViews();
    spdlog::info("Swapchain recreated: {}x{}", m_extent.width, m_extent.height);
}

void Swapchain::shutdown()
{
    destroyImageViews();

    if (m_swapchain != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(m_ctx.getDevice(), m_swapchain, nullptr);
        m_swapchain = VK_NULL_HANDLE;
    }
    if (m_surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(m_ctx.getInstance(), m_surface, nullptr);
        m_surface = VK_NULL_HANDLE;
    }
}

bool Swapchain::acquireNextImage(VkSemaphore imageAvailable)
{
    const VkResult result = vkAcquireNextImageKHR(
        m_ctx.getDevice(),
        m_swapchain,
        UINT64_MAX,
        imageAvailable,
        VK_NULL_HANDLE,
        &m_currentImageIndex
    );

    if (result == VK_SUCCESS)
        return true;

    if (result == VK_SUBOPTIMAL_KHR) {
        spdlog::warn("Swapchain suboptimal — recreate on next frame boundary");
        return true;
    }

    if (result == VK_ERROR_OUT_OF_DATE_KHR)
        return false;

    throw std::runtime_error("vkAcquireNextImageKHR failed: " +
                             std::to_string(static_cast<int>(result)));
}

// ── Private helpers ───────────────────────────────────────────────────────────

void Swapchain::createSurface(void* platformWindow)
{
    auto* sdlWindow = static_cast<SDL_Window*>(platformWindow);

    // SDL3 dispatches to the correct platform surface creation internally:
    //   macOS/MoltenVK → vkCreateMetalSurfaceEXT
    //   Windows        → vkCreateWin32SurfaceKHR
    //   Linux          → vkCreateXcbSurfaceKHR / vkCreateWaylandSurfaceKHR
    if (!SDL_Vulkan_CreateSurface(sdlWindow, m_ctx.getInstance(), nullptr, &m_surface))
        throw std::runtime_error(std::string("SDL_Vulkan_CreateSurface failed: ") + SDL_GetError());
}

void Swapchain::selectFormat()
{
    uint32_t count = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_ctx.getPhysicalDevice(), m_surface, &count, nullptr);
    if (count == 0)
        throw std::runtime_error("No surface formats available");

    std::vector<VkSurfaceFormatKHR> formats(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(m_ctx.getPhysicalDevice(), m_surface, &count, formats.data());

    // BGRA8_SRGB is the native format on most platforms (Metal on macOS, DXGI on Windows).
    // RGBA8_SRGB is the fallback for any system that doesn't expose BGRA.
    const VkFormat preferred[]  = { VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_SRGB };
    const VkColorSpaceKHR space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

    for (VkFormat fmt : preferred) {
        auto it = std::find_if(formats.begin(), formats.end(),
            [&](const VkSurfaceFormatKHR& f) {
                return f.format == fmt && f.colorSpace == space;
            });
        if (it != formats.end()) {
            m_imageFormat = it->format;
            m_colorSpace  = it->colorSpace;
            spdlog::info("Surface format selected: {} (SRGB_NONLINEAR)",
                fmt == VK_FORMAT_B8G8R8A8_SRGB ? "B8G8R8A8_SRGB" : "R8G8B8A8_SRGB");
            return;
        }
    }

    // Last resort: take whatever the driver offers
    m_imageFormat = formats[0].format;
    m_colorSpace  = formats[0].colorSpace;
    spdlog::warn("Preferred SRGB formats unavailable — falling back to format {}",
        static_cast<int>(m_imageFormat));
}

void Swapchain::createSwapchain()
{
    VkSurfaceCapabilitiesKHR cap{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_ctx.getPhysicalDevice(), m_surface, &cap);

    // When currentExtent is UINT32_MAX, the surface allows us to choose our own extent.
    // Otherwise, use the exact extent reported (MoltenVK: Metal drawable size; Win32: window client area).
    if (cap.currentExtent.width != UINT32_MAX) {
        m_extent = cap.currentExtent;
    } else {
        m_extent.width  = std::clamp(m_requestedWidth,  cap.minImageExtent.width,  cap.maxImageExtent.width);
        m_extent.height = std::clamp(m_requestedHeight, cap.minImageExtent.height, cap.maxImageExtent.height);
    }

    // Request one extra image for pipelining (e.g. triple-buffer when min=2)
    uint32_t imageCount = cap.minImageCount + 1;
    if (cap.maxImageCount > 0)
        imageCount = std::min(imageCount, cap.maxImageCount);

    const VkSwapchainCreateInfoKHR createInfo{
        .sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface          = m_surface,
        .minImageCount    = imageCount,
        .imageFormat      = m_imageFormat,
        .imageColorSpace  = m_colorSpace,
        .imageExtent      = m_extent,
        .imageArrayLayers = 1,
        .imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        // EXCLUSIVE avoids unnecessary sync overhead when a single queue family handles
        // both graphics and presentation (true for both Apple Silicon UMA and most
        // NVIDIA configurations where graphics queue also supports present).
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .preTransform     = cap.currentTransform,
        // Opaque compositing — no window transparency
        .compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        // FIFO (vsync) is guaranteed to be supported; avoids tearing without frame timing.
        // On NVIDIA, consider VK_PRESENT_MODE_MAILBOX_KHR for uncapped fps (future option).
        .presentMode      = VK_PRESENT_MODE_FIFO_KHR,
        .clipped          = VK_TRUE,
    };

    VK_CHECK(vkCreateSwapchainKHR(m_ctx.getDevice(), &createInfo, nullptr, &m_swapchain));

    uint32_t actualCount = 0;
    vkGetSwapchainImagesKHR(m_ctx.getDevice(), m_swapchain, &actualCount, nullptr);
    m_images.resize(actualCount);
    vkGetSwapchainImagesKHR(m_ctx.getDevice(), m_swapchain, &actualCount, m_images.data());
}

void Swapchain::createImageViews()
{
    m_imageViews.resize(m_images.size());
    for (size_t i = 0; i < m_images.size(); ++i) {
        const VkImageViewCreateInfo viewInfo{
            .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image    = m_images[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format   = m_imageFormat,
            .components = {
                .r = VK_COMPONENT_SWIZZLE_IDENTITY,
                .g = VK_COMPONENT_SWIZZLE_IDENTITY,
                .b = VK_COMPONENT_SWIZZLE_IDENTITY,
                .a = VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            .subresourceRange = {
                .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel   = 0,
                .levelCount     = 1,
                .baseArrayLayer = 0,
                .layerCount     = 1,
            },
        };
        VK_CHECK(vkCreateImageView(m_ctx.getDevice(), &viewInfo, nullptr, &m_imageViews[i]));
    }
}

void Swapchain::destroyImageViews()
{
    for (auto view : m_imageViews)
        if (view != VK_NULL_HANDLE)
            vkDestroyImageView(m_ctx.getDevice(), view, nullptr);
    m_imageViews.clear();
}
