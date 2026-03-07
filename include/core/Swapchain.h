#pragma once

#include "core/VulkanContext.h"
#include <vector>

class Swapchain {
public:
    explicit Swapchain(VulkanContext& ctx);
    ~Swapchain();

    Swapchain(const Swapchain&)            = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    void init(void* platformWindow, uint32_t width, uint32_t height);
    void recreate(uint32_t newWidth, uint32_t newHeight);
    void shutdown();

    VkSwapchainKHR getSwapchain()        const { return m_swapchain; }
    VkImage        getCurrentImage()     const { return m_images[m_currentImageIndex]; }
    VkImageView    getCurrentImageView() const { return m_imageViews[m_currentImageIndex]; }
    VkFormat       getFormat()           const { return m_imageFormat; }
    VkExtent2D     getExtent()           const { return m_extent; }
    uint32_t       getImageIndex()       const { return m_currentImageIndex; }
    uint32_t       getImageCount()       const { return static_cast<uint32_t>(m_images.size()); }

    // Returns false when the swapchain is out of date — caller must recreate().
    bool acquireNextImage(VkSemaphore imageAvailable);

private:
    void createSurface(void* platformWindow);
    void selectFormat();
    void createSwapchain();
    void createImageViews();
    void destroyImageViews();

    VulkanContext& m_ctx;

    VkSurfaceKHR             m_surface           = VK_NULL_HANDLE;
    VkSwapchainKHR           m_swapchain         = VK_NULL_HANDLE;
    VkFormat                 m_imageFormat       = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR          m_colorSpace        = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkExtent2D               m_extent            = {};
    std::vector<VkImage>     m_images;
    std::vector<VkImageView> m_imageViews;
    uint32_t                 m_currentImageIndex = 0;
    uint32_t                 m_requestedWidth    = 0;
    uint32_t                 m_requestedHeight   = 0;
};
