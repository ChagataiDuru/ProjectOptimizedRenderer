#pragma once

#include "core/VulkanContext.h"

class Image {
public:
    explicit Image(VulkanContext& ctx);
    ~Image();

    Image(Image&&) noexcept;
    Image& operator=(Image&&) noexcept;
    Image(const Image&)            = delete;
    Image& operator=(const Image&) = delete;

    // Create a device-local image.
    // When arrayLayers > 1 (and depth == 1), the main image view is VK_IMAGE_VIEW_TYPE_2D_ARRAY
    // covering all layers. Use createSingleLayerView() for per-layer attachment views.
    void create(uint32_t width, uint32_t height, uint32_t depth,
                VkFormat format, VkImageUsageFlags usage,
                VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT,
                uint32_t arrayLayers = 1);

    // Upload pixel data via an internally-created staging buffer recorded into transferCmd.
    // Caller must submit transferCmd and wait for completion before using the image.
    // Final layout after upload: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL.
    void createFromData(uint32_t width, uint32_t height, VkFormat format,
                        const void* data, VkDeviceSize dataSize,
                        VkCommandBuffer transferCmd);

    void transitionLayout(VkCommandBuffer cmd,
                          VkImageLayout oldLayout, VkImageLayout newLayout);

    // Create a VkImageView targeting a single array layer (VK_IMAGE_VIEW_TYPE_2D).
    // Caller owns the returned handle and must destroy it before this Image is destroyed.
    VkImageView createSingleLayerView(uint32_t layer,
                                      VkImageAspectFlags aspectFlags) const;

    // Release staging buffer after GPU transfer is complete.
    void releaseStaging();

    void destroy();

    VkImage     getImage()       const { return m_image; }
    VkImageView getImageView()   const { return m_imageView; }
    VkFormat    getFormat()      const { return m_format; }
    VkExtent3D  getExtent()      const { return m_extent; }
    uint32_t    getArrayLayers() const { return m_arrayLayers; }

private:
    void createView(VkImageAspectFlags aspectFlags);

    VulkanContext& m_ctx;
    VkImage        m_image       = VK_NULL_HANDLE;
    VkImageView    m_imageView   = VK_NULL_HANDLE;
    VmaAllocation  m_allocation  = nullptr;
    VkFormat       m_format      = VK_FORMAT_UNDEFINED;
    VkExtent3D     m_extent      = {};
    uint32_t       m_arrayLayers = 1;  // >1 → 2D array image

    // Staging resources for createFromData — released after GPU completion
    VkBuffer      m_stagingBuffer = VK_NULL_HANDLE;
    VmaAllocation m_stagingAlloc  = nullptr;
};
