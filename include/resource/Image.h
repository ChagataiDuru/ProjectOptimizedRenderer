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

    void create(uint32_t width, uint32_t height, uint32_t depth,
                VkFormat format, VkImageUsageFlags usage,
                VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT);

    // Upload pixel data via an internally-created staging buffer recorded into transferCmd.
    // Caller must submit transferCmd and wait for completion before using the image.
    // Final layout after upload: VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL.
    void createFromData(uint32_t width, uint32_t height, VkFormat format,
                        const void* data, VkDeviceSize dataSize,
                        VkCommandBuffer transferCmd);

    void transitionLayout(VkCommandBuffer cmd,
                          VkImageLayout oldLayout, VkImageLayout newLayout);

    // Release staging buffer after GPU transfer is complete.
    // Call after the transfer command buffer has been submitted and the fence has signaled.
    void releaseStaging();

    void destroy();

    VkImage     getImage()     const { return m_image; }
    VkImageView getImageView() const { return m_imageView; }
    VkFormat    getFormat()    const { return m_format; }
    VkExtent3D  getExtent()    const { return m_extent; }

private:
    void createView(VkImageAspectFlags aspectFlags);

    VulkanContext& m_ctx;
    VkImage        m_image      = VK_NULL_HANDLE;
    VkImageView    m_imageView  = VK_NULL_HANDLE;
    VmaAllocation  m_allocation = nullptr;
    VkFormat       m_format     = VK_FORMAT_UNDEFINED;
    VkExtent3D     m_extent     = {};

    // Staging resources for createFromData — released after GPU completion
    VkBuffer      m_stagingBuffer = VK_NULL_HANDLE;
    VmaAllocation m_stagingAlloc  = nullptr;
};
