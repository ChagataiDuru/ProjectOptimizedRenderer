#include "resource/Image.h"
#include "core/VulkanUtil.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

// ── Layout transition helper ──────────────────────────────────────────────────

// Maps a VkImageLayout to the pipeline stage and access mask that make sense
// for a synchronization2 barrier. Covers the common render/transfer layouts.
static std::pair<VkPipelineStageFlags2, VkAccessFlags2> layoutToStageAccess(VkImageLayout layout)
{
    switch (layout) {
        case VK_IMAGE_LAYOUT_UNDEFINED:
            return { VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0 };
        case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
            return { VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT };
        case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
            return { VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT };
        case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
            return { VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT };
        case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
            return { VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT };
        case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
            return { VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT };
        case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
            return { VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, 0 };
        default:
            return { VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                     VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT };
    }
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

Image::Image(VulkanContext& ctx)
    : m_ctx(ctx)
{
}

Image::~Image()
{
    destroy();
}

Image::Image(Image&& other) noexcept
    : m_ctx(other.m_ctx)
    , m_image(other.m_image)
    , m_imageView(other.m_imageView)
    , m_allocation(other.m_allocation)
    , m_format(other.m_format)
    , m_extent(other.m_extent)
    , m_arrayLayers(other.m_arrayLayers)
    , m_mipLevels(other.m_mipLevels)
    , m_samples(other.m_samples)
    , m_stagingBuffer(other.m_stagingBuffer)
    , m_stagingAlloc(other.m_stagingAlloc)
{
    other.m_image         = VK_NULL_HANDLE;
    other.m_imageView     = VK_NULL_HANDLE;
    other.m_allocation    = nullptr;
    other.m_arrayLayers   = 1;
    other.m_mipLevels     = 1;
    other.m_samples       = VK_SAMPLE_COUNT_1_BIT;
    other.m_stagingBuffer = VK_NULL_HANDLE;
    other.m_stagingAlloc  = nullptr;
}

Image& Image::operator=(Image&& other) noexcept
{
    if (this != &other) {
        destroy();
        m_image         = other.m_image;
        m_imageView     = other.m_imageView;
        m_allocation    = other.m_allocation;
        m_format        = other.m_format;
        m_extent        = other.m_extent;
        m_arrayLayers   = other.m_arrayLayers;
        m_mipLevels     = other.m_mipLevels;
        m_samples       = other.m_samples;
        m_stagingBuffer = other.m_stagingBuffer;
        m_stagingAlloc  = other.m_stagingAlloc;
        other.m_image         = VK_NULL_HANDLE;
        other.m_imageView     = VK_NULL_HANDLE;
        other.m_allocation    = nullptr;
        other.m_arrayLayers   = 1;
        other.m_mipLevels     = 1;
        other.m_samples       = VK_SAMPLE_COUNT_1_BIT;
        other.m_stagingBuffer = VK_NULL_HANDLE;
        other.m_stagingAlloc  = nullptr;
    }
    return *this;
}

// ── Creation ──────────────────────────────────────────────────────────────────

void Image::create(uint32_t width, uint32_t height, uint32_t depth,
                   VkFormat format, VkImageUsageFlags usage,
                   VkImageAspectFlags aspectFlags, uint32_t arrayLayers,
                   VkSampleCountFlagBits samples, uint32_t mipLevels)
{
    m_format      = format;
    m_extent      = { width, height, depth };
    m_arrayLayers = arrayLayers;
    m_mipLevels   = std::max(mipLevels, 1u);
    m_samples     = samples;

    const VkImageCreateInfo imageCI{
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType     = depth > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
        .format        = format,
        .extent        = m_extent,
        .mipLevels     = m_mipLevels,
        .arrayLayers   = arrayLayers,
        .samples       = samples,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = usage,
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    const VmaAllocationCreateInfo allocCI{
        .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    VK_CHECK(vmaCreateImage(m_ctx.getAllocator(), &imageCI, &allocCI,
                            &m_image, &m_allocation, nullptr));
    createView(aspectFlags);
}

void Image::createFromData(uint32_t width, uint32_t height, VkFormat format,
                           const void* data, VkDeviceSize dataSize,
                           VkCommandBuffer transferCmd,
                           bool generateMips)
{
    // 1. Create the device-local image
    const uint32_t mipLevels = (generateMips && supportsLinearBlit(format))
        ? fullMipChainLevels(width, height)
        : 1u;
    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (mipLevels > 1) {
        usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }
    create(width, height, 1, format, usage, VK_IMAGE_ASPECT_COLOR_BIT,
           1, VK_SAMPLE_COUNT_1_BIT, mipLevels);

    // 2. Staging buffer (held alive until GPU finishes — caller calls destroy() or we keep it)
    const VkBufferCreateInfo stagingCI{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = dataSize,
        .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    };
    const VmaAllocationCreateInfo stagingAllocCI{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    VmaAllocationInfo stagingInfo{};
    VK_CHECK(vmaCreateBuffer(m_ctx.getAllocator(), &stagingCI, &stagingAllocCI,
                             &m_stagingBuffer, &m_stagingAlloc, &stagingInfo));
    std::memcpy(stagingInfo.pMappedData, data, static_cast<size_t>(dataSize));
    vmaFlushAllocation(m_ctx.getAllocator(), m_stagingAlloc, 0, VK_WHOLE_SIZE);

    // 3. UNDEFINED → TRANSFER_DST_OPTIMAL
    transitionLayout(transferCmd, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    // 4. Copy staging buffer into image
    const VkBufferImageCopy region{
        .imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .imageExtent      = m_extent,
    };
    vkCmdCopyBufferToImage(transferCmd, m_stagingBuffer, m_image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // 5. Build the mip chain (ends in SHADER_READ_ONLY_OPTIMAL), or
    //    TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL for a single level.
    if (m_mipLevels > 1) {
        generateMipmaps(transferCmd);
    } else {
        transitionLayout(transferCmd,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
}

uint32_t Image::fullMipChainLevels(uint32_t width, uint32_t height)
{
    const uint32_t largest = std::max(std::max(width, height), 1u);
    return static_cast<uint32_t>(std::floor(std::log2(static_cast<double>(largest)))) + 1u;
}

bool Image::supportsLinearBlit(VkFormat format) const
{
    VkFormatProperties props{};
    vkGetPhysicalDeviceFormatProperties(m_ctx.getPhysicalDevice(), format, &props);
    const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_BLIT_SRC_BIT |
                                          VK_FORMAT_FEATURE_BLIT_DST_BIT |
                                          VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    const bool supported = (props.optimalTilingFeatures & required) == required;
    if (!supported) {
        static bool warned = false;
        if (!warned) {
            spdlog::warn("Format {} cannot be linearly blitted; textures keep a single mip level",
                         static_cast<int>(format));
            warned = true;
        }
    }
    return supported;
}

void Image::generateMipmaps(VkCommandBuffer cmd)
{
    auto levelBarrier = [&](uint32_t level,
                            VkImageLayout oldLayout, VkImageLayout newLayout,
                            VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                            VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
        const VkImageMemoryBarrier2 barrier{
            .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask        = srcStage,
            .srcAccessMask       = srcAccess,
            .dstStageMask        = dstStage,
            .dstAccessMask       = dstAccess,
            .oldLayout           = oldLayout,
            .newLayout           = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = m_image,
            .subresourceRange    = { VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, m_arrayLayers },
        };
        const VkDependencyInfo dep{
            .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers    = &barrier,
        };
        vkCmdPipelineBarrier2(cmd, &dep);
    };

    int32_t srcWidth  = static_cast<int32_t>(m_extent.width);
    int32_t srcHeight = static_cast<int32_t>(m_extent.height);
    for (uint32_t level = 1; level < m_mipLevels; ++level) {
        const int32_t dstWidth  = std::max(srcWidth / 2, 1);
        const int32_t dstHeight = std::max(srcHeight / 2, 1);

        levelBarrier(level - 1,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

        // Blits of sRGB formats filter in linear space, so the chain stays correct.
        const VkImageBlit2 blit{
            .sType          = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
            .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, m_arrayLayers },
            .srcOffsets     = { { 0, 0, 0 }, { srcWidth, srcHeight, 1 } },
            .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, 0, m_arrayLayers },
            .dstOffsets     = { { 0, 0, 0 }, { dstWidth, dstHeight, 1 } },
        };
        const VkBlitImageInfo2 blitInfo{
            .sType          = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
            .srcImage       = m_image,
            .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .dstImage       = m_image,
            .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .regionCount    = 1,
            .pRegions       = &blit,
            .filter         = VK_FILTER_LINEAR,
        };
        vkCmdBlitImage2(cmd, &blitInfo);

        levelBarrier(level - 1,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);

        srcWidth  = dstWidth;
        srcHeight = dstHeight;
    }

    levelBarrier(m_mipLevels - 1,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT);
}

void Image::createView(VkImageAspectFlags aspectFlags)
{
    VkImageViewType viewType;
    if (m_extent.depth > 1)
        viewType = VK_IMAGE_VIEW_TYPE_3D;
    else if (m_arrayLayers > 1)
        viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    else
        viewType = VK_IMAGE_VIEW_TYPE_2D;

    const VkImageViewCreateInfo viewCI{
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image    = m_image,
        .viewType = viewType,
        .format   = m_format,
        .components = {
            VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
        },
        .subresourceRange = { aspectFlags, 0, m_mipLevels, 0, m_arrayLayers },
    };
    VK_CHECK(vkCreateImageView(m_ctx.getDevice(), &viewCI, nullptr, &m_imageView));
}

VkImageView Image::createSingleLayerView(uint32_t layer,
                                          VkImageAspectFlags aspectFlags) const
{
    const VkImageViewCreateInfo viewCI{
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image    = m_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format   = m_format,
        .components = {
            VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
        },
        .subresourceRange = { aspectFlags, 0, 1, layer, 1 },
    };
    VkImageView view = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(m_ctx.getDevice(), &viewCI, nullptr, &view));
    return view;
}

VkImageView Image::createMipView(uint32_t level) const
{
    const VkImageViewCreateInfo viewCI{
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image    = m_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format   = m_format,
        .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 1 },
    };
    VkImageView view = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(m_ctx.getDevice(), &viewCI, nullptr, &view));
    return view;
}

// ── Barriers ─────────────────────────────────────────────────────────────────

void Image::transitionLayout(VkCommandBuffer cmd,
                             VkImageLayout oldLayout, VkImageLayout newLayout)
{
    auto [srcStage, srcAccess] = layoutToStageAccess(oldLayout);
    auto [dstStage, dstAccess] = layoutToStageAccess(newLayout);
    vkutil::transitionImage(cmd, m_image,
                            srcStage, srcAccess,
                            dstStage, dstAccess,
                            oldLayout, newLayout,
                            VK_IMAGE_ASPECT_COLOR_BIT);
}

// ── Cleanup ───────────────────────────────────────────────────────────────────

void Image::releaseStaging()
{
    if (m_stagingBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_ctx.getAllocator(), m_stagingBuffer, m_stagingAlloc);
        m_stagingBuffer = VK_NULL_HANDLE;
        m_stagingAlloc  = nullptr;
    }
}

void Image::destroy()
{
    releaseStaging();
    if (m_imageView != VK_NULL_HANDLE) {
        vkDestroyImageView(m_ctx.getDevice(), m_imageView, nullptr);
        m_imageView = VK_NULL_HANDLE;
    }
    if (m_image != VK_NULL_HANDLE) {
        vmaDestroyImage(m_ctx.getAllocator(), m_image, m_allocation);
        m_image      = VK_NULL_HANDLE;
        m_allocation = nullptr;
    }
    m_samples   = VK_SAMPLE_COUNT_1_BIT;
    m_mipLevels = 1;
}
