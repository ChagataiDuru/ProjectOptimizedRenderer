#include "debug/Screenshot.h"
#include "core/Swapchain.h"
#include "resource/Buffer.h"

#include <spdlog/spdlog.h>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <stdexcept>

// stb_image_write — single-TU implementation, same pattern as stb_image
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

// ── Synchronization2 image layout transition helper ───────────────────────────
// (local duplicate of the one in Renderer.cpp — Screenshot is a standalone module)
static void transitionImage(
    VkCommandBuffer       cmd,
    VkImage               image,
    VkPipelineStageFlags2 srcStage,  VkAccessFlags2 srcAccess,
    VkPipelineStageFlags2 dstStage,  VkAccessFlags2 dstAccess,
    VkImageLayout         oldLayout, VkImageLayout  newLayout,
    VkImageAspectFlags    aspectMask = VK_IMAGE_ASPECT_COLOR_BIT)
{
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
        .image               = image,
        .subresourceRange    = { aspectMask, 0, 1, 0, 1 },
    };
    const VkDependencyInfo dep{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &barrier,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}

// ── Screenshot ────────────────────────────────────────────────────────────────

Screenshot::Screenshot(VulkanContext& ctx)
    : m_ctx(ctx)
{
}

void Screenshot::capture(const Swapchain& swapchain, const std::string& filename)
{
    // Resolve output path
    std::string outputPath = filename;
    if (outputPath.empty()) {
        auto now  = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        char buf[64];
        std::strftime(buf, sizeof(buf), "screenshots/POR_%Y%m%d_%H%M%S.png",
                      std::localtime(&time));
        outputPath = buf;
    }

    // Ensure the output directory exists
    const auto dir = std::filesystem::path(outputPath).parent_path();
    if (!dir.empty())
        std::filesystem::create_directories(dir);

    const VkExtent2D ext   = swapchain.getExtent();
    const uint32_t   w     = ext.width;
    const uint32_t   h     = ext.height;
    const VkDeviceSize bufSize = static_cast<VkDeviceSize>(w) * h * 4;

    // Host-visible readback buffer
    Buffer readback(m_ctx);
    readback.createHostVisible(bufSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

    // Transient command pool + single-use command buffer
    const VkCommandPoolCreateInfo poolCI{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = m_ctx.getGraphicsQueueFamily(),
    };
    VkCommandPool pool;
    VK_CHECK(vkCreateCommandPool(m_ctx.getDevice(), &poolCI, nullptr, &pool));

    const VkCommandBufferAllocateInfo allocCI{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = pool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(m_ctx.getDevice(), &allocCI, &cmd));

    const VkCommandBufferBeginInfo beginCI{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginCI));

    VkImage srcImage = swapchain.getCurrentImage();

    // Transition swapchain image PRESENT_SRC_KHR → TRANSFER_SRC_OPTIMAL
    transitionImage(cmd, srcImage,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,  VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,       VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    const VkBufferImageCopy region{
        .bufferOffset      = 0,
        .bufferRowLength   = 0,
        .bufferImageHeight = 0,
        .imageSubresource  = {
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel       = 0,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        },
        .imageOffset = { 0, 0, 0 },
        .imageExtent = { w, h, 1 },
    };
    vkCmdCopyImageToBuffer(cmd, srcImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           readback.getBuffer(), 1, &region);

    // Restore to PRESENT_SRC_KHR so the layout is consistent for any further use
    transitionImage(cmd, srcImage,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT,      VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,  VK_ACCESS_2_NONE,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,  VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VK_CHECK(vkEndCommandBuffer(cmd));

    VkFence fence;
    const VkFenceCreateInfo fenceCI{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VK_CHECK(vkCreateFence(m_ctx.getDevice(), &fenceCI, nullptr, &fence));

    const VkSubmitInfo submitCI{
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &cmd,
    };
    VK_CHECK(vkQueueSubmit(m_ctx.getGraphicsQueue(), 1, &submitCI, fence));
    VK_CHECK(vkWaitForFences(m_ctx.getDevice(), 1, &fence, VK_TRUE, UINT64_MAX));

    vkDestroyFence(m_ctx.getDevice(), fence, nullptr);
    vkDestroyCommandPool(m_ctx.getDevice(), pool, nullptr);

    // Map, swizzle BGRA → RGBA (swapchain is VK_FORMAT_B8G8R8A8_SRGB), write PNG
    void* ptr = readback.map();
    auto* pixels = static_cast<uint8_t*>(ptr);
    for (uint32_t i = 0; i < w * h; ++i) {
        std::swap(pixels[i * 4 + 0], pixels[i * 4 + 2]); // B ↔ R
    }

    const int written = stbi_write_png(outputPath.c_str(),
                                       static_cast<int>(w),
                                       static_cast<int>(h),
                                       4, pixels,
                                       static_cast<int>(w * 4));
    readback.unmap();
    readback.destroy();

    if (written)
        spdlog::info("Screenshot saved: {}", outputPath);
    else
        spdlog::error("Screenshot failed: stbi_write_png returned 0 for '{}'", outputPath);
}
