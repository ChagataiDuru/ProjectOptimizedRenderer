#include "core/VulkanUtil.h"

namespace vkutil {

void transitionImage(
    VkCommandBuffer       cmd,
    VkImage               image,
    VkPipelineStageFlags2 srcStage,  VkAccessFlags2 srcAccess,
    VkPipelineStageFlags2 dstStage,  VkAccessFlags2 dstAccess,
    VkImageLayout         oldLayout, VkImageLayout  newLayout,
    VkImageAspectFlags    aspectMask)
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
        .subresourceRange    = { aspectMask, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS },
    };
    const VkDependencyInfo dep{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &barrier,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
}

VkCommandBuffer beginSingleUseCommands(VulkanContext& ctx, VkCommandPool& outPool)
{
    const VkCommandPoolCreateInfo poolCI{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = ctx.getGraphicsQueueFamily(),
    };
    VK_CHECK(vkCreateCommandPool(ctx.getDevice(), &poolCI, nullptr, &outPool));

    const VkCommandBufferAllocateInfo allocInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = outPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(ctx.getDevice(), &allocInfo, &cmd));

    const VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));
    return cmd;
}

void endSingleUseCommands(VulkanContext& ctx, VkCommandPool pool, VkCommandBuffer cmd)
{
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkFence fence = VK_NULL_HANDLE;
    const VkFenceCreateInfo fenceCI{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VK_CHECK(vkCreateFence(ctx.getDevice(), &fenceCI, nullptr, &fence));

    const VkSubmitInfo submitInfo{
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &cmd,
    };
    VK_CHECK(vkQueueSubmit(ctx.getGraphicsQueue(), 1, &submitInfo, fence));
    VK_CHECK(vkWaitForFences(ctx.getDevice(), 1, &fence, VK_TRUE, UINT64_MAX));

    vkDestroyFence(ctx.getDevice(), fence, nullptr);
    vkDestroyCommandPool(ctx.getDevice(), pool, nullptr);
}

} // namespace vkutil
