#pragma once

#include "core/VulkanContext.h"

namespace vkutil {

/// Record a VkImageMemoryBarrier2 pipeline barrier for a single image layout transition.
/// Uses synchronization2 (core in Vulkan 1.3/1.4).
///
/// @param aspectMask  Defaults to VK_IMAGE_ASPECT_COLOR_BIT; pass
///                    VK_IMAGE_ASPECT_DEPTH_BIT for depth images.
void transitionImage(
    VkCommandBuffer       cmd,
    VkImage               image,
    VkPipelineStageFlags2 srcStage,  VkAccessFlags2 srcAccess,
    VkPipelineStageFlags2 dstStage,  VkAccessFlags2 dstAccess,
    VkImageLayout         oldLayout, VkImageLayout  newLayout,
    VkImageAspectFlags    aspectMask = VK_IMAGE_ASPECT_COLOR_BIT);

} // namespace vkutil
