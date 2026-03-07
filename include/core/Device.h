#pragma once

#include "core/VulkanContext.h"

// Wraps VulkanContext and serves as the extensibility point for frame
// synchronization, command pools, and per-frame resources (Phase 0.3+).
class Device {
public:
    explicit Device(VulkanContext& context);
    ~Device() = default;

    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;

    VulkanContext&   getContext()        const { return m_context; }
    VkDevice         getDevice()         const { return m_context.getDevice(); }
    VkPhysicalDevice getPhysicalDevice() const { return m_context.getPhysicalDevice(); }

private:
    VulkanContext& m_context;
};
