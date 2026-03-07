#include "resource/Buffer.h"

#include <spdlog/spdlog.h>
#include <cstring>
#include <stdexcept>
#include <utility>

Buffer::Buffer(VulkanContext& ctx)
    : m_ctx(ctx)
{
}

Buffer::~Buffer()
{
    destroy();
}

Buffer::Buffer(Buffer&& other) noexcept
    : m_ctx(other.m_ctx)
    , m_buffer(other.m_buffer)
    , m_allocation(other.m_allocation)
    , m_size(other.m_size)
    , m_mappedPtr(other.m_mappedPtr)
    , m_stagingBuffer(other.m_stagingBuffer)
    , m_stagingAlloc(other.m_stagingAlloc)
{
    other.m_buffer        = VK_NULL_HANDLE;
    other.m_allocation    = nullptr;
    other.m_mappedPtr     = nullptr;
    other.m_stagingBuffer = VK_NULL_HANDLE;
    other.m_stagingAlloc  = nullptr;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept
{
    if (this != &other) {
        destroy();
        m_buffer        = other.m_buffer;
        m_allocation    = other.m_allocation;
        m_size          = other.m_size;
        m_mappedPtr     = other.m_mappedPtr;
        m_stagingBuffer = other.m_stagingBuffer;
        m_stagingAlloc  = other.m_stagingAlloc;
        other.m_buffer        = VK_NULL_HANDLE;
        other.m_allocation    = nullptr;
        other.m_mappedPtr     = nullptr;
        other.m_stagingBuffer = VK_NULL_HANDLE;
        other.m_stagingAlloc  = nullptr;
    }
    return *this;
}

void Buffer::createDeviceLocal(VkDeviceSize size, VkBufferUsageFlags usage)
{
    m_size = size;
    const VkBufferCreateInfo bufCI{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = size,
        // TRANSFER_DST_BIT allows data upload via uploadStaged()
        .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    };
    const VmaAllocationCreateInfo allocCI{
        .usage = VMA_MEMORY_USAGE_AUTO,
        // DEDICATED_MEMORY for large GPU-only resources (vertex/index buffers)
        .flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT,
    };
    VK_CHECK(vmaCreateBuffer(m_ctx.getAllocator(), &bufCI, &allocCI,
                             &m_buffer, &m_allocation, nullptr));
}

void Buffer::createHostVisible(VkDeviceSize size, VkBufferUsageFlags usage)
{
    m_size = size;
    const VkBufferCreateInfo bufCI{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = size,
        .usage = usage,
    };
    const VmaAllocationCreateInfo allocCI{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 // Persistent mapping: VMA keeps the pointer valid for the buffer's lifetime
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    VmaAllocationInfo allocInfo{};
    VK_CHECK(vmaCreateBuffer(m_ctx.getAllocator(), &bufCI, &allocCI,
                             &m_buffer, &m_allocation, &allocInfo));
    m_mappedPtr = allocInfo.pMappedData;
}

void Buffer::upload(const void* data, VkDeviceSize size, VkDeviceSize offset)
{
    if (!m_mappedPtr)
        throw std::runtime_error("Buffer::upload: buffer is not host-visible / not mapped");
    std::memcpy(static_cast<char*>(m_mappedPtr) + offset, data, static_cast<size_t>(size));
    vmaFlushAllocation(m_ctx.getAllocator(), m_allocation, offset, size);
}

void Buffer::uploadStaged(const void* data, VkDeviceSize size, VkCommandBuffer transferCmd)
{
    // Create a host-visible staging buffer that will be copied into this device-local buffer.
    // Staging memory is held in m_stagingBuffer/m_stagingAlloc until releaseStaging() is called
    // after the GPU has finished executing the recorded copy command.
    const VkBufferCreateInfo stagingCI{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size  = size,
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

    std::memcpy(stagingInfo.pMappedData, data, static_cast<size_t>(size));
    vmaFlushAllocation(m_ctx.getAllocator(), m_stagingAlloc, 0, VK_WHOLE_SIZE);

    const VkBufferCopy region{ .size = size };
    vkCmdCopyBuffer(transferCmd, m_stagingBuffer, m_buffer, 1, &region);
}

void Buffer::releaseStaging()
{
    if (m_stagingBuffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_ctx.getAllocator(), m_stagingBuffer, m_stagingAlloc);
        m_stagingBuffer = VK_NULL_HANDLE;
        m_stagingAlloc  = nullptr;
    }
}

void* Buffer::map()
{
    void* ptr = nullptr;
    VK_CHECK(vmaMapMemory(m_ctx.getAllocator(), m_allocation, &ptr));
    return ptr;
}

void Buffer::unmap()
{
    vmaUnmapMemory(m_ctx.getAllocator(), m_allocation);
}

void Buffer::copyFrom(VkCommandBuffer cmd, const Buffer& src, VkDeviceSize size,
                      VkDeviceSize srcOffset, VkDeviceSize dstOffset)
{
    const VkBufferCopy region{
        .srcOffset = srcOffset,
        .dstOffset = dstOffset,
        .size      = size,
    };
    vkCmdCopyBuffer(cmd, src.getBuffer(), m_buffer, 1, &region);
}

void Buffer::destroy()
{
    releaseStaging();
    if (m_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_ctx.getAllocator(), m_buffer, m_allocation);
        m_buffer     = VK_NULL_HANDLE;
        m_allocation = nullptr;
        m_mappedPtr  = nullptr;
    }
}
