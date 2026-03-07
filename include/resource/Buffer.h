#pragma once

#include "core/VulkanContext.h"

class Buffer {
public:
    explicit Buffer(VulkanContext& ctx);
    ~Buffer();

    Buffer(Buffer&&) noexcept;
    Buffer& operator=(Buffer&&) noexcept;
    Buffer(const Buffer&)            = delete;
    Buffer& operator=(const Buffer&) = delete;

    // Device-local: optimal for GPU-only data (vertex/index/storage buffers).
    // TRANSFER_DST_BIT is added automatically to allow staged uploads.
    void createDeviceLocal(VkDeviceSize size, VkBufferUsageFlags usage);

    // Host-visible + persistently mapped: for uniform buffers, staging, streaming data.
    void createHostVisible(VkDeviceSize size, VkBufferUsageFlags usage);

    // Write directly into a host-visible mapped buffer.
    void upload(const void* data, VkDeviceSize size, VkDeviceSize offset = 0);

    // Record a copy from an internal staging buffer into this device-local buffer.
    // The staging buffer is stored as m_stagingBuffer/m_stagingAlloc and must remain
    // alive until the recorded command has been submitted and the GPU has finished.
    // Call releaseStaging() after the fence for that submission signals.
    void uploadStaged(const void* data, VkDeviceSize size, VkCommandBuffer transferCmd);
    void releaseStaging();

    // Map/unmap for non-persistent access (host-visible buffers only).
    void* map();
    void  unmap();

    // Record a buffer-to-buffer copy on cmd (src → this).
    void copyFrom(VkCommandBuffer cmd, const Buffer& src, VkDeviceSize size,
                  VkDeviceSize srcOffset = 0, VkDeviceSize dstOffset = 0);

    void destroy();

    VkBuffer      getBuffer()     const { return m_buffer; }
    VkDeviceSize  getSize()       const { return m_size; }
    VmaAllocation getAllocation()  const { return m_allocation; }
    void*         getMappedPtr()  const { return m_mappedPtr; }

private:
    VulkanContext& m_ctx;
    VkBuffer       m_buffer     = VK_NULL_HANDLE;
    VmaAllocation  m_allocation = nullptr;
    VkDeviceSize   m_size       = 0;
    void*          m_mappedPtr  = nullptr;

    // Staging resources held between uploadStaged() and releaseStaging()
    VkBuffer      m_stagingBuffer = VK_NULL_HANDLE;
    VmaAllocation m_stagingAlloc  = nullptr;
};
