#pragma once

#include "core/VulkanContext.h"
#include <vector>

// Forward declarations to avoid circular includes
class Buffer;
class Image;

// ── Layout ────────────────────────────────────────────────────────────────────

struct DescriptorSetLayoutBinding {
    uint32_t            binding;
    VkDescriptorType    type;
    uint32_t            count  = 1;
    VkShaderStageFlags  stages = VK_SHADER_STAGE_ALL;
};

class DescriptorSetLayout {
public:
    explicit DescriptorSetLayout(VulkanContext& ctx);
    ~DescriptorSetLayout();

    DescriptorSetLayout(const DescriptorSetLayout&)            = delete;
    DescriptorSetLayout& operator=(const DescriptorSetLayout&) = delete;

    void addBinding(const DescriptorSetLayoutBinding& binding);
    // Compiles accumulated bindings into the immutable VkDescriptorSetLayout object.
    void finalize();

    VkDescriptorSetLayout getLayout() const { return m_layout; }

private:
    VulkanContext&                        m_ctx;
    VkDescriptorSetLayout                 m_layout = VK_NULL_HANDLE;
    std::vector<VkDescriptorSetLayoutBinding> m_bindings;
};

// ── Pool ─────────────────────────────────────────────────────────────────────

class DescriptorPool {
public:
    DescriptorPool(VulkanContext& ctx, uint32_t maxSets);
    ~DescriptorPool();

    DescriptorPool(const DescriptorPool&)            = delete;
    DescriptorPool& operator=(const DescriptorPool&) = delete;

    void addPoolSize(VkDescriptorType type, uint32_t count);
    void create();

    VkDescriptorPool getPool() const { return m_pool; }

private:
    VulkanContext&                  m_ctx;
    VkDescriptorPool                m_pool    = VK_NULL_HANDLE;
    uint32_t                        m_maxSets = 0;
    std::vector<VkDescriptorPoolSize> m_sizes;
};

// ── Set ───────────────────────────────────────────────────────────────────────

class DescriptorSet {
public:
    explicit DescriptorSet(VulkanContext& ctx);
    ~DescriptorSet() = default; // Sets are freed with the pool, not individually

    DescriptorSet(const DescriptorSet&)            = delete;
    DescriptorSet& operator=(const DescriptorSet&) = delete;

    void allocate(DescriptorPool& pool, DescriptorSetLayout& layout);

    void writeBuffer(uint32_t binding, const Buffer& buffer,
                     VkDeviceSize offset = 0, VkDeviceSize range = VK_WHOLE_SIZE);
    void writeImage(uint32_t binding, const Image& image,
                    VkSampler sampler = VK_NULL_HANDLE,
                    VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // Reconstructs raw pointers from backing vectors and calls vkUpdateDescriptorSets.
    void update();

    VkDescriptorSet getSet() const { return m_set; }

private:
    VulkanContext&  m_ctx;
    VkDescriptorSet m_set = VK_NULL_HANDLE;

    // Backing storage for descriptor info structs.
    // Pointers into these are fixed up inside update() to handle vector reallocation.
    std::vector<VkDescriptorBufferInfo> m_bufferInfos;
    std::vector<VkDescriptorImageInfo>  m_imageInfos;

    // Writes referencing the above by index (converted to pointers in update())
    struct PendingWrite {
        VkWriteDescriptorSet write;
        uint32_t             bufferInfoIndex = 0; // valid when write.pBufferInfo != nullptr intent
        uint32_t             imageInfoIndex  = 0;
        bool                 isBuffer        = false;
    };
    std::vector<PendingWrite> m_pendingWrites;
};
