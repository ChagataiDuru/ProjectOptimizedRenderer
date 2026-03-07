#include "resource/Descriptor.h"
#include "resource/Buffer.h"
#include "resource/Image.h"

#include <spdlog/spdlog.h>
#include <stdexcept>

// ── DescriptorSetLayout ───────────────────────────────────────────────────────

DescriptorSetLayout::DescriptorSetLayout(VulkanContext& ctx)
    : m_ctx(ctx)
{
}

DescriptorSetLayout::~DescriptorSetLayout()
{
    if (m_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(m_ctx.getDevice(), m_layout, nullptr);
}

void DescriptorSetLayout::addBinding(const DescriptorSetLayoutBinding& b)
{
    m_bindings.push_back({
        .binding            = b.binding,
        .descriptorType     = b.type,
        .descriptorCount    = b.count,
        .stageFlags         = b.stages,
        .pImmutableSamplers = nullptr,
    });
}

void DescriptorSetLayout::finalize()
{
    const VkDescriptorSetLayoutCreateInfo ci{
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(m_bindings.size()),
        .pBindings    = m_bindings.data(),
    };
    VK_CHECK(vkCreateDescriptorSetLayout(m_ctx.getDevice(), &ci, nullptr, &m_layout));
    spdlog::debug("DescriptorSetLayout finalized ({} bindings)", m_bindings.size());
}

// ── DescriptorPool ────────────────────────────────────────────────────────────

DescriptorPool::DescriptorPool(VulkanContext& ctx, uint32_t maxSets)
    : m_ctx(ctx), m_maxSets(maxSets)
{
}

DescriptorPool::~DescriptorPool()
{
    if (m_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(m_ctx.getDevice(), m_pool, nullptr);
}

void DescriptorPool::addPoolSize(VkDescriptorType type, uint32_t count)
{
    m_sizes.push_back({ type, count });
}

void DescriptorPool::create()
{
    const VkDescriptorPoolCreateInfo ci{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = m_maxSets,
        .poolSizeCount = static_cast<uint32_t>(m_sizes.size()),
        .pPoolSizes    = m_sizes.data(),
    };
    VK_CHECK(vkCreateDescriptorPool(m_ctx.getDevice(), &ci, nullptr, &m_pool));
    spdlog::debug("DescriptorPool created (maxSets={})", m_maxSets);
}

// ── DescriptorSet ─────────────────────────────────────────────────────────────

DescriptorSet::DescriptorSet(VulkanContext& ctx)
    : m_ctx(ctx)
{
}

void DescriptorSet::allocate(DescriptorPool& pool, DescriptorSetLayout& layout)
{
    const VkDescriptorSetLayout rawLayout = layout.getLayout();
    const VkDescriptorSetAllocateInfo ai{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = pool.getPool(),
        .descriptorSetCount = 1,
        .pSetLayouts        = &rawLayout,
    };
    VK_CHECK(vkAllocateDescriptorSets(m_ctx.getDevice(), &ai, &m_set));
}

void DescriptorSet::writeBuffer(uint32_t binding, const Buffer& buffer,
                                VkDeviceSize offset, VkDeviceSize range)
{
    const uint32_t idx = static_cast<uint32_t>(m_bufferInfos.size());
    m_bufferInfos.push_back({ buffer.getBuffer(), offset, range });

    PendingWrite pw{};
    pw.isBuffer        = true;
    pw.bufferInfoIndex = idx;
    pw.write = {
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet          = m_set,
        .dstBinding      = binding,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        // pBufferInfo pointer fixed up in update() after all writes are collected
    };
    m_pendingWrites.push_back(pw);
}

void DescriptorSet::writeImage(uint32_t binding, const Image& image,
                               VkSampler sampler, VkImageLayout layout)
{
    const uint32_t idx = static_cast<uint32_t>(m_imageInfos.size());
    m_imageInfos.push_back({ sampler, image.getImageView(), layout });

    PendingWrite pw{};
    pw.isBuffer       = false;
    pw.imageInfoIndex = idx;
    pw.write = {
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet          = m_set,
        .dstBinding      = binding,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
    };
    m_pendingWrites.push_back(pw);
}

void DescriptorSet::update()
{
    // Reconstruct raw pointers from backing vectors here — not in write*() — because
    // vector reallocation between write calls would invalidate previously captured pointers.
    std::vector<VkWriteDescriptorSet> writes;
    writes.reserve(m_pendingWrites.size());

    for (auto& pw : m_pendingWrites) {
        VkWriteDescriptorSet w = pw.write;
        if (pw.isBuffer)
            w.pBufferInfo = &m_bufferInfos[pw.bufferInfoIndex];
        else
            w.pImageInfo  = &m_imageInfos[pw.imageInfoIndex];
        writes.push_back(w);
    }

    vkUpdateDescriptorSets(m_ctx.getDevice(),
                           static_cast<uint32_t>(writes.size()), writes.data(),
                           0, nullptr);
    m_pendingWrites.clear();
}
