#include "resource/SamplerCache.h"

#include <spdlog/spdlog.h>
#include <functional>
#include <stdexcept>

// ── SamplerDesc equality ──────────────────────────────────────────────────────

bool SamplerDesc::operator==(const SamplerDesc& other) const
{
    return magFilter        == other.magFilter
        && minFilter        == other.minFilter
        && mipmapMode       == other.mipmapMode
        && addressModeU     == other.addressModeU
        && addressModeV     == other.addressModeV
        && maxAnisotropy    == other.maxAnisotropy
        && enableAnisotropy == other.enableAnisotropy;
}

// ── SamplerDescHash ───────────────────────────────────────────────────────────

size_t SamplerDescHash::operator()(const SamplerDesc& desc) const
{
    // Combine field hashes with XOR-shift mixing to reduce collisions.
    auto combine = [](size_t seed, size_t val) -> size_t {
        return seed ^ (val + 0x9e3779b9 + (seed << 6) + (seed >> 2));
    };

    size_t h = 0;
    h = combine(h, std::hash<int>{}(static_cast<int>(desc.magFilter)));
    h = combine(h, std::hash<int>{}(static_cast<int>(desc.minFilter)));
    h = combine(h, std::hash<int>{}(static_cast<int>(desc.mipmapMode)));
    h = combine(h, std::hash<int>{}(static_cast<int>(desc.addressModeU)));
    h = combine(h, std::hash<int>{}(static_cast<int>(desc.addressModeV)));
    h = combine(h, std::hash<float>{}(desc.maxAnisotropy));
    h = combine(h, std::hash<bool>{}(desc.enableAnisotropy));
    return h;
}

// ── SamplerCache lifecycle ────────────────────────────────────────────────────

SamplerCache::SamplerCache(VulkanContext& ctx)
    : m_ctx(ctx)
{
}

SamplerCache::~SamplerCache()
{
    shutdown();
}

// ── getSampler ────────────────────────────────────────────────────────────────

VkSampler SamplerCache::getSampler(const SamplerDesc& desc)
{
    auto it = m_cache.find(desc);
    if (it != m_cache.end()) {
        return it->second;
    }

    // Clamp anisotropy to device limit.
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(m_ctx.getPhysicalDevice(), &props);
    const float clampedAniso = desc.enableAnisotropy
        ? std::min(desc.maxAnisotropy, props.limits.maxSamplerAnisotropy)
        : 1.0f;

    const VkSamplerCreateInfo samplerCI{
        .sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter               = desc.magFilter,
        .minFilter               = desc.minFilter,
        .mipmapMode              = desc.mipmapMode,
        .addressModeU            = desc.addressModeU,
        .addressModeV            = desc.addressModeV,
        .addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .mipLodBias              = 0.0f,
        .anisotropyEnable        = desc.enableAnisotropy ? VK_TRUE : VK_FALSE,
        .maxAnisotropy           = clampedAniso,
        .compareEnable           = VK_FALSE,
        .compareOp               = VK_COMPARE_OP_ALWAYS,
        .minLod                  = 0.0f,
        .maxLod                  = VK_LOD_CLAMP_NONE,
        .borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };

    VkSampler sampler = VK_NULL_HANDLE;
    if (vkCreateSampler(m_ctx.getDevice(), &samplerCI, nullptr, &sampler) != VK_SUCCESS) {
        throw std::runtime_error("SamplerCache: failed to create sampler");
    }

    spdlog::debug("SamplerCache: created new sampler (cache size now {})", m_cache.size() + 1);
    m_cache.emplace(desc, sampler);
    return sampler;
}

VkSampler SamplerCache::getDefaultSampler()
{
    return getSampler({});  // All defaults: linear, repeat, 16x aniso
}

// ── shutdown ──────────────────────────────────────────────────────────────────

void SamplerCache::shutdown()
{
    for (auto& [desc, sampler] : m_cache) {
        vkDestroySampler(m_ctx.getDevice(), sampler, nullptr);
    }
    m_cache.clear();
}
