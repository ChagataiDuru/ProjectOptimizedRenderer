#pragma once

#include "core/VulkanContext.h"
#include <unordered_map>

// Describes a unique sampler configuration.
// Covers the settings that vary across glTF materials.
struct SamplerDesc {
    VkFilter             magFilter        = VK_FILTER_LINEAR;
    VkFilter             minFilter        = VK_FILTER_LINEAR;
    VkSamplerMipmapMode  mipmapMode       = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    VkSamplerAddressMode addressModeU     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    VkSamplerAddressMode addressModeV     = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    float                maxAnisotropy    = 16.0f;
    bool                 enableAnisotropy = true;

    bool operator==(const SamplerDesc& other) const;
};

// Hash functor for SamplerDesc — used as unordered_map key.
struct SamplerDescHash {
    size_t operator()(const SamplerDesc& desc) const;
};

class SamplerCache {
public:
    explicit SamplerCache(VulkanContext& ctx);
    ~SamplerCache();

    SamplerCache(const SamplerCache&)            = delete;
    SamplerCache& operator=(const SamplerCache&) = delete;

    // Returns a shared VkSampler for the given description.
    // Creates a new one if no matching sampler exists.
    VkSampler getSampler(const SamplerDesc& desc);

    // Returns a default linear-repeat sampler (the most common case).
    VkSampler getDefaultSampler();

    void shutdown();

private:
    VulkanContext& m_ctx;
    std::unordered_map<SamplerDesc, VkSampler, SamplerDescHash> m_cache;
};
