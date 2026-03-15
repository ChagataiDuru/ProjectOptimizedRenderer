#include "resource/Texture.h"

#include <spdlog/spdlog.h>
#include <stb_image.h>
#include <stdexcept>
#include <utility>

// ── Lifecycle ─────────────────────────────────────────────────────────────────

Texture::Texture(VulkanContext& ctx)
    : m_image(ctx)
{
}

Texture::~Texture()
{
    destroy();
}

Texture::Texture(Texture&& other) noexcept
    : m_image(std::move(other.m_image))
    , m_sampler(other.m_sampler)
    , m_filepath(std::move(other.m_filepath))
{
    other.m_sampler = VK_NULL_HANDLE;
}

Texture& Texture::operator=(Texture&& other) noexcept
{
    if (this != &other) {
        destroy();
        m_image    = std::move(other.m_image);
        m_sampler  = other.m_sampler;
        m_filepath = std::move(other.m_filepath);
        other.m_sampler = VK_NULL_HANDLE;
    }
    return *this;
}

// ── Loading ───────────────────────────────────────────────────────────────────

void Texture::loadFromFile(const std::string& filepath,
                           VkCommandBuffer transferCmd,
                           SamplerCache& samplerCache)
{
    int w = 0, h = 0, channels = 0;
    stbi_uc* pixels = stbi_load(filepath.c_str(), &w, &h, &channels, STBI_rgb_alpha);
    if (!pixels) {
        throw std::runtime_error("Failed to load texture: " + filepath);
    }

    const VkDeviceSize dataSize = static_cast<VkDeviceSize>(w) * h * 4;
    m_image.createFromData(static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                           VK_FORMAT_R8G8B8A8_SRGB,
                           pixels, dataSize, transferCmd);
    stbi_image_free(pixels);

    m_sampler  = samplerCache.getDefaultSampler();
    m_filepath = filepath;

    spdlog::info("Texture loaded: {}x{} {}", w, h, filepath);
}

void Texture::loadFromMemory(const void* rgbaData,
                             uint32_t width, uint32_t height,
                             VkCommandBuffer transferCmd,
                             SamplerCache& samplerCache)
{
    const VkDeviceSize dataSize = static_cast<VkDeviceSize>(width) * height * 4;
    m_image.createFromData(width, height, VK_FORMAT_R8G8B8A8_SRGB,
                           rgbaData, dataSize, transferCmd);

    m_sampler = samplerCache.getDefaultSampler();
}

void Texture::createSolidColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a,
                               VkCommandBuffer transferCmd,
                               SamplerCache& samplerCache)
{
    const uint8_t pixels[4] = { r, g, b, a };
    m_image.createFromData(1, 1, VK_FORMAT_R8G8B8A8_SRGB, pixels, 4, transferCmd);
    m_sampler = samplerCache.getDefaultSampler();
}

// ── Cleanup ───────────────────────────────────────────────────────────────────

void Texture::releaseStaging()
{
    m_image.releaseStaging();
}

void Texture::destroy()
{
    m_image.destroy();
    m_sampler = VK_NULL_HANDLE;  // NOT owned — do not call vkDestroySampler
}
