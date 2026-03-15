#pragma once

#include "resource/Image.h"
#include "resource/SamplerCache.h"
#include <string>

class Texture {
public:
    explicit Texture(VulkanContext& ctx);
    ~Texture();

    Texture(Texture&&) noexcept;
    Texture& operator=(Texture&&) noexcept;
    Texture(const Texture&)            = delete;
    Texture& operator=(const Texture&) = delete;

    // Load from a file on disk (JPG, PNG, BMP, TGA via stb_image).
    // Decodes to RGBA8, uploads via staging buffer recorded into transferCmd.
    // Caller must submit transferCmd and wait for fence before using the texture.
    // Caller must call releaseStaging() after the fence signals.
    void loadFromFile(const std::string& filepath,
                      VkCommandBuffer transferCmd,
                      SamplerCache& samplerCache);

    // Load from raw RGBA8 pixel data already in memory.
    // Used for embedded glTF textures or procedurally generated textures.
    void loadFromMemory(const void* rgbaData,
                        uint32_t width, uint32_t height,
                        VkCommandBuffer transferCmd,
                        SamplerCache& samplerCache);

    // Create a 1x1 solid-color texture (for fallback/default).
    void createSolidColor(uint8_t r, uint8_t g, uint8_t b, uint8_t a,
                          VkCommandBuffer transferCmd,
                          SamplerCache& samplerCache);

    void releaseStaging();
    void destroy();

    VkImageView        getImageView() const { return m_image.getImageView(); }
    VkSampler          getSampler()   const { return m_sampler; }
    bool               isValid()      const { return m_image.getImage() != VK_NULL_HANDLE; }
    const std::string& getPath()      const { return m_filepath; }

private:
    Image       m_image;
    VkSampler   m_sampler  = VK_NULL_HANDLE;  // NOT owned — SamplerCache owns lifetime
    std::string m_filepath;
};
