#pragma once

#include "resource/Buffer.h"
#include "resource/Image.h"
#include "resource/Shader.h"
#include <memory>
#include <string>
#include <unordered_map>

// Lightweight named resource container with automatic RAII cleanup.
// Intended for grouping per-pass or per-material GPU resources.
class ResourceSet {
public:
    explicit ResourceSet(VulkanContext& ctx);
    ~ResourceSet() = default;

    ResourceSet(const ResourceSet&)            = delete;
    ResourceSet& operator=(const ResourceSet&) = delete;

    // Create and register a named buffer. Returns a ref to the owned Buffer.
    Buffer& createBuffer(const std::string& name,
                         VkDeviceSize size, VkBufferUsageFlags usage,
                         bool hostVisible = false);

    // Create and register a named 2D image.
    Image& createImage(const std::string& name,
                       uint32_t width, uint32_t height,
                       VkFormat format, VkImageUsageFlags usage);

    // Load a pre-compiled .spv and register by name. Overwrites if name already exists.
    ShaderModule& loadShader(const std::string& name,
                             const char* spirvPath,
                             VkShaderStageFlagBits stage);

    Buffer&      getBuffer(const std::string& name);
    Image&       getImage(const std::string& name);
    ShaderModule& getShader(const std::string& name);

    // Destroy all shader modules (call after pipeline creation).
    void releaseShaders();

    // Destroy all resources (called automatically by destructor).
    void clear();

private:
    VulkanContext& m_ctx;
    std::unordered_map<std::string, std::unique_ptr<Buffer>>      m_buffers;
    std::unordered_map<std::string, std::unique_ptr<Image>>       m_images;
    std::unordered_map<std::string, ShaderModule>                 m_shaders;
};
