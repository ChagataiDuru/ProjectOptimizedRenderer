#include "resource/ResourceSet.h"

#include <spdlog/spdlog.h>
#include <stdexcept>

ResourceSet::ResourceSet(VulkanContext& ctx)
    : m_ctx(ctx)
{
}

Buffer& ResourceSet::createBuffer(const std::string& name,
                                  VkDeviceSize size, VkBufferUsageFlags usage,
                                  bool hostVisible)
{
    auto buf = std::make_unique<Buffer>(m_ctx);
    if (hostVisible)
        buf->createHostVisible(size, usage);
    else
        buf->createDeviceLocal(size, usage);

    auto& ref = *buf;
    m_buffers[name] = std::move(buf);
    spdlog::debug("ResourceSet: created buffer '{}'", name);
    return ref;
}

Image& ResourceSet::createImage(const std::string& name,
                                uint32_t width, uint32_t height,
                                VkFormat format, VkImageUsageFlags usage)
{
    auto img = std::make_unique<Image>(m_ctx);
    img->create(width, height, 1, format, usage, VK_IMAGE_ASPECT_COLOR_BIT);

    auto& ref = *img;
    m_images[name] = std::move(img);
    spdlog::debug("ResourceSet: created image '{}' ({}x{})", name, width, height);
    return ref;
}

ShaderModule& ResourceSet::loadShader(const std::string& name,
                                      const char* spirvPath,
                                      VkShaderStageFlagBits stage)
{
    m_shaders[name] = ShaderCompiler::loadFromFile(m_ctx, spirvPath, stage);
    spdlog::debug("ResourceSet: loaded shader '{}'", name);
    return m_shaders[name];
}

Buffer& ResourceSet::getBuffer(const std::string& name)
{
    auto it = m_buffers.find(name);
    if (it == m_buffers.end())
        throw std::runtime_error("ResourceSet: buffer not found: " + name);
    return *it->second;
}

Image& ResourceSet::getImage(const std::string& name)
{
    auto it = m_images.find(name);
    if (it == m_images.end())
        throw std::runtime_error("ResourceSet: image not found: " + name);
    return *it->second;
}

ShaderModule& ResourceSet::getShader(const std::string& name)
{
    auto it = m_shaders.find(name);
    if (it == m_shaders.end())
        throw std::runtime_error("ResourceSet: shader not found: " + name);
    return it->second;
}

void ResourceSet::releaseShaders()
{
    for (auto& [name, mod] : m_shaders)
        ShaderCompiler::destroyModule(m_ctx, mod);
    m_shaders.clear();
}

void ResourceSet::clear()
{
    releaseShaders();
    m_images.clear();
    m_buffers.clear();
}
