#include "resource/RendererResourceManager.h"

#include "core/VulkanUtil.h"
#include "resource/Vertex.h"

#include <spdlog/spdlog.h>
#include <stdexcept>

RendererResourceManager::RendererResourceManager(VulkanContext& ctx)
    : m_ctx(ctx)
    , m_vertexBuffer(ctx)
    , m_indexBuffer(ctx)
    , m_samplerCache(ctx)
    , m_fallbackWhite(ctx)
{
}

RendererResourceManager::~RendererResourceManager()
{
    clear();
}

void RendererResourceManager::uploadModel(const Model& model)
{
    clear();

    m_materials = model.materials;

    std::vector<Vertex> allVertices;
    std::vector<uint32_t> allIndices;
    uint32_t vertexBase = 0;

    m_meshes.reserve(model.meshes.size());
    for (const auto& mesh : model.meshes) {
        MeshRenderData data{};
        data.firstIndex = static_cast<uint32_t>(allIndices.size());
        data.indexCount = static_cast<uint32_t>(mesh.indices.size());
        data.boundsMin = mesh.boundsMin;
        data.boundsMax = mesh.boundsMax;
        m_meshes.push_back(data);

        allVertices.insert(allVertices.end(), mesh.vertices.begin(), mesh.vertices.end());
        for (uint32_t idx : mesh.indices) {
            allIndices.push_back(idx + vertexBase);
        }

        vertexBase += static_cast<uint32_t>(mesh.vertices.size());
    }

    if (allVertices.empty() || allIndices.empty()) {
        throw std::runtime_error("RendererResourceManager: model has no renderable geometry");
    }

    const VkDeviceSize vertSize = allVertices.size() * sizeof(Vertex);
    const VkDeviceSize idxSize = allIndices.size() * sizeof(uint32_t);

    m_vertexBuffer.createDeviceLocal(vertSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    m_indexBuffer.createDeviceLocal(idxSize, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    VkCommandPool transferPool = VK_NULL_HANDLE;
    VkCommandBuffer transferCmd = vkutil::beginSingleUseCommands(m_ctx, transferPool);

    m_vertexBuffer.uploadStaged(allVertices.data(), vertSize, transferCmd);
    m_indexBuffer.uploadStaged(allIndices.data(), idxSize, transferCmd);
    m_fallbackWhite.createSolidColor(255, 255, 255, 255, transferCmd, m_samplerCache);

    m_textures.reserve(model.textures.size());
    for (const auto& texEntry : model.textures) {
        m_textures.emplace_back(m_ctx);
        auto& tex = m_textures.back();

        if (texEntry.path.empty()) {
            spdlog::warn("Renderer resources: skipping texture with empty path (embedded/unsupported)");
            continue;
        }

        const VkFormat format = (texEntry.type == TextureType::Color)
            ? VK_FORMAT_R8G8B8A8_SRGB
            : VK_FORMAT_R8G8B8A8_UNORM;

        try {
            tex.loadFromFile(texEntry.path, format, transferCmd, m_samplerCache);
        } catch (const std::exception& e) {
            spdlog::error("Renderer resources: failed to load texture '{}': {}", texEntry.path, e.what());
        }
    }

    vkutil::endSingleUseCommands(m_ctx, transferPool, transferCmd);

    m_vertexBuffer.releaseStaging();
    m_indexBuffer.releaseStaging();
    m_fallbackWhite.releaseStaging();
    for (auto& tex : m_textures) {
        tex.releaseStaging();
    }

    spdlog::info("Renderer resources uploaded: {} meshes, {} vertices, {} indices",
                 model.meshes.size(), allVertices.size(), allIndices.size());
}

void RendererResourceManager::clear()
{
    m_fallbackWhite.destroy();
    for (auto& tex : m_textures) {
        tex.destroy();
    }
    m_textures.clear();
    m_meshes.clear();
    m_materials.clear();
    m_indexBuffer.destroy();
    m_vertexBuffer.destroy();
    m_samplerCache.shutdown();
}

size_t RendererResourceManager::estimateTextureMemoryBytes() const
{
    size_t total = 0;
    for (const auto& tex : m_textures) {
        if (!tex.isValid()) {
            continue;
        }
        const VkExtent3D ext = tex.getExtent();
        total += static_cast<size_t>(ext.width) *
                 static_cast<size_t>(ext.height) *
                 4u;
    }
    return total;
}
