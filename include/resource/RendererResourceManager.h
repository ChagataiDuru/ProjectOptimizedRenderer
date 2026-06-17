#pragma once

#include "core/MeshRenderData.h"
#include "core/VulkanContext.h"
#include "resource/Buffer.h"
#include "resource/Material.h"
#include "resource/Model.h"
#include "resource/SamplerCache.h"
#include "resource/Texture.h"

#include <cstddef>
#include <vector>

class RendererResourceManager {
public:
    explicit RendererResourceManager(VulkanContext& ctx);
    ~RendererResourceManager();

    RendererResourceManager(const RendererResourceManager&) = delete;
    RendererResourceManager& operator=(const RendererResourceManager&) = delete;

    void uploadModel(const Model& model);
    void clear();

    const Buffer& vertexBuffer() const { return m_vertexBuffer; }
    const Buffer& indexBuffer() const { return m_indexBuffer; }
    const std::vector<MeshRenderData>& meshes() const { return m_meshes; }
    const std::vector<Material>& materials() const { return m_materials; }
    const std::vector<Texture>& textures() const { return m_textures; }
    const Texture& fallbackWhite() const { return m_fallbackWhite; }

    uint32_t meshCount() const { return static_cast<uint32_t>(m_meshes.size()); }
    uint32_t materialCount() const { return static_cast<uint32_t>(m_materials.size()); }
    uint32_t textureCount() const { return static_cast<uint32_t>(m_textures.size()); }
    size_t estimateTextureMemoryBytes() const;

private:
    VulkanContext& m_ctx;

    Buffer m_vertexBuffer;
    Buffer m_indexBuffer;
    SamplerCache m_samplerCache;
    std::vector<Texture> m_textures;
    Texture m_fallbackWhite;
    std::vector<MeshRenderData> m_meshes;
    std::vector<Material> m_materials;
};
