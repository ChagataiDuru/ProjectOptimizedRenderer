#include "core/Renderer.h"
#include "core/VulkanUtil.h"
#include "debug/ImGuiManager.h"
#include "resource/GLTFLoader.h"
#include "resource/SceneInfo.h"
#include "resource/Vertex.h"

#include <spdlog/spdlog.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

#ifndef ASSET_DIR
#define ASSET_DIR "assets"
#endif

// ── Helpers ───────────────────────────────────────────────────────────────────

std::vector<uint32_t> Renderer::loadSpv(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Cannot open shader: " + path);

    const auto size = static_cast<size_t>(file.tellg());
    if (size == 0 || size % 4 != 0)
        throw std::runtime_error("Invalid SPIR-V (size not a multiple of 4): " + path);

    file.seekg(0);
    std::vector<uint32_t> code(size / 4);
    file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(size));
    return code;
}

static glm::vec3 safeNormalizeDirection(glm::vec3 v, glm::vec3 fallback)
{
    const float len2 = glm::dot(v, v);
    if (len2 <= 1e-8f) {
        return glm::normalize(fallback);
    }
    return v * glm::inversesqrt(len2);
}

static VkShaderModule makeShaderModule(VkDevice device, const std::vector<uint32_t>& code)
{
    const VkShaderModuleCreateInfo info{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size() * sizeof(uint32_t),
        .pCode    = code.data(),
    };
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(device, &info, nullptr, &mod));
    return mod;
}

static VkGraphicsPipelineCreateInfo makeGraphicsPipelineCreateInfo(
    const VkPipelineRenderingCreateInfo& renderingInfo,
    const VkPipelineShaderStageCreateInfo* stages,
    uint32_t stageCount,
    const VkPipelineVertexInputStateCreateInfo& vertexInput,
    const VkPipelineInputAssemblyStateCreateInfo& inputAssembly,
    const VkPipelineViewportStateCreateInfo& viewportState,
    const VkPipelineRasterizationStateCreateInfo& rasterization,
    const VkPipelineMultisampleStateCreateInfo& multisample,
    const VkPipelineDepthStencilStateCreateInfo& depthStencil,
    const VkPipelineColorBlendStateCreateInfo& colorBlend,
    const VkPipelineDynamicStateCreateInfo& dynamicState,
    VkPipelineLayout layout)
{
    return VkGraphicsPipelineCreateInfo{
        .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext               = &renderingInfo,
        .stageCount          = stageCount,
        .pStages             = stages,
        .pVertexInputState   = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState      = &viewportState,
        .pRasterizationState = &rasterization,
        .pMultisampleState   = &multisample,
        .pDepthStencilState  = &depthStencil,
        .pColorBlendState    = &colorBlend,
        .pDynamicState       = &dynamicState,
        .layout              = layout,
        .renderPass          = VK_NULL_HANDLE,
    };
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

Renderer::Renderer(VulkanContext& ctx, Swapchain& swapchain)
    : m_ctx(ctx)
    , m_swapchain(swapchain)
    , m_commandBuffer(ctx)
    , m_frameSync(ctx)
    , m_vertexBuffer(ctx)
    , m_indexBuffer(ctx)
    , m_cameraUBOBuffer(ctx)
    , m_lightUBOBuffer(ctx)
    , m_pointLightBuffer(ctx)
    , m_clusterGridBuffer(ctx)
    , m_clusterLightIndexBuffer(ctx)
    , m_clusterMetadataBuffer(ctx)
    , m_samplerCache(ctx)
    , m_fallbackWhite(ctx)
    , m_depthImage(ctx)
    , m_hdrTarget(ctx)
    , m_shadowPass(ctx)
    , m_clusteredLightCullingPass(ctx)
    , m_skyPass(ctx)
    , m_gpuTimer(ctx)
    , m_screenshot(ctx)
{
}

Renderer::~Renderer()
{
    shutdown();
}

void Renderer::init()
{
    loadModel(std::string(ASSET_DIR) + "/source/Sponza.gltf");
    m_sceneInfo = computeSceneInfo(m_model.boundsMin, m_model.boundsMax);
    spdlog::info("Scene normalized: scale={:.4f}, radius={:.2f}",
                 m_sceneInfo.scaleFactor, m_sceneInfo.normalizedRadius);
    createDefaultPointLights();
    rebuildDrawCommands();

    m_frameSync.init(3);
    m_commandBuffer.init(m_ctx.getGraphicsQueueFamily(), 3);
    m_tonemapPass.init(m_ctx.getDevice(), m_swapchain.getFormat(), SHADER_DIR);
    createResizeDependentResources();
    createPbrPipeline();
    m_clusteredLightCullingPass.init(m_cameraSetLayout, SHADER_DIR);
    m_skyPass.init(m_ctx, m_cameraSetLayout,
                   VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_D32_SFLOAT,
                   SHADER_DIR);
    createCameraUBO();
    createLightUBO();
    createClusteredLightResources();
    m_shadowPass.init(m_pipelineLayout, VK_NULL_HANDLE, SHADER_DIR);
    createDescriptorPool();
    createDescriptorSet();
    m_shadowPass.setSceneDescriptorSet(m_descriptorSet);
    createMaterialDescriptorSets();
    m_gpuTimer.init(24);

    // Populate static render stats (counts that don't change after load)
    refreshRenderStats();

    spdlog::info("Renderer initialized: {} meshes, {} materials, {} textures ({:.1f} MB GPU tex)",
                 m_renderStats.meshCount, m_renderStats.materialCount,
                 m_renderStats.textureCount,
                 m_renderStats.textureMemoryBytes / (1024.0f * 1024.0f));
}

void Renderer::reloadModel(const std::string& modelPath)
{
    spdlog::info("Reloading model: {}", modelPath);

    // ── Drain the GPU — no frame may reference resources we're about to destroy ──
    vkDeviceWaitIdle(m_ctx.getDevice());

    // ── Destroy model-dependent resources ────────────────────────────────────────
    // Descriptor pool destruction implicitly frees m_descriptorSet and all m_materialSets.
    if (m_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_ctx.getDevice(), m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
        m_descriptorSet  = VK_NULL_HANDLE;
    }
    m_materialSets.clear();

    for (auto& tex : m_textures)
        tex.destroy();
    m_textures.clear();

    // Fallback white is recreated by loadModel(); destroy old allocation first.
    m_fallbackWhite.destroy();

    m_indexBuffer.destroy();
    m_vertexBuffer.destroy();

    m_meshes.clear();
    m_materials.clear();
    m_drawCommands.clear();
    m_model = Model{};

    // ── Load new model and upload resources ──────────────────────────────────────
    try {
        loadModel(modelPath);
    } catch (const std::exception& e) {
        spdlog::error("Failed to load model '{}': {}", modelPath, e.what());
        spdlog::info("Falling back to Sponza");
        try {
            loadModel(std::string(ASSET_DIR) + "/source/Sponza.gltf");
        } catch (const std::exception& e2) {
            spdlog::critical("Failed to load fallback model: {}", e2.what());
            throw;  // Unrecoverable
        }
    }

    // ── Compute normalization transform for the new model ────────────────────────
    m_sceneInfo = computeSceneInfo(m_model.boundsMin, m_model.boundsMax);
    spdlog::info("Scene normalized: scale={:.4f}, radius={:.2f}",
                 m_sceneInfo.scaleFactor, m_sceneInfo.normalizedRadius);
    createDefaultPointLights();
    uploadPointLightBuffer();
    uploadClusterMetadata();
    rebuildDrawCommands();

    // ── Recreate descriptors for the new model ───────────────────────────────────
    createDescriptorPool();
    createDescriptorSet();
    m_shadowPass.setSceneDescriptorSet(m_descriptorSet);
    createMaterialDescriptorSets();

    // ── Update render stats ──────────────────────────────────────────────────────
    refreshRenderStats();

    spdlog::info("Model reloaded: {} meshes, {} materials, {} textures ({:.1f} MB GPU tex)",
                 m_renderStats.meshCount, m_renderStats.materialCount,
                 m_renderStats.textureCount,
                 m_renderStats.textureMemoryBytes / (1024.0f * 1024.0f));
}

void Renderer::shutdown()
{
    if (m_ctx.getDevice() == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(m_ctx.getDevice());

    m_gpuTimer.shutdown();

    destroyResizeDependentResources();
    m_tonemapPass.shutdown(m_ctx.getDevice());
    if (m_hdrSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_ctx.getDevice(), m_hdrSampler, nullptr);
        m_hdrSampler = VK_NULL_HANDLE;
    }

    m_skyPass.shutdown(m_ctx.getDevice());
    m_clusteredLightCullingPass.shutdown(m_ctx.getDevice());

    destroyPipeline();

    // Destroy textures before VMA teardown (textures hold VmaAllocations)
    m_fallbackWhite.destroy();
    for (auto& tex : m_textures)
        tex.destroy();
    m_textures.clear();
    m_meshes.clear();
    m_materials.clear();
    m_drawCommands.clear();
    m_samplerCache.shutdown();

    m_shadowPass.shutdown(m_ctx.getDevice());

    // Destroy GPU resources that hold VMA allocations
    destroyClusteredLightResources();
    m_lightUBOBuffer.destroy();
    m_cameraUBOBuffer.destroy();
    m_indexBuffer.destroy();
    m_vertexBuffer.destroy();

    m_commandBuffer.shutdown();
    m_frameSync.shutdown();
}

void Renderer::refreshRenderStats()
{
    m_renderStats.meshCount          = static_cast<uint32_t>(m_meshes.size());
    m_renderStats.materialCount      = static_cast<uint32_t>(m_materials.size());
    m_renderStats.textureCount       = static_cast<uint32_t>(m_textures.size());
    m_renderStats.textureMemoryBytes = 0;
    for (const auto& tex : m_textures) {
        if (tex.isValid()) {
            const VkExtent3D ext = tex.getExtent();
            // Each texel is 4 bytes (RGBA8); depth = 1 for 2D textures.
            m_renderStats.textureMemoryBytes += static_cast<size_t>(ext.width) * ext.height * 4;
        }
    }
}

// ── Geometry ──────────────────────────────────────────────────────────────────

void Renderer::loadModel(const std::string& modelPath)
{
    m_model = GLTFLoader::loadGLTF(modelPath);
    m_meshes.clear();
    m_materials = m_model.materials;
    m_drawCommands.clear();

    // Flatten all mesh vertex/index data into single contiguous arrays.
    // Each mesh's indices are already absolute (offset applied in GLTFLoader)
    // relative to that mesh's own vertex array — here we make them absolute
    // into the combined vertex array by adding the running vertexBase.
    std::vector<Vertex>   allVertices;
    std::vector<uint32_t> allIndices;
    uint32_t vertexBase = 0;

    for (const auto& mesh : m_model.meshes) {
        MeshRenderData data{};
        data.firstIndex    = static_cast<uint32_t>(allIndices.size());
        data.indexCount    = static_cast<uint32_t>(mesh.indices.size());
        m_meshes.push_back(data);

        allVertices.insert(allVertices.end(), mesh.vertices.begin(), mesh.vertices.end());

        // Re-base indices so they point into the combined vertex buffer.
        for (uint32_t idx : mesh.indices)
            allIndices.push_back(idx + vertexBase);

        vertexBase += static_cast<uint32_t>(mesh.vertices.size());
    }

    const VkDeviceSize vertSize = allVertices.size() * sizeof(Vertex);
    const VkDeviceSize idxSize  = allIndices.size()  * sizeof(uint32_t);

    m_vertexBuffer.createDeviceLocal(vertSize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    m_indexBuffer.createDeviceLocal(idxSize,   VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

    // Staged upload: use a dedicated transient pool so frame command buffers are not disturbed.
    VkCommandPool   transferPool = VK_NULL_HANDLE;
    VkCommandBuffer transferCmd  = vkutil::beginSingleUseCommands(m_ctx, transferPool);

    m_vertexBuffer.uploadStaged(allVertices.data(), vertSize, transferCmd);
    m_indexBuffer.uploadStaged(allIndices.data(),   idxSize,  transferCmd);

    // ── Texture upload ────────────────────────────────────────────────────────
    // All texture copy commands are recorded into the same transferCmd so they
    // share one submit/fence with the geometry buffers.

    m_fallbackWhite.createSolidColor(255, 255, 255, 255, transferCmd, m_samplerCache);

    m_textures.reserve(m_model.textures.size());
    for (const auto& texEntry : m_model.textures) {
        m_textures.emplace_back(m_ctx);
        auto& tex = m_textures.back();

        if (texEntry.path.empty()) {
            spdlog::warn("Renderer: skipping texture with empty path (embedded/unsupported)");
            continue;
        }

        const VkFormat format = (texEntry.type == TextureType::Color)
            ? VK_FORMAT_R8G8B8A8_SRGB
            : VK_FORMAT_R8G8B8A8_UNORM;

        try {
            tex.loadFromFile(texEntry.path, format, transferCmd, m_samplerCache);
        } catch (const std::exception& e) {
            spdlog::error("Renderer: failed to load texture '{}': {}", texEntry.path, e.what());
        }
    }

    vkutil::endSingleUseCommands(m_ctx, transferPool, transferCmd);

    m_vertexBuffer.releaseStaging();
    m_indexBuffer.releaseStaging();
    m_fallbackWhite.releaseStaging();
    for (auto& tex : m_textures)
        tex.releaseStaging();

    spdlog::info("Model uploaded: {} meshes, {} vertices, {} indices",
                 m_model.meshes.size(), allVertices.size(), allIndices.size());
}

void Renderer::rebuildDrawCommands()
{
    m_drawCommands.clear();
    m_drawCommands.reserve(m_model.meshes.size());

    const glm::mat4& transform = m_sceneInfo.modelMatrix;

    for (size_t i = 0; i < m_model.meshes.size() && i < m_meshes.size(); ++i) {
        const Mesh& srcMesh = m_model.meshes[i];
        MeshRenderData& dstMesh = m_meshes[i];

        const glm::vec3& lo = srcMesh.boundsMin;
        const glm::vec3& hi = srcMesh.boundsMax;
        glm::vec3 wMin(std::numeric_limits<float>::max());
        glm::vec3 wMax(-std::numeric_limits<float>::max());

        for (int j = 0; j < 8; ++j) {
            const glm::vec3 corner((j & 1) ? hi.x : lo.x,
                                   (j & 2) ? hi.y : lo.y,
                                   (j & 4) ? hi.z : lo.z);
            const glm::vec3 world = glm::vec3(transform * glm::vec4(corner, 1.0f));
            wMin = glm::min(wMin, world);
            wMax = glm::max(wMax, world);
        }

        dstMesh.worldBoundsMin = wMin;
        dstMesh.worldBoundsMax = wMax;

        MaterialHandle material{};
        if (srcMesh.materialIndex >= 0 &&
            srcMesh.materialIndex < static_cast<int32_t>(m_materials.size())) {
            material.index = static_cast<uint32_t>(srcMesh.materialIndex);
        }

        m_drawCommands.push_back(DrawCommand{
            .transform = transform,
            .mesh = MeshHandle{ static_cast<uint32_t>(i) },
            .material = material,
        });
    }
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

void Renderer::createPbrPipeline()
{
    const std::string dir = SHADER_DIR;
    m_vertModule        = makeShaderModule(m_ctx.getDevice(), loadSpv(dir + "/pbr.vert.spv"));
    m_fragModule        = makeShaderModule(m_ctx.getDevice(), loadSpv(dir + "/pbr.frag.spv"));
    m_normalsFragModule = makeShaderModule(m_ctx.getDevice(), loadSpv(dir + "/pbr_normals.frag.spv"));

    const std::array<VkPipelineShaderStageCreateInfo, 2> stages{{
        {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_VERTEX_BIT,
            .module = m_vertModule,
            .pName  = "main",
        },
        {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = m_fragModule,
            .pName  = "main",
        },
    }};

    const std::array<VkPipelineShaderStageCreateInfo, 2> normalsStages{{
        {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_VERTEX_BIT,
            .module = m_vertModule,
            .pName  = "main",
        },
        {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = m_normalsFragModule,
            .pName  = "main",
        },
    }};

    // Vertex buffer layout from Vertex struct (position, normal, uv, tangent)
    const auto bindingDesc  = Vertex::getBindingDescription();
    const auto attribDescs  = Vertex::getAttributeDescriptions();
    const VkPipelineVertexInputStateCreateInfo vertexInput{
        .sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount   = 1,
        .pVertexBindingDescriptions      = &bindingDesc,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attribDescs.size()),
        .pVertexAttributeDescriptions    = attribDescs.data(),
    };

    const VkPipelineInputAssemblyStateCreateInfo inputAssembly{
        .sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    // Viewport and scissor are dynamic — avoids recreating the pipeline on swapchain resize
    const VkPipelineViewportStateCreateInfo viewportState{
        .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount  = 1,
    };

    VkPipelineRasterizationStateCreateInfo rasterization{
        .sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        // Back-face culling enabled: cube normals are outward-facing CCW, so back faces are CW.
        .cullMode    = VK_CULL_MODE_NONE,
        .frontFace   = VK_FRONT_FACE_CLOCKWISE,
        .lineWidth   = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo multisample{
        .sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };

    const VkPipelineDepthStencilStateCreateInfo depthStencil{
        .sType             = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable   = VK_TRUE,
        .depthWriteEnable  = VK_TRUE,
        // Reverse-Z: a closer fragment has a LARGER depth value, so it passes if greater
        .depthCompareOp    = VK_COMPARE_OP_GREATER,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
    };

    const VkPipelineColorBlendAttachmentState blendAttachment{
        .blendEnable    = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };

    const VkPipelineColorBlendStateCreateInfo colorBlend{
        .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments    = &blendAttachment,
    };

    const std::array<VkDynamicState, 2> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
    };
    const VkPipelineDynamicStateCreateInfo dynamicState{
        .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates    = dynamicStates.data(),
    };

    // Descriptor set layout: set 0 — camera(b0), light(b1), shadowUBO(b2), depth array(b3), moments(b4).
    const std::array<VkDescriptorSetLayoutBinding, 9> setBindings{{
        {   // binding 0: camera matrices — read in vertex + fragment shaders
            .binding         = shader_interface::scene_binding::kCamera,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_VERTEX_BIT |
                               VK_SHADER_STAGE_FRAGMENT_BIT |
                               VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {   // binding 1: directional light — read in fragment shader
            .binding         = shader_interface::scene_binding::kLight,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {   // binding 2: shadow UBO (lightViewProj[4] + splitDepths) — vertex + fragment
            .binding         = shader_interface::scene_binding::kShadow,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {   // binding 3: shadow depth array (D32_SFLOAT, None/PCF modes) — fragment only
            .binding         = shader_interface::scene_binding::kShadowMap,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {   // binding 4: shadow moments array (RG32_SFLOAT, VSM mode) — fragment only
            .binding         = shader_interface::scene_binding::kShadowMoments,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding         = shader_interface::scene_binding::kPointLights,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding         = shader_interface::scene_binding::kClusterGrid,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding         = shader_interface::scene_binding::kClusterLightIndices,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding         = shader_interface::scene_binding::kClusterMetadata,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT,
        },
    }};
    const VkDescriptorSetLayoutCreateInfo setLayoutInfo{
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(setBindings.size()),
        .pBindings    = setBindings.data(),
    };
    VK_CHECK(vkCreateDescriptorSetLayout(m_ctx.getDevice(), &setLayoutInfo, nullptr, &m_cameraSetLayout));

    // ── Material texture set layout (set=1) ──────────────────────────────────
    // 3 combined image samplers: albedo (b0), normal map (b1), metallic-roughness (b2).
    // All read by the fragment shader only.
    const std::array<VkDescriptorSetLayoutBinding, 3> materialBindings{{
        {
            .binding         = shader_interface::material_binding::kAlbedo,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding         = shader_interface::material_binding::kNormal,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding         = shader_interface::material_binding::kMetallicRoughness,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
    }};
    const VkDescriptorSetLayoutCreateInfo materialLayoutInfo{
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(materialBindings.size()),
        .pBindings    = materialBindings.data(),
    };
    VK_CHECK(vkCreateDescriptorSetLayout(m_ctx.getDevice(), &materialLayoutInfo, nullptr, &m_materialSetLayout));

    // Pipeline layout: set=0 (camera+light UBOs), set=1 (material textures)
    const std::array<VkDescriptorSetLayout, 2> allSetLayouts = {
        m_cameraSetLayout,    // set 0
        m_materialSetLayout,  // set 1
    };

    // Shadow and PBR shaders share one physical push-constant block:
    //   bytes  0–63 : model matrix
    //   bytes 64–95 : material factors
    //   bytes 96–99 : cascade index
    // Keep this as one vertex+fragment range so separate push calls for model,
    // material, and cascade index stay validation-clean on all drivers.
    const VkPushConstantRange pushRange{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset     = shader_interface::kModelMatrixPcOffset,
        .size       = shader_interface::kCascadeIndexPcOffset + sizeof(uint32_t),
    };

    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = static_cast<uint32_t>(allSetLayouts.size()),
        .pSetLayouts            = allSetLayouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushRange,
    };
    VK_CHECK(vkCreatePipelineLayout(m_ctx.getDevice(), &layoutInfo, nullptr, &m_pipelineLayout));

    // VkPipelineRenderingCreateInfo replaces VkRenderPass for dynamic rendering (core in 1.3/1.4)
    const VkFormat colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    const VkPipelineRenderingCreateInfo renderingInfo{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &colorFormat,
        .depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT,
        .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
    };

    // ── Create both solid and wireframe pipelines in one batch call ──────────
    // The wireframe pipeline is identical except for polygonMode = LINE.
    VkPipelineRasterizationStateCreateInfo wireframeRasterization = rasterization;
    wireframeRasterization.polygonMode = VK_POLYGON_MODE_LINE;

    const VkGraphicsPipelineCreateInfo pipelineInfo = makeGraphicsPipelineCreateInfo(
        renderingInfo,
        stages.data(),
        static_cast<uint32_t>(stages.size()),
        vertexInput,
        inputAssembly,
        viewportState,
        rasterization,
        multisample,
        depthStencil,
        colorBlend,
        dynamicState,
        m_pipelineLayout);

    VkGraphicsPipelineCreateInfo wireframePipelineInfo = pipelineInfo;
    wireframePipelineInfo.pRasterizationState = &wireframeRasterization;

    // Normals pipeline: same as solid but with the normals-only fragment shader
    VkGraphicsPipelineCreateInfo normalsPipelineInfo = pipelineInfo;
    normalsPipelineInfo.pStages    = normalsStages.data();
    normalsPipelineInfo.stageCount = static_cast<uint32_t>(normalsStages.size());

    const std::array<VkGraphicsPipelineCreateInfo, 3> pipelineInfos = {
        pipelineInfo,
        wireframePipelineInfo,
        normalsPipelineInfo,
    };

    std::array<VkPipeline, 3> pipelines{};
    VK_CHECK(vkCreateGraphicsPipelines(
        m_ctx.getDevice(), VK_NULL_HANDLE,
        static_cast<uint32_t>(pipelineInfos.size()),
        pipelineInfos.data(), nullptr, pipelines.data()));

    m_pipeline          = pipelines[0];
    m_wireframePipeline = pipelines[1];
    m_normalsPipeline   = pipelines[2];

    // Shader modules are only needed during pipeline compilation — free them immediately
    vkDestroyShaderModule(m_ctx.getDevice(), m_vertModule, nullptr);         m_vertModule        = VK_NULL_HANDLE;
    vkDestroyShaderModule(m_ctx.getDevice(), m_fragModule, nullptr);         m_fragModule        = VK_NULL_HANDLE;
    vkDestroyShaderModule(m_ctx.getDevice(), m_normalsFragModule, nullptr);  m_normalsFragModule = VK_NULL_HANDLE;

    spdlog::info("PBR pipelines created: solid + wireframe + normals");
}

void Renderer::destroyPipeline()
{
    const VkDevice dev = m_ctx.getDevice();
    // Descriptor pool destruction implicitly frees all sets allocated from it.
    // Pool destruction implicitly frees all allocated descriptor sets.
    if (m_descriptorPool     != VK_NULL_HANDLE) { vkDestroyDescriptorPool(dev, m_descriptorPool, nullptr);         m_descriptorPool    = VK_NULL_HANDLE; m_descriptorSet = VK_NULL_HANDLE; }
    m_materialSets.clear();
    if (m_vertModule         != VK_NULL_HANDLE) { vkDestroyShaderModule(dev, m_vertModule, nullptr);               m_vertModule        = VK_NULL_HANDLE; }
    if (m_fragModule         != VK_NULL_HANDLE) { vkDestroyShaderModule(dev, m_fragModule, nullptr);               m_fragModule        = VK_NULL_HANDLE; }
    if (m_normalsFragModule  != VK_NULL_HANDLE) { vkDestroyShaderModule(dev, m_normalsFragModule, nullptr);        m_normalsFragModule = VK_NULL_HANDLE; }
    if (m_pipeline           != VK_NULL_HANDLE) { vkDestroyPipeline(dev, m_pipeline, nullptr);                     m_pipeline          = VK_NULL_HANDLE; }
    if (m_wireframePipeline  != VK_NULL_HANDLE) { vkDestroyPipeline(dev, m_wireframePipeline, nullptr);            m_wireframePipeline = VK_NULL_HANDLE; }
    if (m_normalsPipeline    != VK_NULL_HANDLE) { vkDestroyPipeline(dev, m_normalsPipeline, nullptr);              m_normalsPipeline   = VK_NULL_HANDLE; }
    if (m_pipelineLayout     != VK_NULL_HANDLE) { vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);         m_pipelineLayout    = VK_NULL_HANDLE; }
    if (m_materialSetLayout  != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(dev, m_materialSetLayout, nullptr); m_materialSetLayout = VK_NULL_HANDLE; }
    if (m_cameraSetLayout    != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(dev, m_cameraSetLayout, nullptr);   m_cameraSetLayout   = VK_NULL_HANDLE; }
}

// ── Camera UBO ────────────────────────────────────────────────────────────────

void Renderer::createCameraUBO()
{
    // Host-visible + persistently mapped: upload is a memcpy, no staging needed.
    // UBOs change every frame so device-local + staging would be needlessly expensive.
    m_cameraUBOBuffer.createHostVisible(sizeof(CameraUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    const CameraUBO initial{ glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::vec3(0.0f), 0.0f };
    m_cameraUBOBuffer.upload(&initial, sizeof(CameraUBO));

    spdlog::info("Camera UBO created ({} bytes, host-visible)", sizeof(CameraUBO));
}

void Renderer::createLightUBO()
{
    m_lightUBOBuffer.createHostVisible(sizeof(LightUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    // Default sun: slightly off-vertical, warm white, moderate ambient
    setLightParameters(
        glm::normalize(glm::vec3(1.0f, 1.0f, 1.0f)), // Light direction (toward camera)
        glm::vec3(1.0f, 1.0f, 1.0f),                 // White light
        3.5f,                                        // Increase intensity
        0.3f                                         // Ambient light
    );

    spdlog::info("Light UBO created ({} bytes, host-visible)", sizeof(LightUBO));
}

void Renderer::setLightParameters(const glm::vec3& direction, const glm::vec3& color,
                                   float intensity, float ambient)
{
    m_lightDirection    = glm::normalize(direction);
    m_lightColor        = color;
    m_lightIntensity    = intensity;
    m_ambientIntensity  = ambient;
    uploadLightUBO();
    spdlog::debug("Light updated: dir=({:.2f},{:.2f},{:.2f}), intensity={:.2f}",
                  direction.x, direction.y, direction.z, intensity);
}

void Renderer::uploadLightUBO()
{
    const LightUBO light{
        m_lightDirection,
        m_lightIntensity,
        m_lightColor,
        m_ambientIntensity,
        m_shadowSettings.debugCascades ? 1u : 0u,
        static_cast<uint32_t>(m_shadowSettings.filterMode),
        m_shadowSettings.pcfSpreadRadius,
        m_shadowSettings.vsmBleedReduction,
    };
    m_lightUBOBuffer.upload(&light, sizeof(LightUBO));
}

void Renderer::createDefaultPointLights()
{
    const float r = std::max(m_sceneInfo.normalizedRadius, 1.0f);
    const float y = 1.8f;
    const float radius = r * 0.45f;
    const float intensity = 22.0f;

    m_pointLights = {
        { glm::vec4(-0.45f * r, y, -0.35f * r, radius), glm::vec4(1.0f, 0.45f, 0.30f, intensity) },
        { glm::vec4( 0.45f * r, y, -0.35f * r, radius), glm::vec4(0.35f, 0.65f, 1.0f, intensity) },
        { glm::vec4(-0.45f * r, y,  0.35f * r, radius), glm::vec4(0.45f, 1.0f, 0.55f, intensity) },
        { glm::vec4( 0.45f * r, y,  0.35f * r, radius), glm::vec4(1.0f, 0.80f, 0.35f, intensity) },
    };
}

void Renderer::createClusteredLightResources()
{
    m_pointLightBuffer.createHostVisible(
        static_cast<VkDeviceSize>(m_maxPointLights) * sizeof(shader_interface::GpuPointLight),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_clusterMetadataBuffer.createHostVisible(sizeof(ClusterMetadataUBO),
                                              VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    uploadPointLightBuffer();
    resizeClusteredLightResources();

    spdlog::info("Clustered Forward+ light resources created: max point lights={}, max lights/cluster={}",
                 m_maxPointLights, shader_interface::kMaxLightsPerCluster);
}

void Renderer::destroyClusteredLightResources()
{
    m_clusterLightIndexBuffer.destroy();
    m_clusterGridBuffer.destroy();
    m_clusterMetadataBuffer.destroy();
    m_pointLightBuffer.destroy();
}

void Renderer::resizeClusteredLightResources()
{
    const VkExtent2D ext = m_swapchain.getExtent();
    if (ext.width == 0 || ext.height == 0) {
        return;
    }

    m_clusterCountX = (ext.width + shader_interface::kClusterTileSizeX - 1u) /
                      shader_interface::kClusterTileSizeX;
    m_clusterCountY = (ext.height + shader_interface::kClusterTileSizeY - 1u) /
                      shader_interface::kClusterTileSizeY;
    m_clusterCountZ = shader_interface::kClusterZSlices;
    m_clusterCount = m_clusterCountX * m_clusterCountY * m_clusterCountZ;

    m_clusterGridBuffer.destroy();
    m_clusterLightIndexBuffer.destroy();

    m_clusterGridBuffer.createDeviceLocal(
        static_cast<VkDeviceSize>(m_clusterCount) * sizeof(shader_interface::ClusterLightRange),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    m_clusterLightIndexBuffer.createDeviceLocal(
        static_cast<VkDeviceSize>(m_clusterCount) * shader_interface::kMaxLightsPerCluster * sizeof(uint32_t),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

    uploadClusterMetadata();
    updateClusterDescriptors();

    spdlog::info("Cluster grid resized: {}x{}x{} ({} clusters)",
                 m_clusterCountX, m_clusterCountY, m_clusterCountZ, m_clusterCount);
}

void Renderer::setPointLights(const std::vector<shader_interface::GpuPointLight>& pointLights)
{
    m_pointLights.assign(pointLights.begin(),
                         pointLights.begin() + std::min(pointLights.size(),
                                                        static_cast<size_t>(m_maxPointLights)));
    uploadPointLightBuffer();
    uploadClusterMetadata();
}

void Renderer::uploadPointLightBuffer()
{
    if (m_pointLightBuffer.getBuffer() == VK_NULL_HANDLE) return;

    std::vector<shader_interface::GpuPointLight> uploadData(m_maxPointLights);
    const size_t copyCount = std::min(m_pointLights.size(), uploadData.size());
    std::copy_n(m_pointLights.data(), copyCount, uploadData.data());
    m_pointLightBuffer.upload(uploadData.data(),
                              static_cast<VkDeviceSize>(uploadData.size() * sizeof(uploadData[0])));
}

void Renderer::uploadClusterMetadata()
{
    if (m_clusterMetadataBuffer.getBuffer() == VK_NULL_HANDLE) return;

    const VkExtent2D ext = m_swapchain.getExtent();
    const ClusterMetadataUBO metadata{
        glm::uvec4(m_clusterCountX, m_clusterCountY, m_clusterCountZ,
                   shader_interface::kMaxLightsPerCluster),
        glm::uvec4(static_cast<uint32_t>(std::min(m_pointLights.size(),
                                                  static_cast<size_t>(m_maxPointLights))),
                   0u, 0u, 0u),
        glm::vec4(static_cast<float>(ext.width),
                  static_cast<float>(ext.height),
                  m_cameraNearZ,
                  m_cameraFarZ),
    };
    m_clusterMetadataBuffer.upload(&metadata, sizeof(metadata));
}

void Renderer::updateClusterDescriptors()
{
    if (m_descriptorSet == VK_NULL_HANDLE ||
        m_pointLightBuffer.getBuffer() == VK_NULL_HANDLE ||
        m_clusterGridBuffer.getBuffer() == VK_NULL_HANDLE ||
        m_clusterLightIndexBuffer.getBuffer() == VK_NULL_HANDLE ||
        m_clusterMetadataBuffer.getBuffer() == VK_NULL_HANDLE) {
        return;
    }

    const VkDescriptorBufferInfo pointLightInfo{
        .buffer = m_pointLightBuffer.getBuffer(),
        .offset = 0,
        .range = m_pointLightBuffer.getSize(),
    };
    const VkDescriptorBufferInfo clusterGridInfo{
        .buffer = m_clusterGridBuffer.getBuffer(),
        .offset = 0,
        .range = m_clusterGridBuffer.getSize(),
    };
    const VkDescriptorBufferInfo clusterIndicesInfo{
        .buffer = m_clusterLightIndexBuffer.getBuffer(),
        .offset = 0,
        .range = m_clusterLightIndexBuffer.getSize(),
    };
    const VkDescriptorBufferInfo clusterMetadataInfo{
        .buffer = m_clusterMetadataBuffer.getBuffer(),
        .offset = 0,
        .range = sizeof(ClusterMetadataUBO),
    };

    const std::array<VkWriteDescriptorSet, 4> writes{{
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = m_descriptorSet,
          .dstBinding = shader_interface::scene_binding::kPointLights,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &pointLightInfo },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = m_descriptorSet,
          .dstBinding = shader_interface::scene_binding::kClusterGrid,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &clusterGridInfo },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = m_descriptorSet,
          .dstBinding = shader_interface::scene_binding::kClusterLightIndices,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          .pBufferInfo = &clusterIndicesInfo },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = m_descriptorSet,
          .dstBinding = shader_interface::scene_binding::kClusterMetadata,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          .pBufferInfo = &clusterMetadataInfo },
    }};
    vkUpdateDescriptorSets(m_ctx.getDevice(), static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
}

void Renderer::setCascadeDebugEnabled(bool enabled)
{
    m_shadowSettings.debugCascades = enabled;
    uploadLightUBO();
}

void Renderer::setShadowFilterMode(int32_t mode)
{
    m_shadowSettings.filterMode = mode;
    uploadLightUBO();
}

void Renderer::setPcfSpreadRadius(float radius)
{
    m_shadowSettings.pcfSpreadRadius = radius;
    uploadLightUBO();
}

void Renderer::setVsmBleedReduction(float reduction)
{
    m_shadowSettings.vsmBleedReduction = reduction;
    uploadLightUBO();
}

void Renderer::createDepthImage()
{
    const VkExtent2D ext = m_swapchain.getExtent();
    // D32_SFLOAT: 32-bit float depth, no stencil, optimal for reverse-Z precision
    m_depthImage.create(ext.width, ext.height, 1,
                        VK_FORMAT_D32_SFLOAT,
                        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                        VK_IMAGE_ASPECT_DEPTH_BIT);

    spdlog::info("Depth image created: {}x{} D32_SFLOAT (reverse-Z)", ext.width, ext.height);
}

void Renderer::createResizeDependentResources()
{
    const VkExtent2D ext = m_swapchain.getExtent();
    if (ext.width == 0 || ext.height == 0) {
        return;
    }

    createDepthImage();
    createHdrTarget();
    m_tonemapPass.resize(m_ctx.getDevice(), m_hdrSampler, m_hdrTarget.getImageView());
    if (m_clusterMetadataBuffer.getBuffer() != VK_NULL_HANDLE) {
        resizeClusteredLightResources();
    }
}

void Renderer::destroyResizeDependentResources()
{
    m_clusterLightIndexBuffer.destroy();
    m_clusterGridBuffer.destroy();
    m_clusterCountX = 0;
    m_clusterCountY = 0;
    m_clusterCount = 0;
    m_hdrTarget.destroy();
    m_depthImage.destroy();
}

void Renderer::recreateResizeDependentResources()
{
    destroyResizeDependentResources();
    createResizeDependentResources();
}

void Renderer::createHdrTarget()
{
    const VkExtent2D ext = m_swapchain.getExtent();

    m_hdrTarget.create(ext.width, ext.height, 1,
                       VK_FORMAT_R16G16B16A16_SFLOAT,
                       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                       VK_IMAGE_ASPECT_COLOR_BIT);

    // Create the sampler once (on first call); reuse on resize — the sampler itself
    // doesn't reference the image view, so no need to destroy+recreate it.
    if (m_hdrSampler == VK_NULL_HANDLE) {
        const VkSamplerCreateInfo samplerCI{
            .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter    = VK_FILTER_LINEAR,
            .minFilter    = VK_FILTER_LINEAR,
            .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .maxLod       = VK_LOD_CLAMP_NONE,
        };
        VK_CHECK(vkCreateSampler(m_ctx.getDevice(), &samplerCI, nullptr, &m_hdrSampler));
    }

    spdlog::info("HDR target created: {}x{} R16G16B16A16_SFLOAT", ext.width, ext.height);
}

void Renderer::loadHdrPanorama(const std::string& path)
{
    m_skyPass.loadHdrPanorama(m_ctx, path);
}
void Renderer::createDescriptorPool()
{
    // Pool must hold:
    //   1 set  × 3 uniform buffers (camera + light + shadow, set=0)
    //   1 set  × 1 combined image sampler (shadow map, set=0)
    //   N sets × 3 combined image samplers each (material textures, set=1)
    const uint32_t materialCount = static_cast<uint32_t>(m_materials.size());
    const uint32_t maxSets       = 1 + std::max(materialCount, 1u);
    // 2 samplers in set=0 (depth array b3 + moments array b4) + 3 per material
    const uint32_t samplerCount  = 2 + 3 * std::max(materialCount, 1u);

    const std::array<VkDescriptorPoolSize, 3> poolSizes{{
        {
            .type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 4,  // camera + light + shadow + cluster metadata
        },
        {
            .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = samplerCount,
        },
        {
            .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 3,  // point lights + cluster grid + cluster indices
        },
    }};

    const VkDescriptorPoolCreateInfo poolCI{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = maxSets,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes    = poolSizes.data(),
    };
    VK_CHECK(vkCreateDescriptorPool(m_ctx.getDevice(), &poolCI, nullptr, &m_descriptorPool));

    spdlog::info("Descriptor pool created: {} max sets (4 UBO, 3 SSBO, {} sampler descriptors)",
                 maxSets, samplerCount);
}

void Renderer::createDescriptorSet()
{
    // m_cameraSetLayout was already created in createPbrPipeline() and
    // baked into the pipeline layout — reuse it directly (now holds bindings 0 + 1).
    const VkDescriptorSetAllocateInfo allocInfo{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = m_descriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &m_cameraSetLayout,
    };
    VK_CHECK(vkAllocateDescriptorSets(m_ctx.getDevice(), &allocInfo, &m_descriptorSet));

    const VkDescriptorBufferInfo cameraInfo{
        .buffer = m_cameraUBOBuffer.getBuffer(),
        .offset = 0,
        .range  = sizeof(CameraUBO),
    };
    const VkDescriptorBufferInfo lightInfo{
        .buffer = m_lightUBOBuffer.getBuffer(),
        .offset = 0,
        .range  = sizeof(LightUBO),
    };
    const VkDescriptorBufferInfo shadowUBOInfo{
        .buffer = m_shadowPass.getShadowUBOBuffer(),
        .offset = 0,
        .range  = sizeof(ShadowCascadeUBO),
    };
    const VkDescriptorImageInfo shadowMapInfo{
        .sampler     = m_shadowPass.getDepthSampler(),
        .imageView   = m_shadowPass.getDepthImageView(),
        .imageLayout = m_shadowPass.getDepthSampleLayout(),
    };
    const VkDescriptorImageInfo momentsInfo{
        .sampler     = m_shadowPass.getMomentsSampler(),
        .imageView   = m_shadowPass.getMomentsImageView(),
        .imageLayout = m_shadowPass.getMomentsSampleLayout(),
    };

    const std::array<VkWriteDescriptorSet, 5> writes{{
        {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = m_descriptorSet,
            .dstBinding      = shader_interface::scene_binding::kCamera,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo     = &cameraInfo,
        },
        {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = m_descriptorSet,
            .dstBinding      = shader_interface::scene_binding::kLight,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo     = &lightInfo,
        },
        {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = m_descriptorSet,
            .dstBinding      = shader_interface::scene_binding::kShadow,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo     = &shadowUBOInfo,
        },
        {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = m_descriptorSet,
            .dstBinding      = shader_interface::scene_binding::kShadowMap,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo      = &shadowMapInfo,
        },
        {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = m_descriptorSet,
            .dstBinding      = shader_interface::scene_binding::kShadowMoments,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo      = &momentsInfo,
        },
    }};
    vkUpdateDescriptorSets(m_ctx.getDevice(),
                           static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    updateClusterDescriptors();

    spdlog::info("Descriptor set allocated: camera/light/shadow + clustered light buffers");
}

void Renderer::updateShadowDescriptors()
{
    if (m_descriptorSet == VK_NULL_HANDLE) return;

    const VkDescriptorBufferInfo shadowUBOInfo{
        .buffer = m_shadowPass.getShadowUBOBuffer(),
        .offset = 0,
        .range = sizeof(ShadowCascadeUBO),
    };
    const VkDescriptorImageInfo shadowMapInfo{
        .sampler = m_shadowPass.getDepthSampler(),
        .imageView = m_shadowPass.getDepthImageView(),
        .imageLayout = m_shadowPass.getDepthSampleLayout(),
    };
    const VkDescriptorImageInfo momentsInfo{
        .sampler = m_shadowPass.getMomentsSampler(),
        .imageView = m_shadowPass.getMomentsImageView(),
        .imageLayout = m_shadowPass.getMomentsSampleLayout(),
    };

    const std::array<VkWriteDescriptorSet, 3> writes{{
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_descriptorSet,
            .dstBinding = shader_interface::scene_binding::kShadow,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &shadowUBOInfo,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_descriptorSet,
            .dstBinding = shader_interface::scene_binding::kShadowMap,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &shadowMapInfo,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = m_descriptorSet,
            .dstBinding = shader_interface::scene_binding::kShadowMoments,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &momentsInfo,
        },
    }};
    vkUpdateDescriptorSets(m_ctx.getDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void Renderer::createMaterialDescriptorSets()
{
    const uint32_t materialCount = static_cast<uint32_t>(m_materials.size());
    if (materialCount == 0) return;

    // Allocate all material sets in one batch — more efficient than N individual calls.
    std::vector<VkDescriptorSetLayout> layouts(materialCount, m_materialSetLayout);
    m_materialSets.resize(materialCount);

    const VkDescriptorSetAllocateInfo allocInfo{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = m_descriptorPool,
        .descriptorSetCount = materialCount,
        .pSetLayouts        = layouts.data(),
    };
    VK_CHECK(vkAllocateDescriptorSets(m_ctx.getDevice(), &allocInfo, m_materialSets.data()));

    // Helpers: resolve a texture index to its view/sampler, falling back to white.
    auto getView = [&](int32_t idx) -> VkImageView {
        if (idx >= 0 && idx < static_cast<int32_t>(m_textures.size())
            && m_textures[static_cast<size_t>(idx)].isValid())
            return m_textures[static_cast<size_t>(idx)].getImageView();
        return m_fallbackWhite.getImageView();
    };
    auto getSampler = [&](int32_t idx) -> VkSampler {
        if (idx >= 0 && idx < static_cast<int32_t>(m_textures.size())
            && m_textures[static_cast<size_t>(idx)].isValid())
            return m_textures[static_cast<size_t>(idx)].getSampler();
        return m_fallbackWhite.getSampler();
    };

    for (uint32_t i = 0; i < materialCount; ++i) {
        const Material& mat = m_materials[i];

        const VkDescriptorImageInfo albedoInfo{
            .sampler     = getSampler(mat.albedoTextureIndex),
            .imageView   = getView(mat.albedoTextureIndex),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkDescriptorImageInfo normalInfo{
            .sampler     = getSampler(mat.normalTextureIndex),
            .imageView   = getView(mat.normalTextureIndex),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };
        const VkDescriptorImageInfo mrInfo{
            .sampler     = getSampler(mat.metallicRoughnessTextureIndex),
            .imageView   = getView(mat.metallicRoughnessTextureIndex),
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        };

        const std::array<VkWriteDescriptorSet, 3> writes{{
            {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = m_materialSets[i],
                .dstBinding      = shader_interface::material_binding::kAlbedo,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo      = &albedoInfo,
            },
            {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = m_materialSets[i],
                .dstBinding      = shader_interface::material_binding::kNormal,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo      = &normalInfo,
            },
            {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = m_materialSets[i],
                .dstBinding      = shader_interface::material_binding::kMetallicRoughness,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo      = &mrInfo,
            },
        }};
        vkUpdateDescriptorSets(m_ctx.getDevice(),
                               static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    spdlog::info("Allocated {} material descriptor sets", materialCount);
}


void Renderer::handleResize()
{
    const VkExtent2D oldExt = m_swapchain.getExtent();

    vkDeviceWaitIdle(m_ctx.getDevice());
    destroyResizeDependentResources();

    // Recreate swapchain. Passing current extent as hint; Swapchain::createSwapchain()
    // reads surface capabilities and uses cap.currentExtent when available (most platforms),
    // falling back to clamped requested dimensions otherwise.
    m_swapchain.recreate(oldExt.width, oldExt.height);

    const VkExtent2D newExt = m_swapchain.getExtent();
    if (newExt.width > 0 && newExt.height > 0) {
        createResizeDependentResources();
    }

    spdlog::info("Renderer resized: {}x{} -> {}x{}", oldExt.width, oldExt.height,
                 newExt.width, newExt.height);
}

bool Renderer::beginFrame()
{
    m_frameSync.waitForFrame();
    // Fence has signaled — previous frame's GPU work is complete; safe to read timestamps.
    m_gpuTimer.collectResults();

    // NOTE: fence is NOT reset here. It is reset in endFrame() immediately before submit.
    // This prevents a deadlock if beginFrame() returns false (skip frame): the fence
    // remains signaled, so the next waitForFrame() returns immediately.

    m_commandBuffer.resetFrame();

    if (!m_swapchain.acquireNextImage(m_frameSync.getImageAvailableSemaphore())) {
        // Swapchain out of date — recreate and retry.
        // The semaphore was NOT signaled by the failed acquire, so it's safe to reuse.
        handleResize();

        // After resize, check for zero-size extent (window minimized).
        const VkExtent2D ext = m_swapchain.getExtent();
        if (ext.width == 0 || ext.height == 0)
            return false;

        // Retry acquire with the new swapchain.
        if (!m_swapchain.acquireNextImage(m_frameSync.getImageAvailableSemaphore()))
            return false;
    }

    return true;
}

void Renderer::render()
{
    VkCommandBuffer cmd = m_commandBuffer.getFrameCommandBuffer();

    const VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        // ONE_TIME_SUBMIT: buffer is re-recorded every frame after reset
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    // Reset per-frame query slots and GPU timer name map
    m_gpuTimer.beginFrame(cmd);

    // Reset per-frame draw statistics
    m_renderStats.drawCalls = 0;
    m_renderStats.triangles = 0;

    // update shadow matrices each frame so the shadow follows the camera.
    // Also recalculates shadowRadius/shadowDepth from m_sceneInfo.normalizedRadius.
    m_shadowPass.record(cmd, m_vertexBuffer, m_indexBuffer, m_drawCommands,
                        m_meshes, m_materials, m_materialSets, m_gpuTimer);
    if (m_shadowPass.consumeDescriptorDirty()) {
        updateShadowDescriptors();
    }

    m_gpuTimer.writeTimestamp(cmd, "ClusterCull_Begin");
    m_clusteredLightCullingPass.record(cmd, m_descriptorSet,
                                       m_clusterCountX, m_clusterCountY, m_clusterCountZ);

    const std::array<VkBufferMemoryBarrier2, 2> clusterBarriers{{
        {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = m_clusterGridBuffer.getBuffer(),
            .offset = 0,
            .size = m_clusterGridBuffer.getSize(),
        },
        {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = m_clusterLightIndexBuffer.getBuffer(),
            .offset = 0,
            .size = m_clusterLightIndexBuffer.getSize(),
        },
    }};
    const VkDependencyInfo clusterDep{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = static_cast<uint32_t>(clusterBarriers.size()),
        .pBufferMemoryBarriers = clusterBarriers.data(),
    };
    vkCmdPipelineBarrier2(cmd, &clusterDep);
    m_gpuTimer.writeTimestamp(cmd, "ClusterCull_End");

    vkutil::transitionImage(cmd, m_hdrTarget.getImage(),
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,              0,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Depth image transitions from UNDEFINED each frame — LOAD_OP_CLEAR discards previous
    // contents anyway, so UNDEFINED→DEPTH_ATTACHMENT_OPTIMAL is valid and avoids layout tracking.
    vkutil::transitionImage(cmd, m_depthImage.getImage(),
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,              0,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    // ── Scene pass → HDR offscreen target ────────────────────────────────────
    // Clear to black in linear space (sRGB 0.01 ≈ linear ~0.001; using 0 is visually identical).
    const VkRenderingAttachmentInfo colorAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = m_hdrTarget.getImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } },
    };

    // Reverse-Z: clear depth to 0.0 (far plane); closer fragments have larger depth values
    const VkRenderingAttachmentInfo depthAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = m_depthImage.getImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_DONT_CARE,  // not sampled after, no need to store
        .clearValue  = { .depthStencil = { 0.0f, 0 } },
    };

    const VkRenderingInfo renderingInfo{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea           = { .offset = { 0, 0 }, .extent = m_swapchain.getExtent() },
        .layerCount           = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &colorAttachment,
        .pDepthAttachment     = &depthAttachment,
    };

    m_gpuTimer.writeTimestamp(cmd, "ScenePass_Begin");

    vkCmdBeginRendering(cmd, &renderingInfo);

    // Viewport and scissor are shared by sky and PBR geometry passes
    const VkExtent2D ext = m_swapchain.getExtent();
    const VkViewport viewport{
        .x        = 0.0f,
        .y        = 0.0f,
        .width    = static_cast<float>(ext.width),
        .height   = static_cast<float>(ext.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    const VkRect2D scissor{ .offset = { 0, 0 }, .extent = ext };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // ── Sky pass: fullscreen triangle at reverse-Z far plane (z = 0) ─────────
    // GREATER_OR_EQUAL depth: passes where depth == 0.0 (clear value, no geometry yet).
    // Drawn first — PBR geometry (GREATER at z > 0) overwrites sky on covered pixels.
    m_skyPass.record(cmd, m_descriptorSet, ext, m_skyEnabled, m_skyMode);

    // ── PBR geometry pass ─────────────────────────────────────────────────────
    VkPipeline activePipeline = m_pipeline;
    if (m_showNormals)
        activePipeline = m_normalsPipeline;
    else if (m_wireframe)
        activePipeline = m_wireframePipeline;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, activePipeline);

    // Bind camera UBO descriptor set (set=0, binding=0 — view/projection matrices)
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                            shader_interface::kSceneSet, 1, &m_descriptorSet, 0, nullptr);

    // Bind vertex and index buffers
    const VkDeviceSize vertexOffset = 0;
    const VkBuffer vertexBuf = m_vertexBuffer.getBuffer();
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuf, &vertexOffset);
    vkCmdBindIndexBuffer(cmd, m_indexBuffer.getBuffer(), 0, VK_INDEX_TYPE_UINT32);

    // Push normalization model matrix (bytes 0–63): same transform as shadow pass.
    for (const DrawCommand& draw : m_drawCommands) {
        if (!draw.mesh.isValid() || draw.mesh.index >= m_meshes.size())
            continue;

        const MeshRenderData& mesh = m_meshes[draw.mesh.index];

        vkCmdPushConstants(cmd, m_pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           shader_interface::kModelMatrixPcOffset, sizeof(glm::mat4), &draw.transform);

        if (draw.material.isValid() &&
            draw.material.index < m_materialSets.size() &&
            draw.material.index < m_materials.size()) {

            // Bind material texture set (set=1) — set=0 remains bound and unaffected.
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    m_pipelineLayout,
                                    shader_interface::kMaterialSet, 1, &m_materialSets[draw.material.index],
                                    0, nullptr);

            // Push material factors to fragment stage (offset 64, after model matrix).
            const Material& mat = m_materials[draw.material.index];
            const MaterialPushConstants matPC{
                .baseColorFactor  = mat.baseColorFactor,
                .metallicFactor   = mat.metallicFactor,
                .roughnessFactor  = mat.roughnessFactor,
                .alphaCutoff      = (mat.alphaMode == Material::AlphaMode::Mask)
                                    ? mat.alphaCutoff : 0.0f,
            };
            vkCmdPushConstants(cmd, m_pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               shader_interface::kMaterialPcOffset, sizeof(MaterialPushConstants), &matPC);
        } else {
            // No material — push default white/neutral factors.
            const MaterialPushConstants defaultPC{};
            vkCmdPushConstants(cmd, m_pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               shader_interface::kMaterialPcOffset, sizeof(MaterialPushConstants), &defaultPC);
        }

        vkCmdDrawIndexed(cmd, mesh.indexCount, 1, mesh.firstIndex, 0, 0);
        m_renderStats.drawCalls++;
        m_renderStats.triangles += mesh.indexCount / 3;
    }

    vkCmdEndRendering(cmd);

    m_gpuTimer.writeTimestamp(cmd, "ScenePass_End");

    // ── Transition HDR target: attachment write → fragment shader read ────────
    vkutil::transitionImage(cmd, m_hdrTarget.getImage(),
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,            VK_ACCESS_2_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // ── Transition swapchain: UNDEFINED → COLOR_ATTACHMENT (for tonemap blit) ─
    vkutil::transitionImage(cmd, m_swapchain.getCurrentImage(),
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,              0,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // ── Tone map pass: fullscreen triangle HDR→LDR → swapchain SRGB ──────────
    m_gpuTimer.writeTimestamp(cmd, "TonemapPass_Begin");

    const shader_interface::TonemapPC tonemapPC{
        static_cast<uint32_t>(m_tonemapMode),
        m_exposure,
        static_cast<uint32_t>(m_splitScreenMode),
        static_cast<uint32_t>(m_splitRightMode),
    };
    m_tonemapPass.record(cmd, m_swapchain.getCurrentImageView(), ext, tonemapPC);

    m_gpuTimer.writeTimestamp(cmd, "TonemapPass_End");

    // ImGui overlay pass (LOAD_OP_LOAD preserves the tone-mapped image).
    if (m_imguiManager) {
        m_imguiManager->recordRenderPass(
            cmd,
            m_swapchain.getCurrentImageView(),
            m_swapchain.getExtent());
    }

    m_gpuTimer.writeTimestamp(cmd, "ImGuiPass_End");

    // Transition to PRESENT_SRC_KHR so the presentation engine can consume the image.
    // Without this barrier, MoltenVK's Metal presentation layer sees an incorrect layout.
    vkutil::transitionImage(cmd, m_swapchain.getCurrentImage(),
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,           0,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,          VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VK_CHECK(vkEndCommandBuffer(cmd));
}

void Renderer::requestScreenshot(const std::string& filename)
{
    m_screenshotRequested = true;
    m_screenshotFilename  = filename;
}

void Renderer::submitFrame(const RenderFramePacket& packet)
{
    setCameraMatrices(packet.camera.view, packet.camera.projection, packet.camera.position);
    setCameraFrustum(packet.camera.nearZ, packet.camera.farZ);

    m_lightDirection    = safeNormalizeDirection(packet.light.direction, glm::vec3(0.577f));
    m_lightColor        = packet.light.color;
    m_lightIntensity    = packet.light.intensity;
    m_ambientIntensity  = packet.light.ambient;

    m_shadowSettings = packet.shadow;
    m_shadowPass.updateCamera(m_viewMatrix, m_projMatrix, m_cameraNearZ, m_cameraFarZ, m_cameraPos);
    m_shadowPass.updateLightDirection(m_lightDirection);
    const bool shadowResourcesChanged = m_shadowPass.updateSettings(m_shadowSettings);
    const bool shadowDescriptorsDirty = m_shadowPass.consumeDescriptorDirty();
    if (shadowResourcesChanged || shadowDescriptorsDirty) {
        updateShadowDescriptors();
    }
    uploadLightUBO();
    if (!packet.pointLights.empty()) {
        setPointLights(packet.pointLights);
    } else {
        uploadClusterMetadata();
    }

    setTonemapParams(packet.tonemap.mode,
                     packet.tonemap.exposure,
                     packet.tonemap.splitScreen,
                     packet.tonemap.splitRightMode);

    m_skyEnabled = packet.sky.enabled;
    m_skyMode    = packet.sky.mode;

    m_wireframe   = packet.debug.wireframe;
    m_showNormals = packet.debug.showNormals;
}

void Renderer::setCameraMatrices(const glm::mat4& view, const glm::mat4& projection,
                                  const glm::vec3& cameraPos)
{
    m_cameraPos  = cameraPos;
    m_viewMatrix = view;
    m_projMatrix = projection;
    // inverseVP used by sky.vert to reconstruct world-space ray directions from clip space.
    const CameraUBO ubo{ view, projection, glm::inverse(projection * view), cameraPos, 0.0f };
    m_cameraUBOBuffer.upload(&ubo, sizeof(CameraUBO));
}

void Renderer::setCameraFrustum(float nearZ, float farZ)
{
    m_cameraNearZ = nearZ;
    m_cameraFarZ  = farZ;
}

void Renderer::endFrame()
{
    render();

    // Reset fence immediately before the submit that will signal it.
    // (Moved from beginFrame to prevent deadlock on skipped frames.)
    m_frameSync.resetFence();

    m_commandBuffer.submit(
        m_ctx.getGraphicsQueue(),
        m_frameSync.getImageAvailableSemaphore(),
        m_frameSync.getRenderFinishedSemaphore(),
        m_frameSync.getInFlightFence());

    const VkSemaphore   renderDone   = m_frameSync.getRenderFinishedSemaphore();
    const uint32_t      imageIndex   = m_swapchain.getImageIndex();
    const VkSwapchainKHR swapchain   = m_swapchain.getSwapchain();

    const VkPresentInfoKHR presentInfo{
        .sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores    = &renderDone,
        .swapchainCount     = 1,
        .pSwapchains        = &swapchain,
        .pImageIndices      = &imageIndex,
    };

    const VkResult presentResult = vkQueuePresentKHR(m_ctx.getGraphicsQueue(), &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
        handleResize();
    } else if (presentResult != VK_SUCCESS)
        throw std::runtime_error("vkQueuePresentKHR failed: " +
                                 std::to_string(static_cast<int>(presentResult)));

    if (m_screenshotRequested) {
        // Block until the device is fully idle so the presented image is safe to read.
        vkDeviceWaitIdle(m_ctx.getDevice());
        m_screenshot.capture(m_swapchain, m_screenshotFilename);
        m_screenshotRequested = false;
        m_screenshotFilename.clear();
    }

    // Advance both frame indices together to keep sync primitives and command buffers aligned
    m_frameSync.advanceFrame();
    m_commandBuffer.beginFrame();
}
