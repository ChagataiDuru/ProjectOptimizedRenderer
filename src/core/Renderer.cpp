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
    , m_samplerCache(ctx)
    , m_fallbackWhite(ctx)
    , m_depthImage(ctx)
    , m_hdrTarget(ctx)
    , m_shadowMap(ctx)
    , m_shadowUBOBuffer(ctx)
    , m_shadowMoments(ctx)
    , m_shadowMomentsTemp(ctx)
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
    loadModel(std::string(ASSET_DIR) + "/models/sponza/glTF/Sponza.gltf");
    m_sceneInfo = computeSceneInfo(m_model.boundsMin, m_model.boundsMax);
    spdlog::info("Scene normalized: scale={:.4f}, radius={:.2f}",
                 m_sceneInfo.scaleFactor, m_sceneInfo.normalizedRadius);
    m_frameSync.init(3);
    m_commandBuffer.init(m_ctx.getGraphicsQueueFamily(), 3);
    createDepthImage();
    createHdrTarget();
    createTonemapPipeline();
    createPbrPipeline();
    createShadowResources();
    createCameraUBO();
    createLightUBO();
    createDescriptorPool();
    createDescriptorSet();
    createMaterialDescriptorSets();
    m_gpuTimer.init(16);

    // Populate static render stats (counts that don't change after load)
    m_renderStats.meshCount     = static_cast<uint32_t>(m_meshRenderData.size());
    m_renderStats.materialCount = static_cast<uint32_t>(m_model.materials.size());
    m_renderStats.textureCount  = static_cast<uint32_t>(m_textures.size());
    m_renderStats.textureMemoryBytes = 0;
    for (const auto& tex : m_textures) {
        if (tex.isValid()) {
            const VkExtent3D ext = tex.getExtent();
            // Each texel is 4 bytes (RGBA8); depth = 1 for 2D textures.
            m_renderStats.textureMemoryBytes += static_cast<size_t>(ext.width) * ext.height * 4;
        }
    }

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

    m_meshRenderData.clear();
    m_model = Model{};

    // ── Load new model and upload resources ──────────────────────────────────────
    try {
        loadModel(modelPath);
    } catch (const std::exception& e) {
        spdlog::error("Failed to load model '{}': {}", modelPath, e.what());
        spdlog::info("Falling back to Sponza");
        try {
            loadModel(std::string(ASSET_DIR) + "/models/sponza/glTF/Sponza.gltf");
        } catch (const std::exception& e2) {
            spdlog::critical("Failed to load fallback model: {}", e2.what());
            throw;  // Unrecoverable
        }
    }

    // ── Compute normalization transform for the new model ────────────────────────
    m_sceneInfo = computeSceneInfo(m_model.boundsMin, m_model.boundsMax);
    spdlog::info("Scene normalized: scale={:.4f}, radius={:.2f}",
                 m_sceneInfo.scaleFactor, m_sceneInfo.normalizedRadius);

    // ── Recreate descriptors for the new model ───────────────────────────────────
    createDescriptorPool();
    createDescriptorSet();
    createMaterialDescriptorSets();

    // ── Update render stats ──────────────────────────────────────────────────────
    m_renderStats.meshCount     = static_cast<uint32_t>(m_meshRenderData.size());
    m_renderStats.materialCount = static_cast<uint32_t>(m_model.materials.size());
    m_renderStats.textureCount  = static_cast<uint32_t>(m_textures.size());
    m_renderStats.textureMemoryBytes = 0;
    for (const auto& tex : m_textures) {
        if (tex.isValid()) {
            const VkExtent3D ext = tex.getExtent();
            m_renderStats.textureMemoryBytes += static_cast<size_t>(ext.width) * ext.height * 4;
        }
    }

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

    // Phase 5: tone map pass resources (model-independent; destroy before PBR pipeline)
    {
        const VkDevice dev = m_ctx.getDevice();
        if (m_tonemapPipeline       != VK_NULL_HANDLE) { vkDestroyPipeline(dev, m_tonemapPipeline, nullptr);              m_tonemapPipeline       = VK_NULL_HANDLE; }
        if (m_tonemapPipelineLayout != VK_NULL_HANDLE) { vkDestroyPipelineLayout(dev, m_tonemapPipelineLayout, nullptr);   m_tonemapPipelineLayout = VK_NULL_HANDLE; }
        if (m_tonemapPool           != VK_NULL_HANDLE) { vkDestroyDescriptorPool(dev, m_tonemapPool, nullptr);             m_tonemapPool           = VK_NULL_HANDLE; m_tonemapSet = VK_NULL_HANDLE; }
        if (m_tonemapSetLayout      != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(dev, m_tonemapSetLayout, nullptr);   m_tonemapSetLayout      = VK_NULL_HANDLE; }
        if (m_hdrSampler            != VK_NULL_HANDLE) { vkDestroySampler(dev, m_hdrSampler, nullptr);                     m_hdrSampler            = VK_NULL_HANDLE; }
        m_hdrTarget.destroy();
    }

    destroyPipeline();

    // Destroy textures before VMA teardown (textures hold VmaAllocations)
    m_fallbackWhite.destroy();
    for (auto& tex : m_textures)
        tex.destroy();
    m_textures.clear();
    m_samplerCache.shutdown();

    // Destroy shadow resources
    const VkDevice dev = m_ctx.getDevice();

    // VSM blur compute resources
    if (m_blurPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, m_blurPipeline, nullptr);
        m_blurPipeline = VK_NULL_HANDLE;
    }
    if (m_blurPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(dev, m_blurPipelineLayout, nullptr);
        m_blurPipelineLayout = VK_NULL_HANDLE;
    }
    if (m_blurDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(dev, m_blurDescriptorPool, nullptr);
        m_blurDescriptorPool  = VK_NULL_HANDLE;
        m_blurSetHorizontal   = VK_NULL_HANDLE;
        m_blurSetVertical     = VK_NULL_HANDLE;
    }
    if (m_blurSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(dev, m_blurSetLayout, nullptr);
        m_blurSetLayout = VK_NULL_HANDLE;
    }

    // VSM shadow pipeline
    if (m_shadowVsmPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, m_shadowVsmPipeline, nullptr);
        m_shadowVsmPipeline = VK_NULL_HANDLE;
    }

    // VSM moment images and their per-layer views
    if (m_momentsSampler != VK_NULL_HANDLE) {
        vkDestroySampler(dev, m_momentsSampler, nullptr);
        m_momentsSampler = VK_NULL_HANDLE;
    }
    for (auto& view : m_momentsLayerViews) {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(dev, view, nullptr);
            view = VK_NULL_HANDLE;
        }
    }
    m_shadowMomentsTemp.destroy();
    m_shadowMoments.destroy();

    // Hard shadow sampler and per-layer depth views
    if (m_shadowSampler != VK_NULL_HANDLE) {
        vkDestroySampler(dev, m_shadowSampler, nullptr);
        m_shadowSampler = VK_NULL_HANDLE;
    }
    for (auto& view : m_shadowLayerViews) {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(dev, view, nullptr);
            view = VK_NULL_HANDLE;
        }
    }
    m_shadowMap.destroy();
    m_shadowUBOBuffer.destroy();

    // Destroy GPU resources that hold VMA allocations
    m_depthImage.destroy();
    m_lightUBOBuffer.destroy();
    m_cameraUBOBuffer.destroy();
    m_indexBuffer.destroy();
    m_vertexBuffer.destroy();

    m_commandBuffer.shutdown();
    m_frameSync.shutdown();
}

// ── Geometry ──────────────────────────────────────────────────────────────────

void Renderer::loadModel(const std::string& modelPath)
{
    m_model = GLTFLoader::loadGLTF(modelPath);

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
        data.materialIndex = mesh.materialIndex;
        m_meshRenderData.push_back(data);

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
    const VkCommandPoolCreateInfo poolCI{
        .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = m_ctx.getGraphicsQueueFamily(),
    };
    VkCommandPool transferPool;
    VK_CHECK(vkCreateCommandPool(m_ctx.getDevice(), &poolCI, nullptr, &transferPool));

    const VkCommandBufferAllocateInfo allocInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = transferPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer transferCmd;
    VK_CHECK(vkAllocateCommandBuffers(m_ctx.getDevice(), &allocInfo, &transferCmd));

    const VkCommandBufferBeginInfo beginInfo{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    VK_CHECK(vkBeginCommandBuffer(transferCmd, &beginInfo));

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
            // Leave tex invalid — draw loop will use fallback in Phase 2.3
        }
    }

    VK_CHECK(vkEndCommandBuffer(transferCmd));

    VkFence fence;
    const VkFenceCreateInfo fenceCI{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VK_CHECK(vkCreateFence(m_ctx.getDevice(), &fenceCI, nullptr, &fence));

    const VkSubmitInfo submitInfo{
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &transferCmd,
    };
    VK_CHECK(vkQueueSubmit(m_ctx.getGraphicsQueue(), 1, &submitInfo, fence));
    VK_CHECK(vkWaitForFences(m_ctx.getDevice(), 1, &fence, VK_TRUE, UINT64_MAX));

    m_vertexBuffer.releaseStaging();
    m_indexBuffer.releaseStaging();
    m_fallbackWhite.releaseStaging();
    for (auto& tex : m_textures)
        tex.releaseStaging();

    vkDestroyFence(m_ctx.getDevice(), fence, nullptr);
    vkDestroyCommandPool(m_ctx.getDevice(), transferPool, nullptr);

    spdlog::info("Model uploaded: {} meshes, {} vertices, {} indices",
                 m_model.meshes.size(), allVertices.size(), allIndices.size());
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

    // Phase 1.5: reverse-Z depth — near=1.0 far=0.0, clear to 0.0, compare GREATER
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
    const std::array<VkDescriptorSetLayoutBinding, 5> setBindings{{
        {   // binding 0: camera matrices — read in vertex + fragment shaders
            .binding         = 0,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {   // binding 1: directional light — read in fragment shader
            .binding         = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {   // binding 2: shadow UBO (lightViewProj[4] + splitDepths) — vertex + fragment
            .binding         = 2,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {   // binding 3: shadow depth array (D32_SFLOAT, None/PCF modes) — fragment only
            .binding         = 3,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {   // binding 4: shadow moments array (RG32_SFLOAT, VSM mode) — fragment only
            .binding         = 4,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
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
            .binding         = 0,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding         = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding         = 2,
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

    // Three non-overlapping push constant ranges:
    //   bytes  0–63:  mat4 model matrix (vertex stage)
    //   bytes 64–95:  MaterialPushConstants — factors (fragment stage)
    //   bytes 96–99:  uint cascadeIndex (vertex+fragment — shadow pass selects lightViewProj)
    const std::array<VkPushConstantRange, 3> pushRanges{{
        {
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset     = 0,
            .size       = sizeof(glm::mat4),
        },
        {
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset     = 64,
            .size       = sizeof(MaterialPushConstants),
        },
        {
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset     = 96,
            .size       = sizeof(uint32_t),
        },
    }};

    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = static_cast<uint32_t>(allSetLayouts.size()),
        .pSetLayouts            = allSetLayouts.data(),
        .pushConstantRangeCount = static_cast<uint32_t>(pushRanges.size()),
        .pPushConstantRanges    = pushRanges.data(),
    };
    VK_CHECK(vkCreatePipelineLayout(m_ctx.getDevice(), &layoutInfo, nullptr, &m_pipelineLayout));

    // VkPipelineRenderingCreateInfo replaces VkRenderPass for dynamic rendering (core in 1.3/1.4)
    // Phase 5: PBR pipelines now render into the HDR float target, not directly to the swapchain.
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

    const VkGraphicsPipelineCreateInfo pipelineInfo{
        .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext               = &renderingInfo,
        .stageCount          = static_cast<uint32_t>(stages.size()),
        .pStages             = stages.data(),
        .pVertexInputState   = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState      = &viewportState,
        .pRasterizationState = &rasterization,
        .pMultisampleState   = &multisample,
        .pDepthStencilState  = &depthStencil,
        .pColorBlendState    = &colorBlend,
        .pDynamicState       = &dynamicState,
        .layout              = m_pipelineLayout,
        .renderPass          = VK_NULL_HANDLE,  // Dynamic rendering: no render pass object
    };

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
    if (m_shadowPipeline     != VK_NULL_HANDLE) { vkDestroyPipeline(dev, m_shadowPipeline, nullptr);               m_shadowPipeline    = VK_NULL_HANDLE; }
    if (m_shadowVertModule   != VK_NULL_HANDLE) { vkDestroyShaderModule(dev, m_shadowVertModule, nullptr);         m_shadowVertModule  = VK_NULL_HANDLE; }
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

    const CameraUBO initial{ glm::mat4(1.0f), glm::mat4(1.0f), glm::vec3(0.0f), 0.0f };
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
        m_showCascadeDebug ? 1u : 0u,
        static_cast<uint32_t>(m_shadowFilterMode),
        m_pcfSpreadRadius,
        m_vsmBleedReduction,
    };
    m_lightUBOBuffer.upload(&light, sizeof(LightUBO));
}

void Renderer::setCascadeDebugEnabled(bool enabled)
{
    m_showCascadeDebug = enabled;
    uploadLightUBO();
}

void Renderer::setShadowFilterMode(int32_t mode)
{
    m_shadowFilterMode = mode;
    uploadLightUBO();
}

void Renderer::setPcfSpreadRadius(float radius)
{
    m_pcfSpreadRadius = radius;
    uploadLightUBO();
}

void Renderer::setVsmBleedReduction(float reduction)
{
    m_vsmBleedReduction = reduction;
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

// Phase 5: HDR offscreen target ──────────────────────────────────────────────

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

    // Update the descriptor set to point at the new image view (also handles first-time setup
    // after the descriptor pool+set are created by createTonemapPipeline()).
    if (m_tonemapSet != VK_NULL_HANDLE)
        updateTonemapDescriptorSet();

    spdlog::info("HDR target created: {}x{} R16G16B16A16_SFLOAT", ext.width, ext.height);
}

void Renderer::createTonemapPipeline()
{
    const VkDevice dev = m_ctx.getDevice();
    const std::string dir = SHADER_DIR;

    // ── Descriptor set layout: one combined image sampler (HDR input) ─────────
    const VkDescriptorSetLayoutBinding binding{
        .binding         = 0,
        .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    const VkDescriptorSetLayoutCreateInfo setLayoutCI{
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings    = &binding,
    };
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &setLayoutCI, nullptr, &m_tonemapSetLayout));

    // ── Descriptor pool: 1 set, 1 sampler ─────────────────────────────────────
    const VkDescriptorPoolSize poolSize{
        .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
    };
    const VkDescriptorPoolCreateInfo poolCI{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = 1,
        .poolSizeCount = 1,
        .pPoolSizes    = &poolSize,
    };
    VK_CHECK(vkCreateDescriptorPool(dev, &poolCI, nullptr, &m_tonemapPool));

    // ── Allocate descriptor set ────────────────────────────────────────────────
    const VkDescriptorSetAllocateInfo allocInfo{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = m_tonemapPool,
        .descriptorSetCount = 1,
        .pSetLayouts        = &m_tonemapSetLayout,
    };
    VK_CHECK(vkAllocateDescriptorSets(dev, &allocInfo, &m_tonemapSet));

    // ── Push constant: TonemapPC (16 bytes: mode + exposure + splitScreenMode + splitRightMode) ──
    const VkPushConstantRange pcRange{
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset     = 0,
        .size       = 4 * sizeof(uint32_t),  // 16 bytes: aligns to vec4 boundary
    };
    const VkPipelineLayoutCreateInfo layoutCI{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_tonemapSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pcRange,
    };
    VK_CHECK(vkCreatePipelineLayout(dev, &layoutCI, nullptr, &m_tonemapPipelineLayout));

    // ── Shader modules ─────────────────────────────────────────────────────────
    VkShaderModule vertMod = makeShaderModule(dev, loadSpv(dir + "/tonemap.vert.spv"));
    VkShaderModule fragMod = makeShaderModule(dev, loadSpv(dir + "/tonemap.frag.spv"));

    const std::array<VkPipelineShaderStageCreateInfo, 2> stages{{
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT,   .module = vertMod, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragMod, .pName = "main" },
    }};

    // ── No vertex input — fullscreen triangle is generated entirely in the vertex shader ──
    const VkPipelineVertexInputStateCreateInfo vertexInput{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };
    const VkPipelineInputAssemblyStateCreateInfo inputAssembly{
        .sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };
    const VkPipelineViewportStateCreateInfo viewportState{
        .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount  = 1,
    };
    const VkPipelineRasterizationStateCreateInfo rasterization{
        .sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode    = VK_CULL_MODE_NONE,
        .frontFace   = VK_FRONT_FACE_CLOCKWISE,
        .lineWidth   = 1.0f,
    };
    const VkPipelineMultisampleStateCreateInfo multisample{
        .sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    // No depth test: fullscreen blit always overwrites
    const VkPipelineDepthStencilStateCreateInfo depthStencil{
        .sType             = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable   = VK_FALSE,
        .depthWriteEnable  = VK_FALSE,
    };
    const VkPipelineColorBlendAttachmentState blendAtt{
        .blendEnable    = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    const VkPipelineColorBlendStateCreateInfo colorBlend{
        .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments    = &blendAtt,
    };
    const std::array<VkDynamicState, 2> dynStates = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    const VkPipelineDynamicStateCreateInfo dynamicState{
        .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(dynStates.size()),
        .pDynamicStates    = dynStates.data(),
    };

    // Output to swapchain SRGB format
    const VkFormat swapchainFormat = m_swapchain.getFormat();
    const VkPipelineRenderingCreateInfo renderingInfo{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &swapchainFormat,
        // No depth attachment — blit doesn't use depth
    };

    const VkGraphicsPipelineCreateInfo pipelineCI{
        .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext               = &renderingInfo,
        .stageCount          = static_cast<uint32_t>(stages.size()),
        .pStages             = stages.data(),
        .pVertexInputState   = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState      = &viewportState,
        .pRasterizationState = &rasterization,
        .pMultisampleState   = &multisample,
        .pDepthStencilState  = &depthStencil,
        .pColorBlendState    = &colorBlend,
        .pDynamicState       = &dynamicState,
        .layout              = m_tonemapPipelineLayout,
        .renderPass          = VK_NULL_HANDLE,
    };
    VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pipelineCI, nullptr,
                                       &m_tonemapPipeline));

    vkDestroyShaderModule(dev, vertMod, nullptr);
    vkDestroyShaderModule(dev, fragMod, nullptr);

    // Now that pool+set+layout exist, write the HDR image into the descriptor
    // (m_hdrTarget was already created before createTonemapPipeline() is called)
    updateTonemapDescriptorSet();

    spdlog::info("Tone map pipeline created (fullscreen triangle → swapchain SRGB)");
}

void Renderer::updateTonemapDescriptorSet()
{
    const VkDescriptorImageInfo imgInfo{
        .sampler     = m_hdrSampler,
        .imageView   = m_hdrTarget.getImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkWriteDescriptorSet write{
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet          = m_tonemapSet,
        .dstBinding      = 0,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo      = &imgInfo,
    };
    vkUpdateDescriptorSets(m_ctx.getDevice(), 1, &write, 0, nullptr);
}

void Renderer::createDescriptorPool()
{
    // Pool must hold:
    //   1 set  × 3 uniform buffers (camera + light + shadow, set=0)
    //   1 set  × 1 combined image sampler (shadow map, set=0)
    //   N sets × 3 combined image samplers each (material textures, set=1)
    const uint32_t materialCount = static_cast<uint32_t>(m_model.materials.size());
    const uint32_t maxSets       = 1 + std::max(materialCount, 1u);
    // 2 samplers in set=0 (depth array b3 + moments array b4) + 3 per material
    const uint32_t samplerCount  = 2 + 3 * std::max(materialCount, 1u);

    const std::array<VkDescriptorPoolSize, 2> poolSizes{{
        {
            .type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 3,  // camera + light + shadow
        },
        {
            .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = samplerCount,
        },
    }};

    const VkDescriptorPoolCreateInfo poolCI{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = maxSets,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes    = poolSizes.data(),
    };
    VK_CHECK(vkCreateDescriptorPool(m_ctx.getDevice(), &poolCI, nullptr, &m_descriptorPool));

    spdlog::info("Descriptor pool created: {} max sets (3 UBO, {} sampler descriptors)",
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
        .buffer = m_shadowUBOBuffer.getBuffer(),
        .offset = 0,
        .range  = sizeof(ShadowCascadeUBO),
    };
    const VkDescriptorImageInfo shadowMapInfo{
        .sampler     = m_shadowSampler,
        .imageView   = m_shadowMap.getImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
    };
    const VkDescriptorImageInfo momentsInfo{
        .sampler     = m_momentsSampler,
        .imageView   = m_shadowMoments.getImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    const std::array<VkWriteDescriptorSet, 5> writes{{
        {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = m_descriptorSet,
            .dstBinding      = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo     = &cameraInfo,
        },
        {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = m_descriptorSet,
            .dstBinding      = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo     = &lightInfo,
        },
        {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = m_descriptorSet,
            .dstBinding      = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo     = &shadowUBOInfo,
        },
        {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = m_descriptorSet,
            .dstBinding      = 3,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo      = &shadowMapInfo,
        },
        {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = m_descriptorSet,
            .dstBinding      = 4,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo      = &momentsInfo,
        },
    }};
    vkUpdateDescriptorSets(m_ctx.getDevice(),
                           static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    spdlog::info("Descriptor set allocated: camera(b0), light(b1), shadowUBO(b2), depth(b3), moments(b4)");
}

void Renderer::createMaterialDescriptorSets()
{
    const uint32_t materialCount = static_cast<uint32_t>(m_model.materials.size());
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
        const Material& mat = m_model.materials[i];

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
                .dstBinding      = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo      = &albedoInfo,
            },
            {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = m_materialSets[i],
                .dstBinding      = 1,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo      = &normalInfo,
            },
            {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = m_materialSets[i],
                .dstBinding      = 2,
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

// ── Shadow resources ──────────────────────────────────────────────────────────

// Unproject NDC slice [nearNDC, farNDC] using inverse(proj*view) to get 8 world-space corners.
static std::array<glm::vec3, 8> frustumCornersWorld(
    const glm::mat4& view, const glm::mat4& proj,
    float nearNDC, float farNDC)
{
    const glm::mat4 invVP = glm::inverse(proj * view);
    const std::array<glm::vec4, 8> ndc{{
        { -1, -1, nearNDC, 1 }, {  1, -1, nearNDC, 1 },
        { -1,  1, nearNDC, 1 }, {  1,  1, nearNDC, 1 },
        { -1, -1, farNDC,  1 }, {  1, -1, farNDC,  1 },
        { -1,  1, farNDC,  1 }, {  1,  1, farNDC,  1 },
    }};
    std::array<glm::vec3, 8> world;
    for (int i = 0; i < 8; ++i) {
        const glm::vec4 w = invVP * ndc[i];
        world[i] = glm::vec3(w) / w.w;
    }
    return world;
}

// Minimum bounding sphere of 8 points (center = average, radius = max distance).
// Rotation-invariant radius enables stable texel snapping without shimmer.
static std::pair<glm::vec3, float> boundingSphere(const std::array<glm::vec3, 8>& pts)
{
    glm::vec3 center(0.0f);
    for (const auto& p : pts) center += p;
    center /= 8.0f;
    float r = 0.0f;
    for (const auto& p : pts) r = std::max(r, glm::length(p - center));
    return { center, r };
}

void Renderer::createShadowResources()
{
    const VkDevice dev = m_ctx.getDevice();

    // ── Shadow depth image (2D array: one layer per cascade) ─────────────────
    // Standard Z (not reverse-Z): shadow map is compared with LESS_OR_EQUAL.
    // Usage: DEPTH_STENCIL_ATTACHMENT (shadow pass) + SAMPLED (PBR pass reads it).
    m_shadowMap.create(CASCADE_SIZE, CASCADE_SIZE, 1,
                       VK_FORMAT_D32_SFLOAT,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                       VK_IMAGE_ASPECT_DEPTH_BIT,
                       CASCADE_COUNT);

    // Per-layer views for rendering (shadow pass attaches each cascade layer individually)
    for (uint32_t c = 0; c < CASCADE_COUNT; ++c)
        m_shadowLayerViews[c] = m_shadowMap.createSingleLayerView(c, VK_IMAGE_ASPECT_DEPTH_BIT);

    // ── Shadow sampler (non-comparison) ──────────────────────────────────────
    // compareEnable = VK_FALSE: raw depth fetch; manual comparison done in the shader.
    // This avoids mutableComparisonSamplers (VK_KHR_portability_subset on MoltenVK).
    // NEAREST filter: linear filtering of raw depth values is meaningless for hard shadows.
    const VkSamplerCreateInfo samplerCI{
        .sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter        = VK_FILTER_NEAREST,
        .minFilter        = VK_FILTER_NEAREST,
        .mipmapMode       = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeV     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .addressModeW     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
        .mipLodBias       = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .compareEnable    = VK_FALSE,  // Manual comparison in shader — MoltenVK portable
        .minLod           = 0.0f,
        .maxLod           = 0.0f,
        .borderColor      = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,  // outside map = fully lit
    };
    VK_CHECK(vkCreateSampler(dev, &samplerCI, nullptr, &m_shadowSampler));

    // ── Shadow UBO (4× lightViewProj + splitDepths) ───────────────────────────
    m_shadowUBOBuffer.createHostVisible(sizeof(ShadowCascadeUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    updateShadowMatrices();  // compute initial matrices from scene bounds

    // ── Shadow vertex shader ──────────────────────────────────────────────────
    auto shadowSpv = loadSpv(std::string(SHADER_DIR) + "/shadow.vert.spv");
    const VkShaderModuleCreateInfo smCI{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = shadowSpv.size() * sizeof(uint32_t),
        .pCode    = shadowSpv.data(),
    };
    VK_CHECK(vkCreateShaderModule(dev, &smCI, nullptr, &m_shadowVertModule));

    // ── Shadow pipeline ───────────────────────────────────────────────────────
    // Depth-only: no color attachment, no fragment shader.
    // Uses the same pipeline layout as PBR (push constant b0 = model matrix, set=0 b2 = shadow UBO).
    const VkPipelineShaderStageCreateInfo shadowStage{
        .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage  = VK_SHADER_STAGE_VERTEX_BIT,
        .module = m_shadowVertModule,
        .pName  = "main",
    };

    // Vertex layout matches PBR: position(0), normal(1), uv(2), tangent(3)
    const std::array<VkVertexInputBindingDescription, 1> shadowBindings{{
        { .binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX },
    }};
    const std::array<VkVertexInputAttributeDescription, 4> shadowAttribs{{
        { .location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position) },
        { .location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal)   },
        { .location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT,    .offset = offsetof(Vertex, uv)       },
        { .location = 3, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, tangent)  },
    }};
    const VkPipelineVertexInputStateCreateInfo shadowVertexInput{
        .sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount   = static_cast<uint32_t>(shadowBindings.size()),
        .pVertexBindingDescriptions      = shadowBindings.data(),
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(shadowAttribs.size()),
        .pVertexAttributeDescriptions    = shadowAttribs.data(),
    };

    const VkPipelineInputAssemblyStateCreateInfo shadowInputAssembly{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = VK_FALSE,
    };

    const VkPipelineViewportStateCreateInfo shadowViewport{
        .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount  = 1,
    };

    // Depth bias to reduce shadow acne (constant + slope-scaled offset).
    const VkPipelineRasterizationStateCreateInfo shadowRasterization{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable        = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode             = VK_POLYGON_MODE_FILL,
        .cullMode                = VK_CULL_MODE_FRONT_BIT,  // front-face culling avoids peter-panning
        .frontFace               = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable         = VK_TRUE,
        .depthBiasConstantFactor = 1.5f,
        .depthBiasClamp          = 0.0f,
        .depthBiasSlopeFactor    = 2.0f,
        .lineWidth               = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo shadowMultisample{
        .sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .sampleShadingEnable  = VK_FALSE,
    };

    // Standard Z: clear to 1.0, compare LESS — closer fragments overwrite farther ones.
    const VkPipelineDepthStencilStateCreateInfo shadowDepthStencil{
        .sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable       = VK_TRUE,
        .depthWriteEnable      = VK_TRUE,
        .depthCompareOp        = VK_COMPARE_OP_LESS_OR_EQUAL,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable     = VK_FALSE,
    };

    // No color attachments for the shadow pass.
    const VkPipelineColorBlendStateCreateInfo shadowColorBlend{
        .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 0,
        .pAttachments    = nullptr,
    };

    const std::array<VkDynamicState, 3> shadowDynStates{
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_BIAS,
    };
    const VkPipelineDynamicStateCreateInfo shadowDynamic{
        .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(shadowDynStates.size()),
        .pDynamicStates    = shadowDynStates.data(),
    };

    // Dynamic rendering: depth-only, no color formats.
    const VkPipelineRenderingCreateInfo shadowRenderingInfo{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount    = 0,
        .pColorAttachmentFormats = nullptr,
        .depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT,
        .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
    };

    const VkGraphicsPipelineCreateInfo shadowPipelineCI{
        .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext               = &shadowRenderingInfo,
        .stageCount          = 1,
        .pStages             = &shadowStage,
        .pVertexInputState   = &shadowVertexInput,
        .pInputAssemblyState = &shadowInputAssembly,
        .pViewportState      = &shadowViewport,
        .pRasterizationState = &shadowRasterization,
        .pMultisampleState   = &shadowMultisample,
        .pDepthStencilState  = &shadowDepthStencil,
        .pColorBlendState    = &shadowColorBlend,
        .pDynamicState       = &shadowDynamic,
        .layout              = m_pipelineLayout,
        .renderPass          = VK_NULL_HANDLE,
    };
    VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &shadowPipelineCI,
                                       nullptr, &m_shadowPipeline));

    // ── VSM shadow pipeline (depth + RG32 moments color output) ──────────────
    // Reuse the same vertex shader (shadow.vert still loaded in m_shadowVertModule).
    auto vsmFragSpv = loadSpv(std::string(SHADER_DIR) + "/shadow_vsm.frag.spv");
    VkShaderModule vsmFragModule = makeShaderModule(dev, vsmFragSpv);

    const std::array<VkPipelineShaderStageCreateInfo, 2> vsmStages{{
        {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_VERTEX_BIT,
            .module = m_shadowVertModule,
            .pName  = "main",
        },
        {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = vsmFragModule,
            .pName  = "main",
        },
    }};

    const VkPipelineColorBlendAttachmentState vsmColorBlendAtt{
        .blendEnable    = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT,
    };
    const VkPipelineColorBlendStateCreateInfo vsmColorBlend{
        .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments    = &vsmColorBlendAtt,
    };

    const VkFormat momentsFormat = VK_FORMAT_R32G32_SFLOAT;
    const VkPipelineRenderingCreateInfo vsmRenderingInfo{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &momentsFormat,
        .depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT,
        .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
    };

    VkGraphicsPipelineCreateInfo vsmPipelineCI = shadowPipelineCI;
    vsmPipelineCI.pNext          = &vsmRenderingInfo;
    vsmPipelineCI.stageCount     = static_cast<uint32_t>(vsmStages.size());
    vsmPipelineCI.pStages        = vsmStages.data();
    vsmPipelineCI.pColorBlendState = &vsmColorBlend;

    VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &vsmPipelineCI,
                                       nullptr, &m_shadowVsmPipeline));

    // Both shadow pipelines created — release shader modules.
    vkDestroyShaderModule(dev, m_shadowVertModule, nullptr);  m_shadowVertModule = VK_NULL_HANDLE;
    vkDestroyShaderModule(dev, vsmFragModule, nullptr);

    // ── VSM moment images ─────────────────────────────────────────────────────
    // RG32_SFLOAT: stores (depth, depth²) per cascade layer.
    // COLOR_ATTACHMENT (shadow pass write) + SAMPLED (PBR read) + STORAGE (compute blur).
    m_shadowMoments.create(CASCADE_SIZE, CASCADE_SIZE, 1,
        VK_FORMAT_R32G32_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_STORAGE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        CASCADE_COUNT);

    m_shadowMomentsTemp.create(CASCADE_SIZE, CASCADE_SIZE, 1,
        VK_FORMAT_R32G32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        CASCADE_COUNT);

    for (uint32_t c = 0; c < CASCADE_COUNT; ++c)
        m_momentsLayerViews[c] = m_shadowMoments.createSingleLayerView(c, VK_IMAGE_ASPECT_COLOR_BIT);

    // ── Moments sampler (LINEAR — VSM blur gives valid linear interpolation) ──
    const VkSamplerCreateInfo momentsSamplerCI{
        .sType            = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter        = VK_FILTER_LINEAR,
        .minFilter        = VK_FILTER_LINEAR,
        .mipmapMode       = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW     = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .anisotropyEnable = VK_FALSE,
        .compareEnable    = VK_FALSE,
        .minLod           = 0.0f,
        .maxLod           = 0.0f,
        .borderColor      = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
    };
    VK_CHECK(vkCreateSampler(dev, &momentsSamplerCI, nullptr, &m_momentsSampler));

    // ── Initial layout transition: moments → SHADER_READ_ONLY_OPTIMAL ─────────
    // Ensures the PBR descriptor binding 4 is always in a valid layout,
    // even on the first frame before VSM mode is activated.
    {
        const VkCommandPoolCreateInfo initPoolCI{
            .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags            = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = m_ctx.getGraphicsQueueFamily(),
        };
        VkCommandPool initPool;
        VK_CHECK(vkCreateCommandPool(dev, &initPoolCI, nullptr, &initPool));

        const VkCommandBufferAllocateInfo initAllocInfo{
            .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool        = initPool,
            .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VkCommandBuffer initCmd;
        VK_CHECK(vkAllocateCommandBuffers(dev, &initAllocInfo, &initCmd));

        const VkCommandBufferBeginInfo initBegin{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        VK_CHECK(vkBeginCommandBuffer(initCmd, &initBegin));

        vkutil::transitionImage(initCmd, m_shadowMoments.getImage(),
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,        0,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,    VK_ACCESS_2_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_IMAGE_ASPECT_COLOR_BIT);

        VK_CHECK(vkEndCommandBuffer(initCmd));

        VkFence initFence;
        const VkFenceCreateInfo fenceCI{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        VK_CHECK(vkCreateFence(dev, &fenceCI, nullptr, &initFence));

        const VkSubmitInfo initSubmit{
            .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers    = &initCmd,
        };
        VK_CHECK(vkQueueSubmit(m_ctx.getGraphicsQueue(), 1, &initSubmit, initFence));
        VK_CHECK(vkWaitForFences(dev, 1, &initFence, VK_TRUE, UINT64_MAX));

        vkDestroyFence(dev, initFence, nullptr);
        vkDestroyCommandPool(dev, initPool, nullptr);
    }

    // ── Compute blur pipeline ─────────────────────────────────────────────────
    // Descriptor layout: 2 storage images (input and output image2DArrays).
    const std::array<VkDescriptorSetLayoutBinding, 2> blurBindings{{
        {
            .binding         = 0,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding         = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    }};
    const VkDescriptorSetLayoutCreateInfo blurSetLayoutCI{
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(blurBindings.size()),
        .pBindings    = blurBindings.data(),
    };
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &blurSetLayoutCI, nullptr, &m_blurSetLayout));

    const VkPushConstantRange blurPushRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = 2 * sizeof(int32_t),  // direction + radius
    };
    const VkPipelineLayoutCreateInfo blurLayoutCI{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_blurSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &blurPushRange,
    };
    VK_CHECK(vkCreatePipelineLayout(dev, &blurLayoutCI, nullptr, &m_blurPipelineLayout));

    auto blurSpv = loadSpv(std::string(SHADER_DIR) + "/shadow_blur.comp.spv");
    VkShaderModule blurModule = makeShaderModule(dev, blurSpv);

    const VkComputePipelineCreateInfo blurPipelineCI{
        .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage  = {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = blurModule,
            .pName  = "main",
        },
        .layout = m_blurPipelineLayout,
    };
    VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &blurPipelineCI,
                                      nullptr, &m_blurPipeline));
    vkDestroyShaderModule(dev, blurModule, nullptr);

    // ── Blur descriptor pool and sets (separate from the main pool) ───────────
    // 2 sets × 2 storage images = 4 storage image descriptors.
    const VkDescriptorPoolSize blurPoolSize{
        .type            = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 4,
    };
    const VkDescriptorPoolCreateInfo blurPoolCI{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = 2,
        .poolSizeCount = 1,
        .pPoolSizes    = &blurPoolSize,
    };
    VK_CHECK(vkCreateDescriptorPool(dev, &blurPoolCI, nullptr, &m_blurDescriptorPool));

    const std::array<VkDescriptorSetLayout, 2> blurLayouts = {
        m_blurSetLayout, m_blurSetLayout
    };
    const VkDescriptorSetAllocateInfo blurAllocInfo{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = m_blurDescriptorPool,
        .descriptorSetCount = 2,
        .pSetLayouts        = blurLayouts.data(),
    };
    std::array<VkDescriptorSet, 2> blurSets{};
    VK_CHECK(vkAllocateDescriptorSets(dev, &blurAllocInfo, blurSets.data()));
    m_blurSetHorizontal = blurSets[0];
    m_blurSetVertical   = blurSets[1];

    // Write horizontal set: moments (input) → temp (output)
    const VkDescriptorImageInfo blurMomentsInfo{
        .imageView   = m_shadowMoments.getImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo blurTempInfo{
        .imageView   = m_shadowMomentsTemp.getImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };

    const std::array<VkWriteDescriptorSet, 4> blurWrites{{
        { // horizontal: binding 0 = moments (read)
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = m_blurSetHorizontal,
            .dstBinding      = 0,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo      = &blurMomentsInfo,
        },
        { // horizontal: binding 1 = temp (write)
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = m_blurSetHorizontal,
            .dstBinding      = 1,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo      = &blurTempInfo,
        },
        { // vertical: binding 0 = temp (read)
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = m_blurSetVertical,
            .dstBinding      = 0,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo      = &blurTempInfo,
        },
        { // vertical: binding 1 = moments (write)
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = m_blurSetVertical,
            .dstBinding      = 1,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo      = &blurMomentsInfo,
        },
    }};
    vkUpdateDescriptorSets(dev, static_cast<uint32_t>(blurWrites.size()),
                           blurWrites.data(), 0, nullptr);

    spdlog::info("Shadow resources created: {}×{}×{} CSM, hard+VSM pipelines, blur compute",
                 CASCADE_SIZE, CASCADE_SIZE, CASCADE_COUNT);
}

void Renderer::updateShadowMatrices()
{
    const glm::vec3 lightDir = glm::normalize(m_lightDirection);

    // Degenerate lookAt guard: when light is nearly vertical, use X as up vector.
    const glm::vec3 up = (std::abs(lightDir.y) > 0.9f)
                         ? glm::vec3(1.0f, 0.0f, 0.0f)
                         : glm::vec3(0.0f, 1.0f, 0.0f);

    // Practical split scheme: lambda blends log (near-heavy) and uniform distributions.
    const float n      = m_cameraNearZ;
    const float f      = m_cameraFarZ;
    const float lambda = m_csmLambda;

    std::array<float, CASCADE_COUNT + 1> splitDepths;
    splitDepths[0]             = n;
    splitDepths[CASCADE_COUNT] = f;
    for (uint32_t i = 1; i < CASCADE_COUNT; ++i) {
        const float t          = static_cast<float>(i) / CASCADE_COUNT;
        const float logSplit     = n * std::pow(f / n, t);
        const float uniformSplit = n + (f - n) * t;
        splitDepths[i] = lambda * logSplit + (1.0f - lambda) * uniformSplit;
    }

    ShadowCascadeUBO ubo{};
    // splitDepths[1..4] are the cascade far-plane depths (positive view-space |Z|)
    ubo.splitDepths = {
        splitDepths[1], splitDepths[2], splitDepths[3], splitDepths[4]
    };

    // Reverse-Z camera: NDC near = 1.0, NDC far = 0.0.
    // Remap view-space split depths into NDC Z for frustum corner unprojection.
    for (uint32_t c = 0; c < CASCADE_COUNT; ++c) {
        const float cNear    = splitDepths[c];
        const float cFar     = splitDepths[c + 1];
        const float cNearNDC = glm::mix(0.0f, 1.0f, (cNear - n) / (f - n));  // farNDC=0, nearNDC=1
        const float cFarNDC  = glm::mix(0.0f, 1.0f, (cFar  - n) / (f - n));

        // 8 world-space corners of this cascade frustum slice
        // Note: cNearNDC > cFarNDC in reverse-Z, pass larger first
        const auto corners = frustumCornersWorld(m_viewMatrix, m_projMatrix,
                                                  cNearNDC, cFarNDC);

        // Bounding sphere: rotation-invariant, enables stable texel snapping
        auto [center, radius] = boundingSphere(corners);

        // Texel-snap center in light space to eliminate shadow shimmer on camera rotation
        const glm::mat4 lightViewSnap = glm::lookAt(center - lightDir, center, up);
        const float texelSize = (2.0f * radius) / CASCADE_SIZE;
        glm::vec4 centerLS = lightViewSnap * glm::vec4(center, 1.0f);
        centerLS.x = std::floor(centerLS.x / texelSize) * texelSize;
        centerLS.y = std::floor(centerLS.y / texelSize) * texelSize;
        const glm::vec3 snappedCenter = glm::vec3(glm::inverse(lightViewSnap) * centerLS);

        // Build final light matrices for this cascade
        const glm::mat4 finalLightView = glm::lookAt(
            snappedCenter - lightDir * radius,
            snappedCenter, up);
        glm::mat4 lightProj = glm::orthoRH_ZO(
            -radius, radius,
            -radius, radius,
            0.0f, 2.0f * radius);
        lightProj[1][1] *= -1.0f;  // Y-flip for Vulkan

        ubo.lightViewProj[c] = lightProj * finalLightView;
    }

    m_shadowUBOBuffer.upload(&ubo, sizeof(ShadowCascadeUBO));
}

// ── Frame loop ────────────────────────────────────────────────────────────────

void Renderer::handleResize()
{
    const VkExtent2D oldExt = m_swapchain.getExtent();

    // CRITICAL: drain the GPU before destroying any resources.
    // The just-submitted command buffer may still reference the depth image.
    // Swapchain::recreate() also calls vkDeviceWaitIdle() internally —
    // the second call is a no-op since the device is already idle.
    vkDeviceWaitIdle(m_ctx.getDevice());

    m_depthImage.destroy();
    m_hdrTarget.destroy();

    // Recreate swapchain. Passing current extent as hint; Swapchain::createSwapchain()
    // reads surface capabilities and uses cap.currentExtent when available (most platforms),
    // falling back to clamped requested dimensions otherwise.
    m_swapchain.recreate(oldExt.width, oldExt.height);

    const VkExtent2D newExt = m_swapchain.getExtent();
    if (newExt.width > 0 && newExt.height > 0) {
        createDepthImage();
        createHdrTarget();  // also calls updateTonemapDescriptorSet() to rebind new image view
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

    // Phase 3.7: update shadow matrices each frame so the shadow follows the camera.
    // Also recalculates shadowRadius/shadowDepth from m_sceneInfo.normalizedRadius.
    updateShadowMatrices();

    // ── Shadow pass (4 cascades) ──────────────────────────────────────────────
    // Transition all cascade layers from UNDEFINED to DEPTH_ATTACHMENT_OPTIMAL in one barrier.
    // VK_REMAINING_ARRAY_LAYERS in VulkanUtil covers the whole 2D array image.
    vkutil::transitionImage(cmd, m_shadowMap.getImage(),
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,              0,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    // VSM: transition moments from shader-read (resting state) to color attachment for writing.
    const bool isVsm = (m_shadowFilterMode == 2);
    if (isVsm) {
        vkutil::transitionImage(cmd, m_shadowMoments.getImage(),
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,          VK_ACCESS_2_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }

    m_gpuTimer.writeTimestamp(cmd, "ShadowPass_Begin");

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      isVsm ? m_shadowVsmPipeline : m_shadowPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                            0, 1, &m_descriptorSet, 0, nullptr);

    const VkBuffer     shadowVertexBuf = m_vertexBuffer.getBuffer();
    const VkDeviceSize zeroOffset      = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &shadowVertexBuf, &zeroOffset);
    vkCmdBindIndexBuffer(cmd, m_indexBuffer.getBuffer(), 0, VK_INDEX_TYPE_UINT32);

    const VkExtent2D shadowExtent{ CASCADE_SIZE, CASCADE_SIZE };
    const VkViewport shadowVP{
        .x = 0.0f, .y = 0.0f,
        .width  = static_cast<float>(CASCADE_SIZE),
        .height = static_cast<float>(CASCADE_SIZE),
        .minDepth = 0.0f, .maxDepth = 1.0f,
    };
    const VkRect2D shadowScissor{ .offset = { 0, 0 }, .extent = shadowExtent };

    for (uint32_t c = 0; c < CASCADE_COUNT; ++c) {
        const VkRenderingAttachmentInfo cascadeDepthAtt{
            .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView   = m_shadowLayerViews[c],
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue  = { .depthStencil = { 1.0f, 0 } },
        };
        // VSM: moments color attachment (one per cascade layer via per-layer view)
        const VkRenderingAttachmentInfo momentsColorAtt{
            .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView   = m_momentsLayerViews[c],
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue  = { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 0.0f } } },
        };
        const VkRenderingInfo cascadeRenderInfo{
            .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea           = { .offset = { 0, 0 }, .extent = shadowExtent },
            .layerCount           = 1,
            .colorAttachmentCount = isVsm ? 1u : 0u,
            .pColorAttachments    = isVsm ? &momentsColorAtt : nullptr,
            .pDepthAttachment     = &cascadeDepthAtt,
        };
        vkCmdBeginRendering(cmd, &cascadeRenderInfo);
        vkCmdSetViewport(cmd, 0, 1, &shadowVP);
        vkCmdSetScissor(cmd, 0, 1, &shadowScissor);
        vkCmdSetDepthBias(cmd, 1.5f, 0.0f, 2.0f);

        // Push cascade index (bytes 96–99): shadow.vert selects lightViewProj[cascadeIndex]
        vkCmdPushConstants(cmd, m_pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           96, sizeof(uint32_t), &c);

        for (const auto& meshData : m_meshRenderData) {
            vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                               0, sizeof(glm::mat4), &m_sceneInfo.modelMatrix);
            vkCmdDrawIndexed(cmd, meshData.indexCount, 1, meshData.firstIndex, 0, 0);
        }

        vkCmdEndRendering(cmd);
    }

    m_gpuTimer.writeTimestamp(cmd, "ShadowPass_End");

    // Transition all cascade layers to shader-read for the PBR pass.
    vkutil::transitionImage(cmd, m_shadowMap.getImage(),
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,           VK_ACCESS_2_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,          VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    // VSM: separable Gaussian blur on moment maps, then restore to shader-read layout.
    if (isVsm) {
        // Transition moments COLOR_ATTACHMENT → GENERAL (storage image for compute read+write)
        vkutil::transitionImage(cmd, m_shadowMoments.getImage(),
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,          VK_IMAGE_LAYOUT_GENERAL);

        // Transition temp UNDEFINED → GENERAL (discard-write is safe for temp buffer)
        vkutil::transitionImage(cmd, m_shadowMomentsTemp.getImage(),
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,  0,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,               VK_IMAGE_LAYOUT_GENERAL);

        m_gpuTimer.writeTimestamp(cmd, "BlurPass_Begin");

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_blurPipeline);

        // Horizontal pass: moments (read) → temp (write)
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_blurPipelineLayout, 0, 1, &m_blurSetHorizontal, 0, nullptr);
        struct BlurPC { int32_t direction; int32_t radius; };
        BlurPC blurPC{ 0, 3 };
        vkCmdPushConstants(cmd, m_blurPipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BlurPC), &blurPC);
        // dispatch: 2048/16=128 per axis, CASCADE_COUNT layers
        vkCmdDispatch(cmd, CASCADE_SIZE / 16, CASCADE_SIZE / 16, CASCADE_COUNT);

        // Execution+memory barrier: compute write (temp) must be visible to next compute read
        const VkMemoryBarrier2 computeBarrier{
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
            .srcStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
            .dstStageMask  = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
        };
        const VkDependencyInfo blurDepInfo{
            .sType              = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .memoryBarrierCount = 1,
            .pMemoryBarriers    = &computeBarrier,
        };
        vkCmdPipelineBarrier2(cmd, &blurDepInfo);

        // Vertical pass: temp (read) → moments (write)
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_blurPipelineLayout, 0, 1, &m_blurSetVertical, 0, nullptr);
        blurPC.direction = 1;
        vkCmdPushConstants(cmd, m_blurPipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BlurPC), &blurPC);
        vkCmdDispatch(cmd, CASCADE_SIZE / 16, CASCADE_SIZE / 16, CASCADE_COUNT);

        m_gpuTimer.writeTimestamp(cmd, "BlurPass_End");

        // Transition moments GENERAL → SHADER_READ_ONLY_OPTIMAL for PBR sampling
        vkutil::transitionImage(cmd, m_shadowMoments.getImage(),
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL,                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    // Phase 5: HDR target transitions from UNDEFINED every frame (LOAD_OP_CLEAR discards contents).
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

    VkPipeline activePipeline = m_pipeline;
    if (m_showNormals)
        activePipeline = m_normalsPipeline;
    else if (m_wireframe)
        activePipeline = m_wireframePipeline;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, activePipeline);

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

    // Bind camera UBO descriptor set (set=0, binding=0 — view/projection matrices)
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                            0, 1, &m_descriptorSet, 0, nullptr);

    // Bind vertex and index buffers
    const VkDeviceSize vertexOffset = 0;
    const VkBuffer vertexBuf = m_vertexBuffer.getBuffer();
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuf, &vertexOffset);
    vkCmdBindIndexBuffer(cmd, m_indexBuffer.getBuffer(), 0, VK_INDEX_TYPE_UINT32);

    // Push normalization model matrix (bytes 0–63): same transform as shadow pass.
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                       0, sizeof(glm::mat4), &m_sceneInfo.modelMatrix);

    for (const auto& mesh : m_meshRenderData) {
        if (mesh.materialIndex >= 0 &&
            mesh.materialIndex < static_cast<int32_t>(m_materialSets.size())) {

            // Bind material texture set (set=1) — set=0 remains bound and unaffected.
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    m_pipelineLayout,
                                    1, 1, &m_materialSets[static_cast<size_t>(mesh.materialIndex)],
                                    0, nullptr);

            // Push material factors to fragment stage (offset 64, after model matrix).
            const Material& mat = m_model.materials[static_cast<size_t>(mesh.materialIndex)];
            const MaterialPushConstants matPC{
                .baseColorFactor  = mat.baseColorFactor,
                .metallicFactor   = mat.metallicFactor,
                .roughnessFactor  = mat.roughnessFactor,
                .alphaCutoff      = (mat.alphaMode == Material::AlphaMode::Mask)
                                    ? mat.alphaCutoff : 0.0f,
            };
            vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                               64, sizeof(MaterialPushConstants), &matPC);
        } else {
            // No material — push default white/neutral factors.
            const MaterialPushConstants defaultPC{};
            vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                               64, sizeof(MaterialPushConstants), &defaultPC);
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

    const VkRenderingAttachmentInfo tonemapColorAtt{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = m_swapchain.getCurrentImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_DONT_CARE,  // fullscreen overwrite; no need to load
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
    };
    const VkRenderingInfo tonemapRenderInfo{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea           = { .offset = { 0, 0 }, .extent = ext },
        .layerCount           = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments    = &tonemapColorAtt,
        // No depth attachment — fullscreen blit doesn't depth-test
    };

    vkCmdBeginRendering(cmd, &tonemapRenderInfo);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_tonemapPipeline);

    const VkViewport tonemapVP{
        .x = 0.0f, .y = 0.0f,
        .width  = static_cast<float>(ext.width),
        .height = static_cast<float>(ext.height),
        .minDepth = 0.0f, .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmd, 0, 1, &tonemapVP);
    const VkRect2D tonemapScissor{ .offset = { 0, 0 }, .extent = ext };
    vkCmdSetScissor(cmd, 0, 1, &tonemapScissor);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            m_tonemapPipelineLayout, 0, 1, &m_tonemapSet, 0, nullptr);

    struct TonemapPC {
        uint32_t mode;
        float    exposure;
        uint32_t splitScreenMode;
        uint32_t splitRightMode;
    };
    const TonemapPC tonemapPC{
        static_cast<uint32_t>(m_tonemapMode),
        m_exposure,
        static_cast<uint32_t>(m_splitScreenMode),
        static_cast<uint32_t>(m_splitRightMode),
    };
    vkCmdPushConstants(cmd, m_tonemapPipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(TonemapPC), &tonemapPC);

    vkCmdDraw(cmd, 3, 1, 0, 0);  // 3 vertices; fullscreen triangle generated in tonemap.vert

    vkCmdEndRendering(cmd);

    m_gpuTimer.writeTimestamp(cmd, "TonemapPass_End");

    // Phase 2.5: ImGui overlay pass (LOAD_OP_LOAD preserves the tone-mapped image).
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

void Renderer::setCameraMatrices(const glm::mat4& view, const glm::mat4& projection,
                                  const glm::vec3& cameraPos)
{
    m_cameraPos  = cameraPos;
    m_viewMatrix = view;
    m_projMatrix = projection;
    const CameraUBO ubo{ view, projection, cameraPos, 0.0f };
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
