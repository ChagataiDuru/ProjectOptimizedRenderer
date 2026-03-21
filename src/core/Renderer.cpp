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
    , m_shadowMap(ctx)
    , m_shadowUBOBuffer(ctx)
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
    destroyPipeline();

    // Destroy textures before VMA teardown (textures hold VmaAllocations)
    m_fallbackWhite.destroy();
    for (auto& tex : m_textures)
        tex.destroy();
    m_textures.clear();
    m_samplerCache.shutdown();

    // Destroy shadow resources
    if (m_shadowSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_ctx.getDevice(), m_shadowSampler, nullptr);
        m_shadowSampler = VK_NULL_HANDLE;
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

    // Descriptor set layout: set 0 — camera (b0), light (b1), shadow UBO (b2), shadow map (b3).
    const std::array<VkDescriptorSetLayoutBinding, 4> setBindings{{
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
        {   // binding 2: shadow UBO (lightViewProj) — vertex (shadow pass) + fragment (PBR)
            .binding         = 2,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {   // binding 3: shadow map (comparison sampler) — fragment shader only
            .binding         = 3,
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

    // Two non-overlapping push constant ranges:
    //   bytes  0–63:  mat4 model matrix (vertex stage)
    //   bytes 64–95:  MaterialPushConstants — factors (fragment stage)
    const std::array<VkPushConstantRange, 2> pushRanges{{
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
    // Must declare the depth format here so the pipeline is compatible with the render attachment.
    const VkFormat colorFormat = m_swapchain.getFormat();
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
    m_lightDirection = glm::normalize(direction);

    const LightUBO light{
        m_lightDirection,
        intensity,
        color,
        ambient,
    };
    m_lightUBOBuffer.upload(&light, sizeof(LightUBO));
    spdlog::debug("Light updated: dir=({:.2f},{:.2f},{:.2f}), intensity={:.2f}",
                  direction.x, direction.y, direction.z, intensity);
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

void Renderer::createDescriptorPool()
{
    // Pool must hold:
    //   1 set  × 3 uniform buffers (camera + light + shadow, set=0)
    //   1 set  × 1 combined image sampler (shadow map, set=0)
    //   N sets × 3 combined image samplers each (material textures, set=1)
    const uint32_t materialCount = static_cast<uint32_t>(m_model.materials.size());
    const uint32_t maxSets       = 1 + std::max(materialCount, 1u);
    const uint32_t samplerCount  = 1 + 3 * std::max(materialCount, 1u);  // shadow map + materials

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
        .range  = sizeof(ShadowUBO),
    };
    const VkDescriptorImageInfo shadowMapInfo{
        .sampler     = m_shadowSampler,
        .imageView   = m_shadowMap.getImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
    };

    const std::array<VkWriteDescriptorSet, 4> writes{{
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
    }};
    vkUpdateDescriptorSets(m_ctx.getDevice(),
                           static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    spdlog::info("Descriptor set allocated: camera(b0), light(b1), shadowUBO(b2), shadowMap(b3)");
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

void Renderer::createShadowResources()
{
    const VkDevice dev = m_ctx.getDevice();

    // ── Shadow depth image ────────────────────────────────────────────────────
    // Standard Z (not reverse-Z): shadow map is compared with LESS_OR_EQUAL.
    // Usage: DEPTH_STENCIL_ATTACHMENT (shadow pass) + SAMPLED (PBR pass reads it).
    m_shadowMap.create(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1,
                       VK_FORMAT_D32_SFLOAT,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                       VK_IMAGE_ASPECT_DEPTH_BIT);

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

    // ── Shadow UBO (lightViewProj matrix) ─────────────────────────────────────
    m_shadowUBOBuffer.createHostVisible(sizeof(ShadowUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
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

    // Shadow module no longer needed after pipeline creation.
    vkDestroyShaderModule(dev, m_shadowVertModule, nullptr);
    m_shadowVertModule = VK_NULL_HANDLE;

    spdlog::info("Shadow resources created: {}×{} depth map, shadow pipeline",
                 SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
}

void Renderer::updateShadowMatrices()
{
    const glm::vec3 lightDir = glm::normalize(m_lightDirection);

    // Camera-centric: shadow map covers a radius proportional to the normalized scene.
    // Phase 3.7: use normalizedRadius so these values are consistent across all models.
    const float shadowRadius = m_sceneInfo.normalizedRadius * 3.0f;
    const float shadowDepth  = m_sceneInfo.normalizedRadius * 20.0f;

    // Use camera position as shadow focus (projected onto ground if desired)
    const glm::vec3 focusPoint = m_cameraPos;

    // Light eye: pull back along light direction from focus
    const glm::vec3 lightEye = focusPoint + lightDir * (shadowDepth * 0.5f);

    const glm::mat4 lightView = glm::lookAt(lightEye, focusPoint, glm::vec3(0.0f, 1.0f, 0.0f));

    glm::mat4 lightProj = glm::orthoRH_ZO(
        -shadowRadius, shadowRadius,
        -shadowRadius, shadowRadius,
        0.1f, shadowDepth);

    // Y-flip for Vulkan coordinate system
    lightProj[1][1] *= -1.0f;

    const ShadowUBO shadowUBO{ lightProj * lightView };
    m_shadowUBOBuffer.upload(&shadowUBO, sizeof(ShadowUBO));
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

    // Recreate swapchain. Passing current extent as hint; Swapchain::createSwapchain()
    // reads surface capabilities and uses cap.currentExtent when available (most platforms),
    // falling back to clamped requested dimensions otherwise.
    m_swapchain.recreate(oldExt.width, oldExt.height);

    const VkExtent2D newExt = m_swapchain.getExtent();
    if (newExt.width > 0 && newExt.height > 0) {
        createDepthImage();
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

    // ── Shadow pass ───────────────────────────────────────────────────────────
    // Transition shadow map from its previous state to DEPTH_ATTACHMENT_OPTIMAL.
    // First frame layout is UNDEFINED (no previous contents to preserve).
    vkutil::transitionImage(cmd, m_shadowMap.getImage(),
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,              0,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    m_gpuTimer.writeTimestamp(cmd, "ShadowPass_Begin");

    const VkRenderingAttachmentInfo shadowDepthAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = m_shadowMap.getImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = { .depthStencil = { 1.0f, 0 } },  // standard Z: clear to 1.0 (far)
    };
    const VkExtent2D shadowExtent{ SHADOW_MAP_SIZE, SHADOW_MAP_SIZE };
    const VkRenderingInfo shadowRenderingInfo{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea           = { .offset = { 0, 0 }, .extent = shadowExtent },
        .layerCount           = 1,
        .colorAttachmentCount = 0,
        .pColorAttachments    = nullptr,
        .pDepthAttachment     = &shadowDepthAttachment,
    };
    vkCmdBeginRendering(cmd, &shadowRenderingInfo);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline);

    const VkViewport shadowViewport{
        .x = 0.0f, .y = 0.0f,
        .width  = static_cast<float>(SHADOW_MAP_SIZE),
        .height = static_cast<float>(SHADOW_MAP_SIZE),
        .minDepth = 0.0f, .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmd, 0, 1, &shadowViewport);
    const VkRect2D shadowScissor{ .offset = { 0, 0 }, .extent = shadowExtent };
    vkCmdSetScissor(cmd, 0, 1, &shadowScissor);
    // Dynamic depth bias (matches createShadowResources() constants but overridable)
    vkCmdSetDepthBias(cmd, 1.5f, 0.0f, 2.0f);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                            0, 1, &m_descriptorSet, 0, nullptr);

    const VkBuffer     shadowVertexBuf = m_vertexBuffer.getBuffer();
    const VkDeviceSize zeroOffset      = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &shadowVertexBuf, &zeroOffset);
    vkCmdBindIndexBuffer(cmd, m_indexBuffer.getBuffer(), 0, VK_INDEX_TYPE_UINT32);

    for (const auto& meshData : m_meshRenderData) {
        // Push normalization model matrix (bytes 0–63): centers + scales the model.
        vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                           0, sizeof(glm::mat4), &m_sceneInfo.modelMatrix);
        vkCmdDrawIndexed(cmd, meshData.indexCount, 1, meshData.firstIndex, 0, 0);
    }

    vkCmdEndRendering(cmd);

    m_gpuTimer.writeTimestamp(cmd, "ShadowPass_End");

    // Transition shadow map to shader-readable layout for the PBR pass.
    // STORE_OP_STORE above ensures depth values are preserved after rendering.
    vkutil::transitionImage(cmd, m_shadowMap.getImage(),
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,           VK_ACCESS_2_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,          VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    // Swapchain images start in UNDEFINED layout each frame — transition to writable attachment.
    // UNDEFINED oldLayout means the driver is free to discard contents (correct, we clear anyway).
    vkutil::transitionImage(cmd, m_swapchain.getCurrentImage(),
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,             0,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Depth image transitions from UNDEFINED each frame — LOAD_OP_CLEAR discards previous
    // contents anyway, so UNDEFINED→DEPTH_ATTACHMENT_OPTIMAL is valid and avoids layout tracking.
    vkutil::transitionImage(cmd, m_depthImage.getImage(),
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,              0,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    const VkRenderingAttachmentInfo colorAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = m_swapchain.getCurrentImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = { .color = { .float32 = { 0.01f, 0.01f, 0.01f, 1.0f } } },
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

    // Phase 2.5: ImGui overlay pass (LOAD_OP_LOAD preserves the PBR scene).
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
    m_cameraPos = cameraPos;
    const CameraUBO ubo{ view, projection, cameraPos, 0.0f };
    m_cameraUBOBuffer.upload(&ubo, sizeof(CameraUBO));
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
