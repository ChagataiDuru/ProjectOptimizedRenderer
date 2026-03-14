#include "core/Renderer.h"
#include "resource/GLTFLoader.h"
#include "resource/Vertex.h"

#include <spdlog/spdlog.h>
#include <glm/glm.hpp>
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

// Synchronization2 image layout transition helper (core in Vulkan 1.3 / 1.4).
// aspectMask defaults to COLOR but must be VK_IMAGE_ASPECT_DEPTH_BIT for depth images.
static void transitionImage(
    VkCommandBuffer       cmd,
    VkImage               image,
    VkPipelineStageFlags2 srcStage,  VkAccessFlags2 srcAccess,
    VkPipelineStageFlags2 dstStage,  VkAccessFlags2 dstAccess,
    VkImageLayout         oldLayout, VkImageLayout  newLayout,
    VkImageAspectFlags    aspectMask = VK_IMAGE_ASPECT_COLOR_BIT)
{
    const VkImageMemoryBarrier2 barrier{
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask        = srcStage,
        .srcAccessMask       = srcAccess,
        .dstStageMask        = dstStage,
        .dstAccessMask       = dstAccess,
        .oldLayout           = oldLayout,
        .newLayout           = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = image,
        .subresourceRange    = { aspectMask, 0, 1, 0, 1 },
    };
    const VkDependencyInfo dep{
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &barrier,
    };
    vkCmdPipelineBarrier2(cmd, &dep);
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
    , m_depthImage(ctx)
{
}

Renderer::~Renderer()
{
    shutdown();
}

void Renderer::init()
{
    m_frameSync.init(3);
    m_commandBuffer.init(m_ctx.getGraphicsQueueFamily(), 3);
    loadModel(std::string(ASSET_DIR) + "/models/sponza.glb");
    createDepthImage();
    createPbrPipeline();       // also creates m_cameraSetLayout (bindings 0 + 1)
    createCameraUBO();
    createLightUBO();
    createDescriptorPool();
    createDescriptorSet();     // allocates from pool, binds camera (b0) + light (b1)
    spdlog::info("Renderer initialized (Phase 1.4: PBR shading + directional light)");
}

void Renderer::shutdown()
{
    if (m_ctx.getDevice() == VK_NULL_HANDLE) return;
    vkDeviceWaitIdle(m_ctx.getDevice());
    destroyPipeline();
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
        data.firstIndex = static_cast<uint32_t>(allIndices.size());
        data.indexCount = static_cast<uint32_t>(mesh.indices.size());
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

    vkDestroyFence(m_ctx.getDevice(), fence, nullptr);
    vkDestroyCommandPool(m_ctx.getDevice(), transferPool, nullptr);

    spdlog::info("Model uploaded: {} meshes, {} vertices, {} indices",
                 m_model.meshes.size(), allVertices.size(), allIndices.size());
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

void Renderer::createPbrPipeline()
{
    const std::string dir = SHADER_DIR;
    m_vertModule = makeShaderModule(m_ctx.getDevice(), loadSpv(dir + "/pbr.vert.spv"));
    m_fragModule = makeShaderModule(m_ctx.getDevice(), loadSpv(dir + "/pbr.frag.spv"));

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

    const VkPipelineRasterizationStateCreateInfo rasterization{
        .sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        // Back-face culling enabled: cube normals are outward-facing CCW, so back faces are CW.
        .cullMode    = VK_CULL_MODE_BACK_BIT,
        .frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE,
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

    // Descriptor set layout: set 0 holds both camera (binding 0) and light (binding 1) UBOs.
    const std::array<VkDescriptorSetLayoutBinding, 2> setBindings{{
        {   // binding 0: camera matrices — read in vertex shader
            .binding         = 0,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_VERTEX_BIT,
        },
        {   // binding 1: directional light — read in fragment shader
            .binding         = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
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

    // Push constant: per-object model matrix (64 bytes, vertex stage only)
    const VkPushConstantRange pushConstant{
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .offset     = 0,
        .size       = sizeof(glm::mat4),
    };

    const VkPipelineLayoutCreateInfo layoutInfo{
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = &m_cameraSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges    = &pushConstant,
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

    VK_CHECK(vkCreateGraphicsPipelines(
        m_ctx.getDevice(), VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline));

    // Shader modules are only needed during pipeline compilation — free them immediately
    vkDestroyShaderModule(m_ctx.getDevice(), m_vertModule, nullptr); m_vertModule = VK_NULL_HANDLE;
    vkDestroyShaderModule(m_ctx.getDevice(), m_fragModule, nullptr); m_fragModule = VK_NULL_HANDLE;

    spdlog::info("PBR pipeline created (pbr.vert + pbr.frag)");
}

void Renderer::destroyPipeline()
{
    const VkDevice dev = m_ctx.getDevice();
    // Descriptor pool destruction implicitly frees all sets allocated from it.
    if (m_descriptorPool  != VK_NULL_HANDLE) { vkDestroyDescriptorPool(dev, m_descriptorPool, nullptr);      m_descriptorPool  = VK_NULL_HANDLE; m_descriptorSet = VK_NULL_HANDLE; }
    if (m_vertModule      != VK_NULL_HANDLE) { vkDestroyShaderModule(dev, m_vertModule, nullptr);             m_vertModule      = VK_NULL_HANDLE; }
    if (m_fragModule      != VK_NULL_HANDLE) { vkDestroyShaderModule(dev, m_fragModule, nullptr);             m_fragModule      = VK_NULL_HANDLE; }
    if (m_pipeline        != VK_NULL_HANDLE) { vkDestroyPipeline(dev, m_pipeline, nullptr);                   m_pipeline        = VK_NULL_HANDLE; }
    if (m_pipelineLayout  != VK_NULL_HANDLE) { vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);      m_pipelineLayout  = VK_NULL_HANDLE; }
    if (m_cameraSetLayout != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(dev, m_cameraSetLayout, nullptr); m_cameraSetLayout = VK_NULL_HANDLE; }
}

// ── Camera UBO ────────────────────────────────────────────────────────────────

void Renderer::createCameraUBO()
{
    // Host-visible + persistently mapped: upload is a memcpy, no staging needed.
    // UBOs change every frame so device-local + staging would be needlessly expensive.
    m_cameraUBOBuffer.createHostVisible(sizeof(CameraUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    const CameraUBO initial{ glm::mat4(1.0f), glm::mat4(1.0f) };
    m_cameraUBOBuffer.upload(&initial, sizeof(CameraUBO));

    spdlog::info("Camera UBO created ({} bytes, host-visible)", sizeof(CameraUBO));
}

void Renderer::createLightUBO()
{
    m_lightUBOBuffer.createHostVisible(sizeof(LightUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

    // Default sun: slightly off-vertical, warm white, moderate ambient
    setLightParameters(
        glm::normalize(glm::vec3(0.5f, 1.0f, 0.3f)),
        glm::vec3(1.0f),
        1.0f,
        0.2f);

    spdlog::info("Light UBO created ({} bytes, host-visible)", sizeof(LightUBO));
}

void Renderer::setLightParameters(const glm::vec3& direction, const glm::vec3& color,
                                   float intensity, float ambient)
{
    const LightUBO light{
        glm::normalize(direction),
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
    // 2 uniform buffer descriptors: camera (binding 0) + light (binding 1)
    const VkDescriptorPoolSize poolSize{
        .type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 2,
    };
    const VkDescriptorPoolCreateInfo poolCI{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = 1,
        .poolSizeCount = 1,
        .pPoolSizes    = &poolSize,
    };
    VK_CHECK(vkCreateDescriptorPool(m_ctx.getDevice(), &poolCI, nullptr, &m_descriptorPool));
    spdlog::info("Descriptor pool created (2 uniform buffers: camera + light)");
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

    const std::array<VkWriteDescriptorSet, 2> writes{{
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
    }};
    vkUpdateDescriptorSets(m_ctx.getDevice(),
                           static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    spdlog::info("Descriptor set allocated: camera UBO (binding=0), light UBO (binding=1)");
}

// ── Frame loop ────────────────────────────────────────────────────────────────

void Renderer::beginFrame()
{
    m_frameSync.waitForFrame();
    m_frameSync.resetFence();
    m_commandBuffer.resetFrame();

    if (!m_swapchain.acquireNextImage(m_frameSync.getImageAvailableSemaphore()))
        spdlog::warn("Swapchain out of date on acquire — resize handling in next phase");
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

    // Swapchain images start in UNDEFINED layout each frame — transition to writable attachment.
    // UNDEFINED oldLayout means the driver is free to discard contents (correct, we clear anyway).
    transitionImage(cmd, m_swapchain.getCurrentImage(),
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,             0,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    // Depth image transitions from UNDEFINED each frame — LOAD_OP_CLEAR discards previous
    // contents anyway, so UNDEFINED→DEPTH_ATTACHMENT_OPTIMAL is valid and avoids layout tracking.
    transitionImage(cmd, m_depthImage.getImage(),
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

    vkCmdBeginRendering(cmd, &renderingInfo);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

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

    // Push model matrix — one identity matrix for the whole scene (Phase 1.3).
    const glm::mat4 modelMatrix(1.0f);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT,
                       0, sizeof(glm::mat4), &modelMatrix);

    for (const auto& mesh : m_meshRenderData) {
        vkCmdDrawIndexed(cmd, mesh.indexCount, 1, mesh.firstIndex, 0, 0);
    }

    vkCmdEndRendering(cmd);

    // Transition to PRESENT_SRC_KHR so the presentation engine can consume the image.
    // Without this barrier, MoltenVK's Metal presentation layer sees an incorrect layout.
    transitionImage(cmd, m_swapchain.getCurrentImage(),
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,           0,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,          VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VK_CHECK(vkEndCommandBuffer(cmd));
}

void Renderer::setCameraMatrices(const glm::mat4& view, const glm::mat4& projection)
{
    const CameraUBO ubo{ view, projection };
    m_cameraUBOBuffer.upload(&ubo, sizeof(CameraUBO));
}

void Renderer::endFrame()
{
    render();

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
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR)
        spdlog::warn("Swapchain suboptimal/out-of-date on present — resize handling in next phase");
    else if (presentResult != VK_SUCCESS)
        throw std::runtime_error("vkQueuePresentKHR failed: " +
                                 std::to_string(static_cast<int>(presentResult)));

    // Advance both frame indices together to keep sync primitives and command buffers aligned
    m_frameSync.advanceFrame();
    m_commandBuffer.beginFrame();
}
