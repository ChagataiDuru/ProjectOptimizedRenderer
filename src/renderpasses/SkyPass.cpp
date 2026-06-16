#include "renderpasses/SkyPass.h"

#include "core/ShaderInterface.h"
#include "core/VulkanUtil.h"

#include <spdlog/spdlog.h>
#include <stb_image.h>

#include <array>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<uint32_t> loadSpv(const std::string& path)
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

VkShaderModule makeShaderModule(VkDevice device, const std::vector<uint32_t>& code)
{
    const VkShaderModuleCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size() * sizeof(uint32_t),
        .pCode = code.data(),
    };
    VkShaderModule mod = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &info, nullptr, &mod));
    return mod;
}

void writePanoramaDescriptor(VkDevice dev, VkDescriptorSet set,
                             VkSampler sampler, VkImageView imageView)
{
    const VkDescriptorImageInfo imgInfo{
        .sampler = sampler,
        .imageView = imageView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = set,
        .dstBinding = shader_interface::sky_binding::kPanorama,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &imgInfo,
    };
    vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);
}

VkPipeline createFullscreenPipeline(
    VkDevice dev,
    const std::array<VkPipelineShaderStageCreateInfo, 2>& stages,
    VkPipelineLayout layout,
    VkFormat colorFormat,
    VkFormat depthFormat,
    VkSampleCountFlagBits sceneSamples,
    bool sampleShadingEnabled,
    float minSampleShading)
{
    const VkPipelineVertexInputStateCreateInfo vertexInput{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
    };
    const VkPipelineInputAssemblyStateCreateInfo inputAssembly{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };
    const VkPipelineViewportStateCreateInfo viewportState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };
    const VkPipelineRasterizationStateCreateInfo rasterization{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE,
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .lineWidth = 1.0f,
    };
    const VkPipelineMultisampleStateCreateInfo multisample{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = sceneSamples,
        .sampleShadingEnable = sampleShadingEnabled ? VK_TRUE : VK_FALSE,
        .minSampleShading = sampleShadingEnabled ? minSampleShading : 0.0f,
    };
    const VkPipelineDepthStencilStateCreateInfo depthStencil{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL,
    };
    const VkPipelineColorBlendAttachmentState blendAtt{
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                          VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
    };
    const VkPipelineColorBlendStateCreateInfo colorBlend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &blendAtt,
    };
    const std::array<VkDynamicState, 2> dynStates = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    const VkPipelineDynamicStateCreateInfo dynamicState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(dynStates.size()),
        .pDynamicStates = dynStates.data(),
    };
    const VkPipelineRenderingCreateInfo renderingInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &colorFormat,
        .depthAttachmentFormat = depthFormat,
        .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
    };
    const VkGraphicsPipelineCreateInfo pipelineCI{
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &renderingInfo,
        .stageCount = static_cast<uint32_t>(stages.size()),
        .pStages = stages.data(),
        .pVertexInputState = &vertexInput,
        .pInputAssemblyState = &inputAssembly,
        .pViewportState = &viewportState,
        .pRasterizationState = &rasterization,
        .pMultisampleState = &multisample,
        .pDepthStencilState = &depthStencil,
        .pColorBlendState = &colorBlend,
        .pDynamicState = &dynamicState,
        .layout = layout,
        .renderPass = VK_NULL_HANDLE,
    };

    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &pipeline));
    return pipeline;
}

} // namespace

SkyPass::SkyPass(VulkanContext& ctx)
    : m_panorama(ctx)
{
}

void SkyPass::init(VulkanContext& ctx,
                   VkDescriptorSetLayout sceneSetLayout,
                   VkFormat colorFormat,
                   VkFormat depthFormat,
                   VkSampleCountFlagBits sceneSamples,
                   bool sampleShadingEnabled,
                   float minSampleShading,
                   const std::string& shaderDir)
{
    const VkDevice dev = ctx.getDevice();
    m_colorFormat = colorFormat;
    m_depthFormat = depthFormat;
    m_shaderDir = shaderDir;

    const VkDescriptorSetLayoutBinding binding{
        .binding = shader_interface::sky_binding::kPanorama,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
    };
    const VkDescriptorSetLayoutCreateInfo setLayoutCI{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &binding,
    };
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &setLayoutCI, nullptr, &m_panoramaSetLayout));

    const VkDescriptorPoolSize poolSize{
        .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 1,
    };
    const VkDescriptorPoolCreateInfo poolCI{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &poolSize,
    };
    VK_CHECK(vkCreateDescriptorPool(dev, &poolCI, nullptr, &m_panoramaPool));

    const VkDescriptorSetAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = m_panoramaPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &m_panoramaSetLayout,
    };
    VK_CHECK(vkAllocateDescriptorSets(dev, &allocInfo, &m_panoramaSet));

    const VkSamplerCreateInfo samplerCI{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .maxAnisotropy = 1.0f,
        .minLod = 0.0f,
        .maxLod = 1.0f,
    };
    VK_CHECK(vkCreateSampler(dev, &samplerCI, nullptr, &m_panoramaSampler));

    const std::array<float, 4> whitePanorama = { 1.0f, 1.0f, 1.0f, 1.0f };
    VkCommandPool uploadPool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = vkutil::beginSingleUseCommands(ctx, uploadPool);
    m_panorama.createFromData(1, 1, VK_FORMAT_R32G32B32A32_SFLOAT,
                              whitePanorama.data(),
                              static_cast<VkDeviceSize>(whitePanorama.size() * sizeof(float)),
                              cmd);
    vkutil::endSingleUseCommands(ctx, uploadPool, cmd);
    m_panorama.releaseStaging();
    writePanoramaDescriptor(dev, m_panoramaSet, m_panoramaSampler, m_panorama.getImageView());

    const VkPushConstantRange pcRange{
        .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        .offset = shader_interface::kSkyPcOffset,
        .size = sizeof(uint32_t),
    };
    const std::array<VkDescriptorSetLayout, 2> skyLayouts = {
        sceneSetLayout,
        m_panoramaSetLayout,
    };
    const VkPipelineLayoutCreateInfo layoutCI{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = static_cast<uint32_t>(skyLayouts.size()),
        .pSetLayouts = skyLayouts.data(),
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &pcRange,
    };
    VK_CHECK(vkCreatePipelineLayout(dev, &layoutCI, nullptr, &m_pipelineLayout));

    recreatePipeline(ctx, sceneSamples, sampleShadingEnabled, minSampleShading);

    spdlog::info("Sky pipeline created (procedural Rayleigh+Mie + HDR equirectangular panorama)");
}

void SkyPass::recreatePipeline(VulkanContext& ctx,
                               VkSampleCountFlagBits sceneSamples,
                               bool sampleShadingEnabled,
                               float minSampleShading)
{
    if (m_pipelineLayout == VK_NULL_HANDLE || m_shaderDir.empty()) {
        return;
    }

    const VkDevice dev = ctx.getDevice();
    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }

    m_sceneSamples = sceneSamples;
    m_sampleShadingEnabled = sampleShadingEnabled;
    m_minSampleShading = sampleShadingEnabled ? minSampleShading : 0.0f;

    VkShaderModule vertMod = makeShaderModule(dev, loadSpv(m_shaderDir + "/sky.vert.spv"));
    VkShaderModule fragMod = makeShaderModule(dev, loadSpv(m_shaderDir + "/sky.frag.spv"));

    const std::array<VkPipelineShaderStageCreateInfo, 2> stages{{
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vertMod, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragMod, .pName = "main" },
    }};

    m_pipeline = createFullscreenPipeline(
        dev,
        stages,
        m_pipelineLayout,
        m_colorFormat,
        m_depthFormat,
        m_sceneSamples,
        m_sampleShadingEnabled,
        m_minSampleShading);

    vkDestroyShaderModule(dev, vertMod, nullptr);
    vkDestroyShaderModule(dev, fragMod, nullptr);
}

void SkyPass::record(VkCommandBuffer cmd,
                     VkDescriptorSet sceneSet,
                     VkExtent2D extent,
                     bool enabled,
                     int32_t mode) const
{
    if (!enabled || m_pipeline == VK_NULL_HANDLE)
        return;

    const VkViewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(extent.width),
        .height = static_cast<float>(extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    const VkRect2D scissor{ .offset = { 0, 0 }, .extent = extent };
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                            shader_interface::kSceneSet, 1, &sceneSet, 0, nullptr);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                            shader_interface::kSkyPanoramaSet, 1, &m_panoramaSet, 0, nullptr);
    const uint32_t skyMode = static_cast<uint32_t>(mode);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT,
                       shader_interface::kSkyPcOffset, sizeof(uint32_t), &skyMode);
    vkCmdDraw(cmd, 3, 1, 0, 0);
}

void SkyPass::loadHdrPanorama(VulkanContext& ctx, const std::string& path)
{
    vkDeviceWaitIdle(ctx.getDevice());

    int w = 0;
    int h = 0;
    int channels = 0;
    float* pixels = stbi_loadf(path.c_str(), &w, &h, &channels, 4);
    if (!pixels) {
        spdlog::error("HDR panorama load failed '{}': {}", path, stbi_failure_reason());
        return;
    }

    const VkDeviceSize dataSize = static_cast<VkDeviceSize>(w) * h * 4 * sizeof(float);

    VkCommandPool uploadPool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = vkutil::beginSingleUseCommands(ctx, uploadPool);

    m_panorama.destroy();
    m_panorama.createFromData(static_cast<uint32_t>(w), static_cast<uint32_t>(h),
                              VK_FORMAT_R32G32B32A32_SFLOAT, pixels, dataSize, cmd);

    stbi_image_free(pixels);

    vkutil::endSingleUseCommands(ctx, uploadPool, cmd);

    m_panorama.releaseStaging();
    writePanoramaDescriptor(ctx.getDevice(), m_panoramaSet,
                            m_panoramaSampler, m_panorama.getImageView());

    spdlog::info("HDR panorama loaded: {}x{} from '{}'", w, h, path);
}

void SkyPass::shutdown(VkDevice device)
{
    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
    if (m_panoramaPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, m_panoramaPool, nullptr);
        m_panoramaPool = VK_NULL_HANDLE;
        m_panoramaSet = VK_NULL_HANDLE;
    }
    if (m_panoramaSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, m_panoramaSetLayout, nullptr);
        m_panoramaSetLayout = VK_NULL_HANDLE;
    }
    if (m_panoramaSampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, m_panoramaSampler, nullptr);
        m_panoramaSampler = VK_NULL_HANDLE;
    }
    m_panorama.destroy();
}
