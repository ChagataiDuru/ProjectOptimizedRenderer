#include "renderpasses/GtaoPass.h"

#include "core/VulkanUtil.h"
#include "debug/GPUTimer.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace {

constexpr VkFormat kAoFormat = VK_FORMAT_R8_UNORM;

struct GtaoPC {
    glm::mat4 inverseProjection;
    glm::vec4 outputSizeAndInvFull;
    glm::vec4 params;
};

struct BlurPC {
    glm::vec4 sizeAndDirection;
};

static_assert(sizeof(GtaoPC) == 96);
static_assert(sizeof(BlurPC) == 16);

VkShaderModule makeShaderModule(VkDevice device, const std::vector<uint32_t>& code)
{
    const VkShaderModuleCreateInfo info{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.size() * sizeof(uint32_t),
        .pCode = code.data(),
    };
    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &info, nullptr, &module));
    return module;
}

void transitionAo(VkCommandBuffer cmd, VkImage image,
                  VkImageLayout oldLayout, VkImageLayout newLayout)
{
    auto stageAccess = [](VkImageLayout layout) -> std::pair<VkPipelineStageFlags2, VkAccessFlags2> {
        switch (layout) {
            case VK_IMAGE_LAYOUT_GENERAL:
                return { VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT };
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                return { VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT };
            default:
                return { VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0 };
        }
    };
    const auto [srcStage, srcAccess] = stageAccess(oldLayout);
    const auto [dstStage, dstAccess] = stageAccess(newLayout);
    vkutil::transitionImage(cmd, image, srcStage, srcAccess, dstStage, dstAccess,
                            oldLayout, newLayout);
}

uint32_t groupCount(uint32_t size)
{
    return (size + 7) / 8;
}

} // namespace

GtaoPass::GtaoPass(VulkanContext& ctx)
    : m_ctx(ctx)
    , m_aoRaw(ctx)
    , m_aoBlurred(ctx)
    , m_aoTemp(ctx)
{
}

GtaoPass::~GtaoPass()
{
    shutdown(m_ctx.getDevice());
}

std::vector<uint32_t> GtaoPass::loadSpv(const std::string& path)
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

void GtaoPass::init(const std::string& shaderDir)
{
    const VkDevice dev = m_ctx.getDevice();
    m_shaderDir = shaderDir;

    const std::array<VkDescriptorSetLayoutBinding, 2> bindings{{
        { .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
    }};
    const VkDescriptorSetLayoutCreateInfo setLayoutCI{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(bindings.size()),
        .pBindings = bindings.data(),
    };
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &setLayoutCI, nullptr, &m_setLayout));

    const std::array<VkDescriptorPoolSize, 2> poolSizes{{
        { .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 3 },
        { .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 3 },
    }};
    const VkDescriptorPoolCreateInfo poolCI{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 3,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };
    VK_CHECK(vkCreateDescriptorPool(dev, &poolCI, nullptr, &m_pool));

    const std::array<VkDescriptorSetLayout, 3> layouts{ m_setLayout, m_setLayout, m_setLayout };
    std::array<VkDescriptorSet, 3> sets{};
    const VkDescriptorSetAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = m_pool,
        .descriptorSetCount = static_cast<uint32_t>(layouts.size()),
        .pSetLayouts = layouts.data(),
    };
    VK_CHECK(vkAllocateDescriptorSets(dev, &allocInfo, sets.data()));
    m_gtaoSet = sets[0];
    m_blurHorizontalSet = sets[1];
    m_blurVerticalSet = sets[2];

    const VkSamplerCreateInfo samplerCI{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .maxAnisotropy = 1.0f,
        .maxLod = 0.0f,
    };
    VK_CHECK(vkCreateSampler(dev, &samplerCI, nullptr, &m_aoSampler));

    auto createPipeline = [&](const std::string& path, uint32_t pushSize,
                              VkPipelineLayout& outLayout) {
        const VkPushConstantRange range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = pushSize };
        const VkPipelineLayoutCreateInfo layoutCI{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &m_setLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        };
        VK_CHECK(vkCreatePipelineLayout(dev, &layoutCI, nullptr, &outLayout));
        VkShaderModule module = makeShaderModule(dev, loadSpv(path));
        const VkComputePipelineCreateInfo pipelineCI{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = module,
                .pName = "main",
            },
            .layout = outLayout,
        };
        VkPipeline pipeline = VK_NULL_HANDLE;
        VK_CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pipelineCI, nullptr, &pipeline));
        vkDestroyShaderModule(dev, module, nullptr);
        return pipeline;
    };
    m_gtaoPipeline = createPipeline(shaderDir + "/gtao.comp.spv", sizeof(GtaoPC), m_gtaoLayout);
    m_blurPipeline = createPipeline(shaderDir + "/gtao_blur.comp.spv", sizeof(BlurPC), m_blurLayout);

    spdlog::info("GTAO pass created (half-resolution R8 ambient occlusion)");
}

void GtaoPass::destroyTargets()
{
    m_aoRaw.destroy();
    m_aoTemp.destroy();
    m_aoBlurred.destroy();
}

void GtaoPass::writeSets(VkImageView sceneDepthView, VkSampler sceneDepthSampler)
{
    auto write = [&](VkDescriptorSet set, VkImageView storageView,
                     VkImageView sampledView, VkSampler sampler) {
        const VkDescriptorImageInfo storageInfo{
            .imageView = storageView, .imageLayout = VK_IMAGE_LAYOUT_GENERAL };
        const VkDescriptorImageInfo sampledInfo{
            .sampler = sampler, .imageView = sampledView,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
        const std::array<VkWriteDescriptorSet, 2> writes{{
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0,
              .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              .pImageInfo = &storageInfo },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1,
              .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
              .pImageInfo = &sampledInfo },
        }};
        vkUpdateDescriptorSets(m_ctx.getDevice(), static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    };

    write(m_gtaoSet, m_aoRaw.getImageView(), sceneDepthView, sceneDepthSampler);
    write(m_blurHorizontalSet, m_aoTemp.getImageView(), m_aoRaw.getImageView(), m_aoSampler);
    write(m_blurVerticalSet, m_aoBlurred.getImageView(), m_aoTemp.getImageView(), m_aoSampler);
}

void GtaoPass::resize(uint32_t fullWidth, uint32_t fullHeight, VkImageView sceneDepthView,
                      VkSampler sceneDepthSampler)
{
    if (fullWidth == 0 || fullHeight == 0 || sceneDepthView == VK_NULL_HANDLE) {
        return;
    }
    destroyTargets();

    m_fullWidth = fullWidth;
    m_fullHeight = fullHeight;
    m_width = std::max(fullWidth / 2, 1u);
    m_height = std::max(fullHeight / 2, 1u);

    const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    m_aoRaw.create(m_width, m_height, 1, kAoFormat, usage);
    m_aoTemp.create(m_width, m_height, 1, kAoFormat, usage);
    m_aoBlurred.create(m_width, m_height, 1, kAoFormat, usage);

    // Start in SHADER_READ_ONLY so the scene set can bind the image before the first
    // AO dispatch (a frame with AO disabled never writes it).
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = vkutil::beginSingleUseCommands(m_ctx, pool);
    for (Image* image : { &m_aoRaw, &m_aoTemp, &m_aoBlurred }) {
        transitionAo(cmd, image->getImage(), VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    vkutil::endSingleUseCommands(m_ctx, pool, cmd);

    writeSets(sceneDepthView, sceneDepthSampler);
}

void GtaoPass::record(VkCommandBuffer cmd, const glm::mat4& inverseProjection,
                      float worldRadius, float intensity, GPUTimer* timer)
{
    if (!isReady() || m_gtaoPipeline == VK_NULL_HANDLE) {
        return;
    }
    if (timer) timer->writeTimestamp(cmd, "Gtao_Begin");

    const GtaoPC gtaoPC{
        .inverseProjection = inverseProjection,
        .outputSizeAndInvFull = glm::vec4(static_cast<float>(m_width), static_cast<float>(m_height),
                                          1.0f / static_cast<float>(m_fullWidth),
                                          1.0f / static_cast<float>(m_fullHeight)),
        // Projection scale for the UV radius: P00 and |P11| recovered from the inverse.
        .params = glm::vec4(worldRadius, intensity,
                            1.0f / std::max(std::abs(inverseProjection[0][0]), 1e-6f),
                            1.0f / std::max(std::abs(inverseProjection[1][1]), 1e-6f)),
    };

    transitionAo(cmd, m_aoRaw.getImage(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_gtaoPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_gtaoLayout,
                            0, 1, &m_gtaoSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_gtaoLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gtaoPC), &gtaoPC);
    vkCmdDispatch(cmd, groupCount(m_width), groupCount(m_height), 1);
    transitionAo(cmd, m_aoRaw.getImage(), VK_IMAGE_LAYOUT_GENERAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_blurPipeline);
    const auto blurAxis = [&](VkDescriptorSet set, Image& target, const glm::vec2& direction) {
        transitionAo(cmd, target.getImage(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
        const BlurPC blurPC{ glm::vec4(static_cast<float>(m_width), static_cast<float>(m_height),
                                       direction.x, direction.y) };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_blurLayout,
                                0, 1, &set, 0, nullptr);
        vkCmdPushConstants(cmd, m_blurLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(blurPC), &blurPC);
        vkCmdDispatch(cmd, groupCount(m_width), groupCount(m_height), 1);
        transitionAo(cmd, target.getImage(), VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    };
    blurAxis(m_blurHorizontalSet, m_aoTemp, glm::vec2(1.0f, 0.0f));
    blurAxis(m_blurVerticalSet, m_aoBlurred, glm::vec2(0.0f, 1.0f));

    if (timer) timer->writeTimestamp(cmd, "Gtao_End");
}

void GtaoPass::shutdown(VkDevice device)
{
    if (device == VK_NULL_HANDLE) return;

    for (VkPipeline* pipeline : { &m_gtaoPipeline, &m_blurPipeline }) {
        if (*pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, *pipeline, nullptr);
            *pipeline = VK_NULL_HANDLE;
        }
    }
    for (VkPipelineLayout* layout : { &m_gtaoLayout, &m_blurLayout }) {
        if (*layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, *layout, nullptr);
            *layout = VK_NULL_HANDLE;
        }
    }
    if (m_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, m_pool, nullptr);
        m_pool = VK_NULL_HANDLE;
    }
    if (m_setLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, m_setLayout, nullptr);
        m_setLayout = VK_NULL_HANDLE;
    }
    if (m_aoSampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, m_aoSampler, nullptr);
        m_aoSampler = VK_NULL_HANDLE;
    }
    destroyTargets();
}
