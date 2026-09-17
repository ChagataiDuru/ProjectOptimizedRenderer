#include "renderpasses/IblPass.h"

#include "core/VulkanUtil.h"
#include "debug/GPUTimer.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace {

constexpr VkFormat kEnvironmentFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat kLutFormat = VK_FORMAT_R16G16_SFLOAT;
constexpr uint32_t kPrefilterSampleCount = 64;

// Re-bake when the procedural sun moves by more than ~0.5 degrees or its
// intensity changes by more than 1%.
constexpr float kSunDirectionCosThreshold = 0.99996f;
constexpr float kSunIntensityRelThreshold = 0.01f;

struct CapturePC {
    glm::vec4 sunDirIntensity;
    uint32_t mode;
    float panoramaLod;
};

struct IrradiancePC {
    int32_t sourceLod;
};

struct PrefilterPC {
    float roughness;
    uint32_t sampleCount;
    float environmentWidth;
    float environmentHeight;
};

static_assert(sizeof(CapturePC) == 24);
static_assert(sizeof(IrradiancePC) == 4);
static_assert(sizeof(PrefilterPC) == 16);

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

void transitionForCompute(VkCommandBuffer cmd, VkImage image,
                          VkImageLayout oldLayout, VkImageLayout newLayout)
{
    auto stageAccess = [](VkImageLayout layout) -> std::pair<VkPipelineStageFlags2, VkAccessFlags2> {
        switch (layout) {
            case VK_IMAGE_LAYOUT_GENERAL:
                return { VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT };
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                return { VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT };
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

float panoramaLodFor(uint32_t panoramaWidth)
{
    if (panoramaWidth <= IblPass::kEnvironmentWidth) {
        return 0.0f;
    }
    return std::log2(static_cast<float>(panoramaWidth) /
                     static_cast<float>(IblPass::kEnvironmentWidth));
}

} // namespace

IblPass::IblPass(VulkanContext& ctx)
    : m_ctx(ctx)
    , m_environment(ctx)
    , m_irradiance(ctx)
    , m_prefiltered(ctx)
    , m_brdfLut(ctx)
{
}

IblPass::~IblPass()
{
    shutdown(m_ctx.getDevice());
}

std::vector<uint32_t> IblPass::loadSpv(const std::string& path)
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

VkPipeline IblPass::createComputePipeline(const std::string& path, uint32_t pushConstantSize,
                                          VkPipelineLayout& outLayout)
{
    const VkDevice dev = m_ctx.getDevice();
    const VkPushConstantRange range{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = pushConstantSize,
    };
    const VkPipelineLayoutCreateInfo layoutCI{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &m_setLayout,
        .pushConstantRangeCount = pushConstantSize > 0 ? 1u : 0u,
        .pPushConstantRanges = pushConstantSize > 0 ? &range : nullptr,
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
}

void IblPass::writeSet(VkDescriptorSet set, VkImageView storageView, VkImageView sampledView,
                       VkSampler sampler)
{
    const VkDescriptorImageInfo storageInfo{
        .imageView = storageView,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo sampledInfo{
        .sampler = sampler,
        .imageView = sampledView,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const std::array<VkWriteDescriptorSet, 2> writes{{
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &storageInfo,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &sampledInfo,
        },
    }};
    vkUpdateDescriptorSets(m_ctx.getDevice(), static_cast<uint32_t>(writes.size()),
                           writes.data(), 0, nullptr);
}

void IblPass::init(const std::string& shaderDir,
                   VkImageView panoramaView, VkSampler panoramaSampler,
                   uint32_t panoramaWidth)
{
    const VkDevice dev = m_ctx.getDevice();
    m_shaderDir = shaderDir;

    const uint32_t envMips = Image::fullMipChainLevels(kEnvironmentWidth, kEnvironmentHeight);
    if (!m_environment.supportsLinearBlit(kEnvironmentFormat)) {
        throw std::runtime_error("IBL environment format cannot be linearly blitted");
    }
    m_environment.create(kEnvironmentWidth, kEnvironmentHeight, 1, kEnvironmentFormat,
                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                             VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_SAMPLE_COUNT_1_BIT, envMips);
    m_irradiance.create(kIrradianceWidth, kIrradianceHeight, 1, kEnvironmentFormat,
                        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    m_prefiltered.create(kEnvironmentWidth, kEnvironmentHeight, 1, kEnvironmentFormat,
                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                         VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_SAMPLE_COUNT_1_BIT, kPrefilteredMips);
    m_brdfLut.create(kBrdfLutSize, kBrdfLutSize, 1, kLutFormat,
                     VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    m_environmentLevel0View = m_environment.createMipView(0);
    for (uint32_t level = 0; level < kPrefilteredMips; ++level) {
        m_prefilteredLevelViews[level] = m_prefiltered.createMipView(level);
    }

    // Equirect maps wrap horizontally and clamp at the poles.
    const VkSamplerCreateInfo envSamplerCI{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .maxAnisotropy = 1.0f,
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
    };
    VK_CHECK(vkCreateSampler(dev, &envSamplerCI, nullptr, &m_environmentSampler));
    VkSamplerCreateInfo lutSamplerCI = envSamplerCI;
    lutSamplerCI.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    lutSamplerCI.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    VK_CHECK(vkCreateSampler(dev, &lutSamplerCI, nullptr, &m_lutSampler));

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

    const uint32_t setCount = 3 + kPrefilteredMips;
    const std::array<VkDescriptorPoolSize, 2> poolSizes{{
        { .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = setCount },
        { .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = setCount },
    }};
    const VkDescriptorPoolCreateInfo poolCI{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = setCount,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes = poolSizes.data(),
    };
    VK_CHECK(vkCreateDescriptorPool(dev, &poolCI, nullptr, &m_pool));

    std::vector<VkDescriptorSetLayout> layouts(setCount, m_setLayout);
    std::vector<VkDescriptorSet> sets(setCount, VK_NULL_HANDLE);
    const VkDescriptorSetAllocateInfo allocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = m_pool,
        .descriptorSetCount = setCount,
        .pSetLayouts = layouts.data(),
    };
    VK_CHECK(vkAllocateDescriptorSets(dev, &allocInfo, sets.data()));
    m_captureSet = sets[0];
    m_irradianceSet = sets[1];
    m_lutSet = sets[2];
    for (uint32_t level = 0; level < kPrefilteredMips; ++level) {
        m_prefilterSets[level] = sets[3 + level];
    }

    const VkImageView envView = m_environment.getImageView();
    writeSet(m_captureSet, m_environmentLevel0View, panoramaView, panoramaSampler);
    writeSet(m_irradianceSet, m_irradiance.getImageView(), envView, m_environmentSampler);
    writeSet(m_lutSet, m_brdfLut.getImageView(), envView, m_environmentSampler);
    for (uint32_t level = 0; level < kPrefilteredMips; ++level) {
        writeSet(m_prefilterSets[level], m_prefilteredLevelViews[level], envView, m_environmentSampler);
    }

    m_capturePipeline = createComputePipeline(shaderDir + "/ibl_env_capture.comp.spv",
                                              sizeof(CapturePC), m_captureLayout);
    m_irradiancePipeline = createComputePipeline(shaderDir + "/ibl_irradiance.comp.spv",
                                                 sizeof(IrradiancePC), m_irradianceLayout);
    m_prefilterPipeline = createComputePipeline(shaderDir + "/ibl_prefilter.comp.spv",
                                                sizeof(PrefilterPC), m_prefilterLayout);
    m_lutPipeline = createComputePipeline(shaderDir + "/ibl_brdf_lut.comp.spv", 0, m_lutLayout);

    // Bake the LUT and a default procedural environment so every sampled image is
    // valid (SHADER_READ_ONLY) before the first frame binds it.
    updateSource(0, glm::normalize(glm::vec3(0.577f)), 3.5f, 0);
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = vkutil::beginSingleUseCommands(m_ctx, pool);
    bakeBrdfLut(cmd);
    bake(cmd, panoramaView, panoramaSampler, panoramaWidth, nullptr);
    vkutil::endSingleUseCommands(m_ctx, pool, cmd);

    spdlog::info("IBL pass created: environment {}x{}, irradiance {}x{}, prefiltered {} mips, BRDF LUT {}",
                 kEnvironmentWidth, kEnvironmentHeight, kIrradianceWidth, kIrradianceHeight,
                 kPrefilteredMips, kBrdfLutSize);
}

void IblPass::bakeBrdfLut(VkCommandBuffer cmd)
{
    transitionForCompute(cmd, m_brdfLut.getImage(),
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_lutPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_lutLayout,
                            0, 1, &m_lutSet, 0, nullptr);
    vkCmdDispatch(cmd, groupCount(kBrdfLutSize), groupCount(kBrdfLutSize), 1);
    transitionForCompute(cmd, m_brdfLut.getImage(),
                         VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void IblPass::updateSource(int32_t skyMode, const glm::vec3& sunDirection, float sunIntensity,
                           uint64_t panoramaGeneration)
{
    m_requested.skyMode = skyMode;
    m_requested.sunDirection = sunDirection;
    m_requested.sunIntensity = sunIntensity;
    m_requested.panoramaGeneration = panoramaGeneration;

    if (m_requested.skyMode != m_baked.skyMode) {
        m_dirty = true;
    } else if (skyMode == 1) {
        m_dirty = m_dirty || panoramaGeneration != m_baked.panoramaGeneration;
    } else {
        const float cosDelta = glm::dot(glm::normalize(sunDirection),
                                        glm::normalize(m_baked.sunDirection + glm::vec3(1e-9f)));
        const float intensityDelta = std::abs(sunIntensity - m_baked.sunIntensity) /
                                     std::max(std::abs(m_baked.sunIntensity), 1e-3f);
        m_dirty = m_dirty || cosDelta < kSunDirectionCosThreshold ||
                  intensityDelta > kSunIntensityRelThreshold;
    }
}

void IblPass::bake(VkCommandBuffer cmd, VkImageView panoramaView, VkSampler panoramaSampler,
                   uint32_t panoramaWidth, GPUTimer* timer)
{
    if (timer) timer->writeTimestamp(cmd, "IblBake_Begin");

    // The panorama descriptor is refreshed every bake: a newly loaded panorama has a new view.
    writeSet(m_captureSet, m_environmentLevel0View, panoramaView, panoramaSampler);

    // 1. Capture the sky into environment mip 0.
    transitionForCompute(cmd, m_environment.getImage(),
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    const CapturePC capturePC{
        .sunDirIntensity = glm::vec4(m_requested.sunDirection, m_requested.sunIntensity),
        .mode = m_requested.skyMode == 1 ? 1u : 0u,
        .panoramaLod = panoramaLodFor(panoramaWidth),
    };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_capturePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_captureLayout,
                            0, 1, &m_captureSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_captureLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(capturePC), &capturePC);
    vkCmdDispatch(cmd, groupCount(kEnvironmentWidth), groupCount(kEnvironmentHeight), 1);

    // 2. Mip chain (GENERAL -> TRANSFER_DST keeps level 0; ends SHADER_READ_ONLY).
    transitionForCompute(cmd, m_environment.getImage(),
                         VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    m_environment.generateMipmaps(cmd);

    // 3. Diffuse irradiance from the 32x16 environment mip.
    transitionForCompute(cmd, m_irradiance.getImage(),
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    const IrradiancePC irradiancePC{
        .sourceLod = static_cast<int32_t>(std::log2(static_cast<float>(kEnvironmentWidth) /
                                                    static_cast<float>(kIrradianceWidth))),
    };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_irradiancePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_irradianceLayout,
                            0, 1, &m_irradianceSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_irradianceLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(irradiancePC), &irradiancePC);
    vkCmdDispatch(cmd, groupCount(kIrradianceWidth), groupCount(kIrradianceHeight), 1);
    transitionForCompute(cmd, m_irradiance.getImage(),
                         VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // 4. Specular prefilter, one dispatch per roughness mip.
    transitionForCompute(cmd, m_prefiltered.getImage(),
                         VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_prefilterPipeline);
    for (uint32_t level = 0; level < kPrefilteredMips; ++level) {
        const uint32_t width = std::max(kEnvironmentWidth >> level, 1u);
        const uint32_t height = std::max(kEnvironmentHeight >> level, 1u);
        const PrefilterPC prefilterPC{
            .roughness = static_cast<float>(level) / static_cast<float>(kPrefilteredMips - 1),
            .sampleCount = kPrefilterSampleCount,
            .environmentWidth = static_cast<float>(kEnvironmentWidth),
            .environmentHeight = static_cast<float>(kEnvironmentHeight),
        };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_prefilterLayout,
                                0, 1, &m_prefilterSets[level], 0, nullptr);
        vkCmdPushConstants(cmd, m_prefilterLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(prefilterPC), &prefilterPC);
        vkCmdDispatch(cmd, groupCount(width), groupCount(height), 1);
    }
    transitionForCompute(cmd, m_prefiltered.getImage(),
                         VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    if (timer) timer->writeTimestamp(cmd, "IblBake_End");

    m_baked = m_requested;
    m_dirty = false;
    ++m_bakeCount;
    spdlog::debug("IBL environment baked (mode {}, bake #{})", m_baked.skyMode, m_bakeCount);
}

void IblPass::shutdown(VkDevice device)
{
    if (device == VK_NULL_HANDLE) return;

    for (VkPipeline* pipeline : { &m_capturePipeline, &m_irradiancePipeline,
                                  &m_prefilterPipeline, &m_lutPipeline }) {
        if (*pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, *pipeline, nullptr);
            *pipeline = VK_NULL_HANDLE;
        }
    }
    for (VkPipelineLayout* layout : { &m_captureLayout, &m_irradianceLayout,
                                      &m_prefilterLayout, &m_lutLayout }) {
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
    for (VkSampler* sampler : { &m_environmentSampler, &m_lutSampler }) {
        if (*sampler != VK_NULL_HANDLE) {
            vkDestroySampler(device, *sampler, nullptr);
            *sampler = VK_NULL_HANDLE;
        }
    }
    if (m_environmentLevel0View != VK_NULL_HANDLE) {
        vkDestroyImageView(device, m_environmentLevel0View, nullptr);
        m_environmentLevel0View = VK_NULL_HANDLE;
    }
    for (VkImageView& view : m_prefilteredLevelViews) {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, view, nullptr);
            view = VK_NULL_HANDLE;
        }
    }
    m_environment.destroy();
    m_irradiance.destroy();
    m_prefiltered.destroy();
    m_brdfLut.destroy();
}
