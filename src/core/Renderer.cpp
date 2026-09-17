#include "core/Renderer.h"
#include "core/FrustumCulling.h"
#include "core/VulkanUtil.h"
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
#include <utility>

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
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

static constexpr std::array<MsaaSampleCount, 4> kMsaaSampleCounts = {
    MsaaSampleCount::X1,
    MsaaSampleCount::X2,
    MsaaSampleCount::X4,
    MsaaSampleCount::X8,
};

static uint32_t sampleCountValue(MsaaSampleCount count)
{
    return static_cast<uint32_t>(count);
}

static size_t sampleCountIndex(MsaaSampleCount count)
{
    switch (count) {
        case MsaaSampleCount::X1: return 0;
        case MsaaSampleCount::X2: return 1;
        case MsaaSampleCount::X4: return 2;
        case MsaaSampleCount::X8: return 3;
    }
    return 0;
}

static VkSampleCountFlagBits toVkSampleCount(MsaaSampleCount count)
{
    switch (count) {
        case MsaaSampleCount::X1: return VK_SAMPLE_COUNT_1_BIT;
        case MsaaSampleCount::X2: return VK_SAMPLE_COUNT_2_BIT;
        case MsaaSampleCount::X4: return VK_SAMPLE_COUNT_4_BIT;
        case MsaaSampleCount::X8: return VK_SAMPLE_COUNT_8_BIT;
    }
    return VK_SAMPLE_COUNT_1_BIT;
}

static const char* antiAliasingModeName(AntiAliasingMode mode)
{
    switch (mode) {
        case AntiAliasingMode::None: return "None";
        case AntiAliasingMode::MSAA: return "MSAA";
    }
    return "Unknown";
}

static std::string supportedSampleCountsToString(const std::array<bool, 4>& supported)
{
    std::string result;
    for (size_t i = 0; i < supported.size(); ++i) {
        if (!supported[i]) {
            continue;
        }
        if (!result.empty()) {
            result += ", ";
        }
        result += std::to_string(sampleCountValue(kMsaaSampleCounts[i])) + "x";
    }
    return result.empty() ? "none" : result;
}

static MsaaSampleCount fallbackSupportedSampleCount(
    MsaaSampleCount requested,
    const std::array<bool, 4>& supported)
{
    size_t index = sampleCountIndex(requested);
    while (index > 0 && !supported[index]) {
        --index;
    }
    if (supported[index]) {
        return kMsaaSampleCounts[index];
    }
    return MsaaSampleCount::X1;
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

Renderer::Renderer(VulkanContext& ctx, Swapchain& swapchain)
    : m_ctx(ctx)
    , m_swapchain(swapchain)
    , m_commandBuffer(ctx)
    , m_frameSync(ctx)
    , m_pointLightBuffer(ctx)
    , m_clusterGridBuffer(ctx)
    , m_clusterLightIndexBuffer(ctx)
    , m_resources(ctx)
    , m_sceneDepthTarget(ctx)
    , m_msaaHdrTarget(ctx)
    , m_resolvedHdrTarget(ctx)
    , m_shadowPass(ctx)
    , m_clusteredLightCullingPass(ctx)
    , m_skyPass(ctx)
    , m_iblPass(ctx)
    , m_gtaoPass(ctx)
    , m_resolvedDepthTarget(ctx)
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
    m_antiAliasingStatus = resolveAntiAliasingSettings(m_antiAliasingSettings);
    logAntiAliasingStatus();

    m_frameSync.init(kFramesInFlight);
    m_commandBuffer.init(m_ctx.getGraphicsQueueFamily(), kFramesInFlight);
    m_tonemapPass.init(m_ctx.getDevice(), m_swapchain.getFormat(), SHADER_DIR);
    createResizeDependentResources();
    createPbrPipeline();
    m_clusteredLightCullingPass.init(m_sceneSetLayout, SHADER_DIR);
    m_skyPass.init(m_ctx, m_sceneSetLayout,
                   VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_D32_SFLOAT,
                   getActiveSceneSampleCount(),
                   m_antiAliasingStatus.sampleShadingEnabled,
                   m_antiAliasingStatus.minSampleShading,
                   SHADER_DIR);
    m_iblPass.init(SHADER_DIR, m_skyPass.panoramaView(), m_skyPass.panoramaSampler(),
                   m_skyPass.panoramaWidth());
    m_gtaoPass.init(SHADER_DIR);
    createFrameResources();
    createClusteredLightResources();
    m_shadowPass.init(m_pipelineLayout, kFramesInFlight, SHADER_DIR);
    createFrameSceneDescriptorPool();
    allocateFrameSceneDescriptorSets();
    createMaterialDescriptorPool();
    allocateMaterialDescriptorSets();
    refreshResizeDependentBindings();
    m_gpuTimer.init(24);

    // Populate static render stats (counts that don't change after load)
    refreshRenderStats();

    spdlog::info("Renderer initialized: {} meshes, {} materials, {} textures ({:.1f} MB GPU tex)",
                 m_renderStats.meshCount, m_renderStats.materialCount,
                 m_renderStats.textureCount,
                 m_renderStats.textureMemoryBytes / (1024.0f * 1024.0f));
}

void Renderer::uploadModelResources(const Model& model)
{
    spdlog::info("Uploading renderer model resources: {} meshes, {} materials, {} textures",
                 model.meshes.size(), model.materials.size(), model.textures.size());

    // Drain the GPU: no frame may reference resources/descriptors we are replacing.
    vkDeviceWaitIdle(m_ctx.getDevice());

    destroyRendererDescriptorPools();
    m_activeScene = {};
    m_resources.uploadModel(model);

    createFrameSceneDescriptorPool();
    allocateFrameSceneDescriptorSets();
    createMaterialDescriptorPool();
    allocateMaterialDescriptorSets();
    refreshResizeDependentBindings();

    // ── Update render stats ──────────────────────────────────────────────────────
    refreshRenderStats();

    spdlog::info("Renderer model resources uploaded: {} meshes, {} materials, {} textures ({:.1f} MB GPU tex)",
                 m_renderStats.meshCount, m_renderStats.materialCount,
                 m_renderStats.textureCount,
                 m_renderStats.textureMemoryBytes / (1024.0f * 1024.0f));
    spdlog::info("Model resources reloaded; waiting for caller to resubmit scene content");
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

    m_gtaoPass.shutdown(m_ctx.getDevice());
    m_iblPass.shutdown(m_ctx.getDevice());
    if (m_depthSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_ctx.getDevice(), m_depthSampler, nullptr);
        m_depthSampler = VK_NULL_HANDLE;
    }
    m_skyPass.shutdown(m_ctx.getDevice());
    m_clusteredLightCullingPass.shutdown(m_ctx.getDevice());
    m_shadowPass.shutdown(m_ctx.getDevice());

    destroyPipeline();

    m_activeScene = {};
    m_resources.clear();

    // Destroy GPU resources that hold VMA allocations
    destroyClusteredLightResources();
    destroyRendererDescriptorPools();
    m_frameResources.clear();

    m_commandBuffer.shutdown();
    m_frameSync.shutdown();
}

void Renderer::refreshRenderStats()
{
    m_renderStats.meshCount          = m_resources.meshCount();
    m_renderStats.materialCount      = m_resources.materialCount();
    m_renderStats.textureCount       = m_resources.textureCount();
    m_renderStats.aaMode             = m_antiAliasingStatus.mode;
    m_renderStats.activeSampleCount  = sampleCountValue(m_antiAliasingStatus.activeSampleCount);
    m_renderStats.sampleShadingEnabled = m_antiAliasingStatus.sampleShadingEnabled;
    m_renderStats.minSampleShading   = m_antiAliasingStatus.minSampleShading;
    m_renderStats.alphaToCoverageEnabled = m_antiAliasingStatus.alphaToCoverageEnabled;
    m_renderStats.estimatedMsaaAttachmentMemoryBytes = estimateMsaaAttachmentMemoryBytes();
    m_renderStats.textureMemoryBytes = m_resources.estimateTextureMemoryBytes();
}

AntiAliasingStatus Renderer::resolveAntiAliasingSettings(
    const AntiAliasingSettings& settings) const
{
    AntiAliasingStatus status{};
    status.mode = settings.mode;
    status.requestedSampleCount = settings.requestedSampleCount;
    status.supportedSampleCounts = m_ctx.getDeviceFeatures().supportedSceneSampleCounts;
    status.sampleRateShadingSupported = m_ctx.getDeviceFeatures().sampleRateShading;
    status.minSampleShading = std::clamp(settings.minSampleShading, 0.0f, 1.0f);

    if (settings.mode == AntiAliasingMode::None) {
        status.activeSampleCount = MsaaSampleCount::X1;
        status.fallbackSampleCount = MsaaSampleCount::X1;
        status.sampleShadingEnabled = false;
        status.alphaToCoverageEnabled = false;
        status.minSampleShading = 0.0f;
        return status;
    }

    status.activeSampleCount = fallbackSupportedSampleCount(
        settings.requestedSampleCount,
        status.supportedSampleCounts);
    status.fallbackSampleCount = status.activeSampleCount;

    const bool multisampled = status.activeSampleCount != MsaaSampleCount::X1;
    status.sampleShadingEnabled =
        multisampled &&
        settings.sampleShadingEnabled &&
        m_ctx.getDeviceFeatures().sampleRateShading;
    status.alphaToCoverageEnabled =
        multisampled &&
        settings.alphaToCoverageEnabled;
    if (!status.sampleShadingEnabled) {
        status.minSampleShading = 0.0f;
    }
    return status;
}

bool Renderer::isMsaaEnabled() const
{
    return m_antiAliasingStatus.activeSampleCount != MsaaSampleCount::X1;
}

bool Renderer::isAlphaToCoverageActive() const
{
    return isMsaaEnabled() && m_antiAliasingStatus.alphaToCoverageEnabled;
}

const char* Renderer::pbrFragmentShaderName() const
{
    // MoltenVK accepts sampleShadingEnable but still shades once per pixel, so the
    // sample-shading path uses a variant with sample-qualified inputs (ART-VPW-005).
    // Metal has no fractional mode: every sample is shaded regardless of minSampleShading.
    return m_antiAliasingStatus.sampleShadingEnabled ? "pbr_sample.frag.spv" : "pbr.frag.spv";
}

VkSampleCountFlagBits Renderer::getActiveSceneSampleCount() const
{
    return toVkSampleCount(m_antiAliasingStatus.activeSampleCount);
}

VkImageView Renderer::getActiveSceneColorAttachmentView() const
{
    return isMsaaEnabled()
        ? m_msaaHdrTarget.getImageView()
        : m_resolvedHdrTarget.getImageView();
}

VkImageView Renderer::getResolvedHdrView() const
{
    return m_resolvedHdrTarget.getImageView();
}

bool Renderer::isDrawAlphaMasked(const DrawCommand& draw) const
{
    const auto& materials = m_resources.materials();
    return draw.material.isValid() &&
           draw.material.index < materials.size() &&
           materials[draw.material.index].alphaMode == Material::AlphaMode::Mask;
}

VkPipeline Renderer::selectSolidPipelineForDraw(const DrawCommand& draw) const
{
    if (!isDrawAlphaMasked(draw)) {
        return m_pipeline;
    }
    if (isAlphaToCoverageActive() && m_maskedA2cPipeline != VK_NULL_HANDLE) {
        return m_maskedA2cPipeline;
    }
    return m_maskedPipeline != VK_NULL_HANDLE ? m_maskedPipeline : m_pipeline;
}

size_t Renderer::estimateMsaaAttachmentMemoryBytes() const
{
    if (!isMsaaEnabled()) {
        return 0;
    }

    const VkExtent2D ext = m_swapchain.getExtent();
    if (ext.width == 0 || ext.height == 0) {
        return 0;
    }

    constexpr size_t kHdrBytesPerPixel = 8;   // R16G16B16A16_SFLOAT
    constexpr size_t kDepthBytesPerPixel = 4; // D32_SFLOAT
    const size_t pixels = static_cast<size_t>(ext.width) * ext.height;
    const size_t samples = sampleCountValue(m_antiAliasingStatus.activeSampleCount);
    return pixels * samples * (kHdrBytesPerPixel + kDepthBytesPerPixel);
}

void Renderer::refreshTimingStats()
{
    m_renderStats.shadowGpuMs = m_gpuTimer.getElapsedMs("ShadowPass_Begin", "ShadowPass_End");
    m_renderStats.blurGpuMs = m_gpuTimer.getElapsedMs("BlurPass_Begin", "BlurPass_End");
    m_renderStats.clusterGpuMs = m_gpuTimer.getElapsedMs("ClusterCull_Begin", "ClusterCull_End");
    m_renderStats.sceneGpuMs = m_gpuTimer.getElapsedMs("ScenePass_Begin", "ScenePass_End");
    m_renderStats.tonemapGpuMs = m_gpuTimer.getElapsedMs("TonemapPass_Begin", "TonemapPass_End");
    const float iblBakeMs = m_gpuTimer.getElapsedMs("IblBake_Begin", "IblBake_End");
    if (iblBakeMs > 0.0f) {
        m_renderStats.iblBakeGpuMs = iblBakeMs;
    }
    m_renderStats.iblBakeCount = m_iblPass.bakeCount();
    m_renderStats.prepassGpuMs = m_gpuTimer.getElapsedMs("DepthPrepass_Begin", "DepthPrepass_End");
    m_renderStats.aoGpuMs = m_gpuTimer.getElapsedMs("Gtao_Begin", "Gtao_End");
    const float imguiMs = m_gpuTimer.getElapsedMs("TonemapPass_End", "ImGuiPass_End");
    m_renderStats.totalGpuFrameMs =
        m_renderStats.shadowGpuMs + m_renderStats.blurGpuMs + m_renderStats.clusterGpuMs +
        m_renderStats.sceneGpuMs + m_renderStats.tonemapGpuMs + imguiMs;
}

void Renderer::logAntiAliasingStatus() const
{
    spdlog::info("AA configuration applied:");
    spdlog::info("  Mode: {}", antiAliasingModeName(m_antiAliasingStatus.mode));
    spdlog::info("  Requested: {}x", sampleCountValue(m_antiAliasingStatus.requestedSampleCount));
    spdlog::info("  Supported: {}",
                 supportedSampleCountsToString(m_antiAliasingStatus.supportedSampleCounts));
    spdlog::info("  Active: {}x", sampleCountValue(m_antiAliasingStatus.activeSampleCount));
    spdlog::info("  Sample shading: {}",
                 m_antiAliasingStatus.sampleShadingEnabled ? "enabled" : "disabled");
    spdlog::info("  Minimum shading fraction: {:.2f}",
                 m_antiAliasingStatus.minSampleShading);
    spdlog::info("  Alpha-to-coverage: {}",
                 m_antiAliasingStatus.alphaToCoverageEnabled ? "enabled" : "disabled");
}

void Renderer::setAntiAliasingSettings(const AntiAliasingSettings& settings)
{
    const AntiAliasingStatus previousStatus = m_antiAliasingStatus;
    m_antiAliasingSettings = settings;
    m_antiAliasingStatus = resolveAntiAliasingSettings(m_antiAliasingSettings);
    refreshRenderStats();
    logAntiAliasingStatus();
    if (m_antiAliasingSettings.sampleShadingEnabled &&
        !m_antiAliasingStatus.sampleRateShadingSupported &&
        isMsaaEnabled() &&
        !m_loggedSampleShadingFallback) {
        spdlog::warn("Sample-rate shading requested but unsupported; MSAA remains enabled without sample shading");
        m_loggedSampleShadingFallback = true;
    }

    const bool rendererInitialized = m_pipelineLayout != VK_NULL_HANDLE;
    if (!rendererInitialized) {
        return;
    }

    const bool sampleCountChanged =
        previousStatus.activeSampleCount != m_antiAliasingStatus.activeSampleCount;
    const bool scenePipelineSampleStateChanged =
        sampleCountChanged ||
        previousStatus.sampleShadingEnabled != m_antiAliasingStatus.sampleShadingEnabled ||
        std::abs(previousStatus.minSampleShading - m_antiAliasingStatus.minSampleShading) > 0.0001f;
    const bool pbrPipelineStateChanged =
        scenePipelineSampleStateChanged ||
        previousStatus.alphaToCoverageEnabled != m_antiAliasingStatus.alphaToCoverageEnabled;

    if (!sampleCountChanged && !pbrPipelineStateChanged) {
        return;
    }

    vkDeviceWaitIdle(m_ctx.getDevice());

    if (pbrPipelineStateChanged) {
        destroyPbrGraphicsPipelines();
    }

    if (sampleCountChanged) {
        destroyResizeDependentResources();
        createResizeDependentResources();
        refreshResizeDependentBindings();
    }

    if (scenePipelineSampleStateChanged) {
        m_skyPass.recreatePipeline(m_ctx,
                                   getActiveSceneSampleCount(),
                                   m_antiAliasingStatus.sampleShadingEnabled,
                                   m_antiAliasingStatus.minSampleShading);
    }

    if (pbrPipelineStateChanged) {
        createPbrGraphicsPipelines();
    }
}

// ── Pipeline ──────────────────────────────────────────────────────────────────

void Renderer::createPbrPipeline()
{
    // Descriptor set layout: set 0 — camera(b0), light(b1), shadowUBO(b2), depth array(b3), moments(b4).
    const std::array<VkDescriptorSetLayoutBinding, 13> setBindings{{
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
        {   // bindings 9-11: IBL irradiance, prefiltered specular, BRDF LUT (IblPass)
            .binding         = shader_interface::scene_binding::kIblIrradiance,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding         = shader_interface::scene_binding::kIblPrefiltered,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding         = shader_interface::scene_binding::kIblBrdfLut,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {   // binding 12: screen-space ambient occlusion (GtaoPass)
            .binding         = shader_interface::scene_binding::kAmbientOcclusion,
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
    VK_CHECK(vkCreateDescriptorSetLayout(m_ctx.getDevice(), &setLayoutInfo, nullptr, &m_sceneSetLayout));

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
        m_sceneSetLayout,     // set 0
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

    // Graphics pipelines are (re)created separately so AA changes can rebuild them
    // without touching the descriptor/pipeline layouts above.
    createPbrGraphicsPipelines();
}

void Renderer::createPbrGraphicsPipelines()
{
    if (m_pipelineLayout == VK_NULL_HANDLE) {
        throw std::runtime_error("Cannot create PBR graphics pipelines before pipeline layout");
    }

    const std::string dir = SHADER_DIR;
    m_vertModule        = makeShaderModule(m_ctx.getDevice(), loadSpv(dir + "/pbr.vert.spv"));
    m_fragModule        = makeShaderModule(m_ctx.getDevice(), loadSpv(dir + "/" + pbrFragmentShaderName()));
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

    const auto bindingDesc = Vertex::getBindingDescription();
    const auto attribDescs = Vertex::getAttributeDescriptions();
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

    const VkPipelineViewportStateCreateInfo viewportState{
        .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount  = 1,
    };

    // glTF front faces are CCW. The camera projection flips Y, which keeps them CCW in
    // framebuffer space, so gl_FrontFacing is only correct with COUNTER_CLOCKWISE here
    // (pbr.frag negates N for back faces of double-sided materials).
    // Cull mode is dynamic: set per draw from Material::doubleSided (ART-VPW-008).
    VkPipelineRasterizationStateCreateInfo rasterization{
        .sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode    = VK_CULL_MODE_BACK_BIT,
        .frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .lineWidth   = 1.0f,
    };

    const VkPipelineMultisampleStateCreateInfo multisample{
        .sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = getActiveSceneSampleCount(),
        .sampleShadingEnable  = m_antiAliasingStatus.sampleShadingEnabled ? VK_TRUE : VK_FALSE,
        .minSampleShading     = m_antiAliasingStatus.minSampleShading,
        .alphaToCoverageEnable = VK_FALSE,
    };
    VkPipelineMultisampleStateCreateInfo maskedA2cMultisample = multisample;
    maskedA2cMultisample.alphaToCoverageEnable = isAlphaToCoverageActive() ? VK_TRUE : VK_FALSE;

    const VkPipelineDepthStencilStateCreateInfo depthStencil{
        .sType             = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable   = VK_TRUE,
        .depthWriteEnable  = VK_TRUE,
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

    const std::array<VkDynamicState, 5> dynamicStates = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_CULL_MODE,
        // Set per draw: EQUAL without depth writes for draws the prepass already
        // covered, GREATER with writes otherwise (ART-PRF-002).
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
        VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
    };
    const VkPipelineDynamicStateCreateInfo dynamicState{
        .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates    = dynamicStates.data(),
    };

    const VkFormat colorFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    const VkPipelineRenderingCreateInfo renderingInfo{
        .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount    = 1,
        .pColorAttachmentFormats = &colorFormat,
        .depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT,
        .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
    };

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

    VkGraphicsPipelineCreateInfo maskedPipelineInfo = pipelineInfo;

    VkGraphicsPipelineCreateInfo maskedA2cPipelineInfo = pipelineInfo;
    maskedA2cPipelineInfo.pMultisampleState = &maskedA2cMultisample;

    VkGraphicsPipelineCreateInfo wireframePipelineInfo = pipelineInfo;
    wireframePipelineInfo.pRasterizationState = &wireframeRasterization;

    VkGraphicsPipelineCreateInfo normalsPipelineInfo = pipelineInfo;
    normalsPipelineInfo.pStages    = normalsStages.data();
    normalsPipelineInfo.stageCount = static_cast<uint32_t>(normalsStages.size());

    std::vector<VkGraphicsPipelineCreateInfo> pipelineInfos;
    pipelineInfos.reserve(isAlphaToCoverageActive() ? 5 : 4);
    pipelineInfos.push_back(pipelineInfo);
    pipelineInfos.push_back(maskedPipelineInfo);
    if (isAlphaToCoverageActive()) {
        pipelineInfos.push_back(maskedA2cPipelineInfo);
    }
    pipelineInfos.push_back(wireframePipelineInfo);
    pipelineInfos.push_back(normalsPipelineInfo);

    std::vector<VkPipeline> pipelines(pipelineInfos.size(), VK_NULL_HANDLE);
    VK_CHECK(vkCreateGraphicsPipelines(
        m_ctx.getDevice(), VK_NULL_HANDLE,
        static_cast<uint32_t>(pipelineInfos.size()),
        pipelineInfos.data(), nullptr, pipelines.data()));

    size_t pipelineIndex = 0;
    m_pipeline          = pipelines[pipelineIndex++];
    m_maskedPipeline    = pipelines[pipelineIndex++];
    m_maskedA2cPipeline = isAlphaToCoverageActive()
        ? pipelines[pipelineIndex++]
        : VK_NULL_HANDLE;
    m_wireframePipeline = pipelines[pipelineIndex++];
    m_normalsPipeline   = pipelines[pipelineIndex++];

    // ── Depth prepass pipelines (depth only, no colour attachment) ───────────
    // Same vertex shader and vertex layout as the scene pass so the depth values
    // match exactly; the masked variant repeats pbr.frag's alpha cutout.
    {
        VkShaderModule prepassVert = makeShaderModule(m_ctx.getDevice(), loadSpv(dir + "/pbr.vert.spv"));
        VkShaderModule prepassAlphaFrag =
            makeShaderModule(m_ctx.getDevice(), loadSpv(dir + "/pbr_prepass_alpha.frag.spv"));

        const VkPipelineShaderStageCreateInfo vertStage{
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_VERTEX_BIT,
            .module = prepassVert,
            .pName  = "main",
        };
        const std::array<VkPipelineShaderStageCreateInfo, 2> maskedStages{{
            vertStage,
            {
                .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = prepassAlphaFrag,
                .pName  = "main",
            },
        }};

        const VkPipelineDepthStencilStateCreateInfo prepassDepth{
            .sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable  = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp   = VK_COMPARE_OP_GREATER,
        };
        const VkPipelineColorBlendStateCreateInfo prepassBlend{
            .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = 0,
        };
        const VkPipelineRenderingCreateInfo prepassRendering{
            .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .colorAttachmentCount    = 0,
            .depthAttachmentFormat   = VK_FORMAT_D32_SFLOAT,
            .stencilAttachmentFormat = VK_FORMAT_UNDEFINED,
        };
        const std::array<VkDynamicState, 3> prepassDynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_CULL_MODE,
        };
        const VkPipelineDynamicStateCreateInfo prepassDynamicState{
            .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = static_cast<uint32_t>(prepassDynamicStates.size()),
            .pDynamicStates    = prepassDynamicStates.data(),
        };

        VkGraphicsPipelineCreateInfo opaqueInfo = makeGraphicsPipelineCreateInfo(
            prepassRendering, &vertStage, 1, vertexInput, inputAssembly, viewportState,
            rasterization, multisample, prepassDepth, prepassBlend, prepassDynamicState,
            m_pipelineLayout);
        VkGraphicsPipelineCreateInfo maskedInfo = opaqueInfo;
        maskedInfo.pStages = maskedStages.data();
        maskedInfo.stageCount = static_cast<uint32_t>(maskedStages.size());

        const std::array<VkGraphicsPipelineCreateInfo, 2> prepassInfos{ opaqueInfo, maskedInfo };
        std::array<VkPipeline, 2> prepassPipelines{};
        VK_CHECK(vkCreateGraphicsPipelines(m_ctx.getDevice(), VK_NULL_HANDLE,
                                           static_cast<uint32_t>(prepassInfos.size()),
                                           prepassInfos.data(), nullptr, prepassPipelines.data()));
        m_depthPrepassPipeline = prepassPipelines[0];
        m_depthPrepassMaskedPipeline = prepassPipelines[1];

        vkDestroyShaderModule(m_ctx.getDevice(), prepassVert, nullptr);
        vkDestroyShaderModule(m_ctx.getDevice(), prepassAlphaFrag, nullptr);
    }

    vkDestroyShaderModule(m_ctx.getDevice(), m_vertModule, nullptr);         m_vertModule        = VK_NULL_HANDLE;
    vkDestroyShaderModule(m_ctx.getDevice(), m_fragModule, nullptr);         m_fragModule        = VK_NULL_HANDLE;
    vkDestroyShaderModule(m_ctx.getDevice(), m_normalsFragModule, nullptr);  m_normalsFragModule = VK_NULL_HANDLE;

    spdlog::info("PBR graphics pipelines created for {}x scene samples{}",
                 sampleCountValue(m_antiAliasingStatus.activeSampleCount),
                 isAlphaToCoverageActive() ? " with masked A2C" : "");
}

void Renderer::destroyPbrGraphicsPipelines()
{
    const VkDevice dev = m_ctx.getDevice();
    if (m_vertModule         != VK_NULL_HANDLE) { vkDestroyShaderModule(dev, m_vertModule, nullptr);               m_vertModule        = VK_NULL_HANDLE; }
    if (m_fragModule         != VK_NULL_HANDLE) { vkDestroyShaderModule(dev, m_fragModule, nullptr);               m_fragModule        = VK_NULL_HANDLE; }
    if (m_normalsFragModule  != VK_NULL_HANDLE) { vkDestroyShaderModule(dev, m_normalsFragModule, nullptr);        m_normalsFragModule = VK_NULL_HANDLE; }
    if (m_pipeline           != VK_NULL_HANDLE) { vkDestroyPipeline(dev, m_pipeline, nullptr);                     m_pipeline          = VK_NULL_HANDLE; }
    if (m_maskedPipeline     != VK_NULL_HANDLE) { vkDestroyPipeline(dev, m_maskedPipeline, nullptr);               m_maskedPipeline    = VK_NULL_HANDLE; }
    if (m_maskedA2cPipeline  != VK_NULL_HANDLE) { vkDestroyPipeline(dev, m_maskedA2cPipeline, nullptr);            m_maskedA2cPipeline = VK_NULL_HANDLE; }
    if (m_wireframePipeline  != VK_NULL_HANDLE) { vkDestroyPipeline(dev, m_wireframePipeline, nullptr);            m_wireframePipeline = VK_NULL_HANDLE; }
    if (m_normalsPipeline    != VK_NULL_HANDLE) { vkDestroyPipeline(dev, m_normalsPipeline, nullptr);              m_normalsPipeline   = VK_NULL_HANDLE; }
    if (m_depthPrepassPipeline != VK_NULL_HANDLE) { vkDestroyPipeline(dev, m_depthPrepassPipeline, nullptr);        m_depthPrepassPipeline = VK_NULL_HANDLE; }
    if (m_depthPrepassMaskedPipeline != VK_NULL_HANDLE) { vkDestroyPipeline(dev, m_depthPrepassMaskedPipeline, nullptr); m_depthPrepassMaskedPipeline = VK_NULL_HANDLE; }
}

void Renderer::destroyPipeline()
{
    const VkDevice dev = m_ctx.getDevice();
    destroyPbrGraphicsPipelines();
    if (m_pipelineLayout     != VK_NULL_HANDLE) { vkDestroyPipelineLayout(dev, m_pipelineLayout, nullptr);         m_pipelineLayout    = VK_NULL_HANDLE; }
    if (m_materialSetLayout  != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(dev, m_materialSetLayout, nullptr); m_materialSetLayout = VK_NULL_HANDLE; }
    if (m_sceneSetLayout     != VK_NULL_HANDLE) { vkDestroyDescriptorSetLayout(dev, m_sceneSetLayout, nullptr);    m_sceneSetLayout    = VK_NULL_HANDLE; }
}

void Renderer::destroyRendererDescriptorPools()
{
    const VkDevice dev = m_ctx.getDevice();
    if (dev == VK_NULL_HANDLE) return;

    if (m_frameSceneDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(dev, m_frameSceneDescriptorPool, nullptr);
        m_frameSceneDescriptorPool = VK_NULL_HANDLE;
    }
    for (auto& frame : m_frameResources) {
        frame.sceneDescriptorSet = VK_NULL_HANDLE;
    }

    if (m_materialDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(dev, m_materialDescriptorPool, nullptr);
        m_materialDescriptorPool = VK_NULL_HANDLE;
    }
    m_materialDescriptorSets.clear();
}

Renderer::FrameResources& Renderer::currentFrameResources()
{
    return m_frameResources[m_frameSync.getCurrentFrame()];
}

const Renderer::FrameResources& Renderer::currentFrameResources() const
{
    return m_frameResources[m_frameSync.getCurrentFrame()];
}

Renderer::CameraUBO Renderer::buildCurrentCameraUBO() const
{
    return CameraUBO{
        m_viewMatrix,
        m_projMatrix,
        glm::inverse(m_projMatrix * m_viewMatrix),
        m_cameraPos,
        0.0f,
    };
}

Renderer::LightUBO Renderer::buildCurrentLightUBO() const
{
    return LightUBO{
        m_lightDirection,
        m_lightIntensity,
        m_lightColor,
        m_ambientIntensity,
        m_shadowSettings.debugCascades ? 1u : 0u,
        static_cast<uint32_t>(m_shadowSettings.filterMode),
        m_shadowSettings.pcfSpreadRadius,
        m_shadowSettings.vsmBleedReduction,
        m_iblSettings.enabled ? 1u : 0u,
        std::max(m_iblSettings.intensity, 0.0f),
        m_iblPass.prefilteredMaxLod(),
        (m_aoSettings.enabled && isDepthPrepassActive()) ? 1u : 0u,
        std::clamp(m_aoSettings.intensity, 0.0f, 1.0f),
        m_aoSettings.debugView ? 1u : 0u,
        0.0f,
        0.0f,
    };
}

Renderer::ClusterMetadataUBO Renderer::buildCurrentClusterMetadata() const
{
    const VkExtent2D ext = m_swapchain.getExtent();
    return ClusterMetadataUBO{
        glm::uvec4(m_clusterCountX, m_clusterCountY, m_clusterCountZ,
                   shader_interface::kMaxLightsPerCluster),
        glm::uvec4(static_cast<uint32_t>(std::min(m_activeScene.submission.pointLights.size(),
                                                  static_cast<size_t>(m_maxPointLights))),
                   0u, 0u, 0u),
        glm::vec4(static_cast<float>(ext.width),
                  static_cast<float>(ext.height),
                  m_cameraNearZ,
                  m_cameraFarZ),
    };
}

void Renderer::createFrameResources()
{
    m_frameResources.clear();
    m_frameResources.reserve(kFramesInFlight);

    const CameraUBO initialCamera = buildCurrentCameraUBO();
    const LightUBO initialLight = buildCurrentLightUBO();
    const ClusterMetadataUBO initialClusterMetadata = buildCurrentClusterMetadata();

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        m_frameResources.emplace_back(m_ctx);
        auto& frame = m_frameResources.back();
        frame.cameraUBO.createHostVisible(sizeof(CameraUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        frame.lightUBO.createHostVisible(sizeof(LightUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        frame.clusterMetadataUBO.createHostVisible(sizeof(ClusterMetadataUBO),
                                                   VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
        frame.cameraUBO.upload(&initialCamera, sizeof(initialCamera));
        frame.lightUBO.upload(&initialLight, sizeof(initialLight));
        frame.clusterMetadataUBO.upload(&initialClusterMetadata, sizeof(initialClusterMetadata));
    }

    spdlog::info("Frame resources created for {} frames in flight", kFramesInFlight);
}

void Renderer::buildVisibleDrawOrder()
{
    const auto& draws = m_activeScene.submission.draws;
    const auto& bounds = m_activeScene.drawBounds;
    m_visibleDrawOrder.clear();
    m_visibleDrawOrder.reserve(draws.size());

    // Camera planes from an equivalent [-1, 1]-depth projection: the reverse-Z camera
    // matrix does not fit the Gribb–Hartmann near/far rows (same approach as ShadowPass).
    const float projY = std::abs(m_projMatrix[1][1]);
    const float projX = std::abs(m_projMatrix[0][0]);
    const bool cull = m_cullingSettings.enableFrustumCulling &&
                      projX > 1e-6f && projY > 1e-6f &&
                      bounds.size() == draws.size();
    std::array<culling::Plane, 6> planes{};
    if (cull) {
        const float nearZ = std::max(m_cameraNearZ, 1e-4f);
        const float farZ = std::max(m_cameraFarZ, nearZ + 1e-3f);
        const glm::mat4 proj = glm::perspectiveRH_NO(2.0f * std::atan(1.0f / projY),
                                                     projY / projX, nearZ, farZ);
        planes = culling::extractFrustumPlanesNO(proj * m_viewMatrix);
    }

    for (uint32_t i = 0; i < draws.size(); ++i) {
        if (cull && culling::aabbOutsideAny(bounds[i].worldMin, bounds[i].worldMax,
                                            planes.data(), planes.size())) {
            ++m_renderStats.culledDrawCalls;
            continue;
        }
        m_visibleDrawOrder.push_back(i);
    }

    if (!m_cullingSettings.sortDraws || bounds.size() != draws.size()) {
        return;
    }

    // Opaque first, then alpha-tested, then alpha-to-coverage draws; front to back
    // within each group so the depth test rejects hidden fragments before shading.
    m_drawSortScratch.clear();
    m_drawSortScratch.reserve(m_visibleDrawOrder.size());
    for (uint32_t i : m_visibleDrawOrder) {
        const DrawCommand& draw = draws[i];
        uint64_t group = 0;
        if (isDrawAlphaMasked(draw)) {
            group = isAlphaToCoverageActive() ? 2 : 1;
        }
        const glm::vec3 closest = glm::clamp(m_cameraPos, bounds[i].worldMin, bounds[i].worldMax);
        const float distance = glm::length(closest - m_cameraPos);
        const uint64_t distanceKey = static_cast<uint64_t>(
            std::min(distance * 1024.0f, static_cast<float>(0xFFFFFFFFu)));
        m_drawSortScratch.emplace_back((group << 32) | distanceKey, i);
    }
    std::stable_sort(m_drawSortScratch.begin(), m_drawSortScratch.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });
    for (size_t k = 0; k < m_drawSortScratch.size(); ++k) {
        m_visibleDrawOrder[k] = m_drawSortScratch[k].second;
    }
}

void Renderer::rebuildActiveSceneDrawBounds()
{
    m_activeScene.drawBounds.clear();
    m_activeScene.drawBounds.reserve(m_activeScene.submission.draws.size());
    const auto& meshes = m_resources.meshes();

    for (const DrawCommand& draw : m_activeScene.submission.draws) {
        if (!draw.mesh.isValid() || draw.mesh.index >= meshes.size()) {
            m_activeScene.drawBounds.push_back({});
            continue;
        }

        const MeshRenderData& mesh = meshes[draw.mesh.index];
        const glm::vec3& lo = mesh.boundsMin;
        const glm::vec3& hi = mesh.boundsMax;
        glm::vec3 worldMin(std::numeric_limits<float>::max());
        glm::vec3 worldMax(-std::numeric_limits<float>::max());

        for (int cornerIndex = 0; cornerIndex < 8; ++cornerIndex) {
            const glm::vec3 corner((cornerIndex & 1) ? hi.x : lo.x,
                                   (cornerIndex & 2) ? hi.y : lo.y,
                                   (cornerIndex & 4) ? hi.z : lo.z);
            const glm::vec3 world = glm::vec3(draw.transform * glm::vec4(corner, 1.0f));
            worldMin = glm::min(worldMin, world);
            worldMax = glm::max(worldMax, world);
        }

        m_activeScene.drawBounds.push_back({
            .worldMin = worldMin,
            .worldMax = worldMax,
        });
    }
}

void Renderer::applyFrameSubmission(const RenderFramePacket& packet)
{
    m_iblSettings = packet.ibl;
    uploadCurrentFrameCameraState(packet.camera);
    m_lightDirection   = safeNormalizeDirection(packet.light.direction, glm::vec3(0.577f));
    m_lightColor       = packet.light.color;
    m_lightIntensity   = packet.light.intensity;
    m_ambientIntensity = packet.light.ambient;
    m_shadowSettings   = packet.shadow;

    m_shadowPass.updateCamera(m_viewMatrix, m_projMatrix, m_cameraNearZ, m_cameraFarZ, m_cameraPos);
    m_shadowPass.updateLightDirection(m_lightDirection);
    const bool shadowResourcesChanged = m_shadowPass.updateSettings(m_shadowSettings);
    // Adopt the pass's sanitized settings: the light UBO (filter mode, PCF spread,
    // VSM bleed reduction) must describe the shadow pass that actually runs, not the
    // raw caller values. An unsanitized vsmBleedReduction of 1.0 would divide by zero.
    m_shadowSettings = m_shadowPass.getSettings();
    const bool shadowDescriptorsDirty = m_shadowPass.consumeDescriptorDirty();
    if (shadowResourcesChanged || shadowDescriptorsDirty) {
        rewriteFrameSceneShadowBindings();
    }
    uploadCurrentFrameLightState();
    uploadCurrentClusterMetadata();

    m_tonemapMode     = packet.tonemap.mode;
    m_exposure        = packet.tonemap.exposure;
    m_splitScreenMode = packet.tonemap.splitScreen ? 1 : 0;
    m_splitRightMode  = packet.tonemap.splitRightMode;

    m_skyEnabled = packet.sky.enabled;
    m_skyMode    = packet.sky.mode;
    // The IBL environment follows the sky mode even when the sky itself is hidden.
    m_iblPass.updateSource(m_skyMode, m_lightDirection, m_lightIntensity,
                           m_skyPass.panoramaGeneration());

    m_wireframe   = packet.debug.wireframe;
    m_showNormals = packet.debug.showNormals;

    m_cullingSettings = packet.culling;
    m_aoSettings = packet.ao;
}

void Renderer::uploadCurrentFrameCameraState(const CameraData& camera)
{
    m_cameraPos  = camera.position;
    m_viewMatrix = camera.view;
    m_projMatrix = camera.projection;
    m_cameraNearZ = camera.nearZ;
    m_cameraFarZ  = camera.farZ;
    const CameraUBO cameraUBO = buildCurrentCameraUBO();
    currentFrameResources().cameraUBO.upload(&cameraUBO, sizeof(cameraUBO));
}

void Renderer::uploadCurrentFrameLightState()
{
    const LightUBO lightUBO = buildCurrentLightUBO();
    currentFrameResources().lightUBO.upload(&lightUBO, sizeof(lightUBO));
}

void Renderer::createClusteredLightResources()
{
    m_pointLightBuffer.createHostVisible(
        static_cast<VkDeviceSize>(m_maxPointLights) * sizeof(shader_interface::GpuPointLight),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    resizeClusteredLightResources();
    uploadActiveScenePointLights();

    spdlog::info("Clustered Forward+ light resources created: max point lights={}, max lights/cluster={}",
                 m_maxPointLights, shader_interface::kMaxLightsPerCluster);
}

void Renderer::destroyClusteredLightResources()
{
    m_clusterLightIndexBuffer.destroy();
    m_clusterGridBuffer.destroy();
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

    spdlog::info("Cluster grid resized: {}x{}x{} ({} clusters)",
                 m_clusterCountX, m_clusterCountY, m_clusterCountZ, m_clusterCount);
}

void Renderer::uploadActiveScenePointLights()
{
    if (m_pointLightBuffer.getBuffer() == VK_NULL_HANDLE) return;

    std::vector<shader_interface::GpuPointLight> uploadData(m_maxPointLights);
    const size_t copyCount = std::min(m_activeScene.submission.pointLights.size(), uploadData.size());
    if (copyCount > 0) {
        std::copy_n(m_activeScene.submission.pointLights.data(), copyCount, uploadData.data());
    }
    m_pointLightBuffer.upload(uploadData.data(),
                              static_cast<VkDeviceSize>(uploadData.size() * sizeof(uploadData[0])));
}

void Renderer::uploadCurrentClusterMetadata()
{
    if (m_frameResources.empty()) return;
    const ClusterMetadataUBO metadata = buildCurrentClusterMetadata();
    currentFrameResources().clusterMetadataUBO.upload(&metadata, sizeof(metadata));
}

void Renderer::uploadClusterMetadataForAllFrames()
{
    if (m_frameResources.empty()) return;

    const ClusterMetadataUBO metadata = buildCurrentClusterMetadata();
    for (auto& frame : m_frameResources) {
        frame.clusterMetadataUBO.upload(&metadata, sizeof(metadata));
    }
}

bool Renderer::isDepthPrepassActive() const
{
    // AO needs prepass depth, so enabling AO implies the prepass.
    return m_cullingSettings.enableDepthPrepass || m_aoSettings.enabled;
}

void Renderer::rewriteFrameSceneAoBinding()
{
    if (m_frameResources.empty() || !m_gtaoPass.isReady()) {
        return;
    }
    const VkDescriptorImageInfo aoInfo{
        .sampler     = m_gtaoPass.aoSampler(),
        .imageView   = m_gtaoPass.aoView(),
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    for (auto& frame : m_frameResources) {
        if (frame.sceneDescriptorSet == VK_NULL_HANDLE) continue;
        const VkWriteDescriptorSet write{
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet          = frame.sceneDescriptorSet,
            .dstBinding      = shader_interface::scene_binding::kAmbientOcclusion,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo      = &aoInfo,
        };
        vkUpdateDescriptorSets(m_ctx.getDevice(), 1, &write, 0, nullptr);
    }
}

void Renderer::rewriteFrameSceneClusterBindings()
{
    if (m_frameResources.empty() ||
        m_pointLightBuffer.getBuffer() == VK_NULL_HANDLE ||
        m_clusterGridBuffer.getBuffer() == VK_NULL_HANDLE ||
        m_clusterLightIndexBuffer.getBuffer() == VK_NULL_HANDLE) {
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

    for (auto& frame : m_frameResources) {
        if (frame.sceneDescriptorSet == VK_NULL_HANDLE) {
            continue;
        }

        const VkDescriptorBufferInfo clusterMetadataInfo{
            .buffer = frame.clusterMetadataUBO.getBuffer(),
            .offset = 0,
            .range = sizeof(ClusterMetadataUBO),
        };

        const std::array<VkWriteDescriptorSet, 4> writes{{
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              .dstSet = frame.sceneDescriptorSet,
              .dstBinding = shader_interface::scene_binding::kPointLights,
              .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .pBufferInfo = &pointLightInfo },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              .dstSet = frame.sceneDescriptorSet,
              .dstBinding = shader_interface::scene_binding::kClusterGrid,
              .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .pBufferInfo = &clusterGridInfo },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              .dstSet = frame.sceneDescriptorSet,
              .dstBinding = shader_interface::scene_binding::kClusterLightIndices,
              .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              .pBufferInfo = &clusterIndicesInfo },
            { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
              .dstSet = frame.sceneDescriptorSet,
              .dstBinding = shader_interface::scene_binding::kClusterMetadata,
              .descriptorCount = 1,
              .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              .pBufferInfo = &clusterMetadataInfo },
        }};
        vkUpdateDescriptorSets(m_ctx.getDevice(), static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }
}

void Renderer::createSceneDepthTarget()
{
    const VkExtent2D ext = m_swapchain.getExtent();
    // D32_SFLOAT: 32-bit float depth, no stencil, optimal for reverse-Z precision
    m_sceneDepthTarget.create(ext.width, ext.height, 1,
                              VK_FORMAT_D32_SFLOAT,
                              VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                  VK_IMAGE_USAGE_SAMPLED_BIT,
                              VK_IMAGE_ASPECT_DEPTH_BIT,
                              1,
                              getActiveSceneSampleCount());

    // Single-sample copy of the scene depth for AO. With MSAA it is produced by the
    // prepass depth resolve; without MSAA the scene depth is sampled directly.
    if (isMsaaEnabled()) {
        m_resolvedDepthTarget.create(ext.width, ext.height, 1,
                                     VK_FORMAT_D32_SFLOAT,
                                     VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                         VK_IMAGE_USAGE_SAMPLED_BIT,
                                     VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    if (m_depthSampler == VK_NULL_HANDLE) {
        const VkSamplerCreateInfo samplerCI{
            .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter    = VK_FILTER_NEAREST,
            .minFilter    = VK_FILTER_NEAREST,
            .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .maxLod       = 0.0f,
        };
        VK_CHECK(vkCreateSampler(m_ctx.getDevice(), &samplerCI, nullptr, &m_depthSampler));
    }

    spdlog::info("Scene depth target created: {}x{} D32_SFLOAT (reverse-Z, {}x)",
                 ext.width, ext.height,
                 sampleCountValue(m_antiAliasingStatus.activeSampleCount));
}

void Renderer::createResizeDependentResources()
{
    const VkExtent2D ext = m_swapchain.getExtent();
    if (ext.width == 0 || ext.height == 0) {
        return;
    }

    createSceneDepthTarget();
    createSceneHdrTargets();
    refreshRenderStats();
    if (m_pointLightBuffer.getBuffer() != VK_NULL_HANDLE) {
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
    m_msaaHdrTarget.destroy();
    m_resolvedHdrTarget.destroy();
    m_sceneDepthTarget.destroy();
    m_resolvedDepthTarget.destroy();
}

void Renderer::recreateResizeDependentResources()
{
    destroyResizeDependentResources();
    createResizeDependentResources();
}

void Renderer::refreshResizeDependentBindings()
{
    // Frame scene descriptor sets own the clustered-light buffer bindings that
    // point at resize-dependent cluster resources.
    rewriteFrameSceneClusterBindings();
    uploadClusterMetadataForAllFrames();

    // AO reads the single-sample depth image; its own targets follow the swapchain size.
    const VkExtent2D aoExtent = m_swapchain.getExtent();
    const VkImageView aoDepthView = isMsaaEnabled()
        ? m_resolvedDepthTarget.getImageView()
        : m_sceneDepthTarget.getImageView();
    if (aoDepthView != VK_NULL_HANDLE && m_depthSampler != VK_NULL_HANDLE) {
        m_gtaoPass.resize(aoExtent.width, aoExtent.height, aoDepthView, m_depthSampler);
        rewriteFrameSceneAoBinding();
    }

    // TonemapPass owns its descriptor set, but Renderer owns the resolved HDR target it samples.
    if (getResolvedHdrView() != VK_NULL_HANDLE) {
        m_tonemapPass.resize(m_ctx.getDevice(), m_hdrSampler, getResolvedHdrView());
    }
}

void Renderer::createSceneHdrTargets()
{
    const VkExtent2D ext = m_swapchain.getExtent();

    m_resolvedHdrTarget.create(ext.width, ext.height, 1,
                               VK_FORMAT_R16G16B16A16_SFLOAT,
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                               VK_IMAGE_ASPECT_COLOR_BIT);

    if (isMsaaEnabled()) {
        m_msaaHdrTarget.create(ext.width, ext.height, 1,
                               VK_FORMAT_R16G16B16A16_SFLOAT,
                               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                               VK_IMAGE_ASPECT_COLOR_BIT,
                               1,
                               getActiveSceneSampleCount());
    }

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

    spdlog::info("Scene HDR targets created: {}x{} R16G16B16A16_SFLOAT (scene {}x, resolved 1x)",
                 ext.width, ext.height,
                 sampleCountValue(m_antiAliasingStatus.activeSampleCount));
}

void Renderer::loadHdrPanorama(const std::string& path)
{
    m_skyPass.loadHdrPanorama(m_ctx, path);
}

void Renderer::createFrameSceneDescriptorPool()
{
    // Renderer owns one pool for frame scene descriptor sets because those sets
    // are rewritten per frame slot and reference frame-local UBOs.
    const uint32_t sceneSetCount = kFramesInFlight;

    const std::array<VkDescriptorPoolSize, 3> poolSizes{{
        {
            .type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 4 * sceneSetCount,
        },
        {
            .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 6 * sceneSetCount,
        },
        {
            .type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 3 * sceneSetCount,
        },
    }};

    const VkDescriptorPoolCreateInfo poolCI{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = sceneSetCount,
        .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
        .pPoolSizes    = poolSizes.data(),
    };
    VK_CHECK(vkCreateDescriptorPool(m_ctx.getDevice(), &poolCI, nullptr,
                                    &m_frameSceneDescriptorPool));

    spdlog::info("Frame scene descriptor pool created for {} frame slots", sceneSetCount);
}

void Renderer::allocateFrameSceneDescriptorSets()
{
    std::vector<VkDescriptorSetLayout> layouts(kFramesInFlight, m_sceneSetLayout);
    std::vector<VkDescriptorSet> descriptorSets(kFramesInFlight, VK_NULL_HANDLE);

    const VkDescriptorSetAllocateInfo allocInfo{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = m_frameSceneDescriptorPool,
        .descriptorSetCount = kFramesInFlight,
        .pSetLayouts        = layouts.data(),
    };
    VK_CHECK(vkAllocateDescriptorSets(m_ctx.getDevice(), &allocInfo, descriptorSets.data()));

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
    const VkDescriptorImageInfo iblIrradianceInfo{
        .sampler     = m_iblPass.environmentSampler(),
        .imageView   = m_iblPass.irradianceView(),
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkDescriptorImageInfo iblPrefilteredInfo{
        .sampler     = m_iblPass.environmentSampler(),
        .imageView   = m_iblPass.prefilteredView(),
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    const VkDescriptorImageInfo iblBrdfLutInfo{
        .sampler     = m_iblPass.lutSampler(),
        .imageView   = m_iblPass.brdfLutView(),
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
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

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        auto& frame = m_frameResources[i];
        frame.sceneDescriptorSet = descriptorSets[i];

        const VkDescriptorBufferInfo cameraInfo{
            .buffer = frame.cameraUBO.getBuffer(),
            .offset = 0,
            .range  = sizeof(CameraUBO),
        };
        const VkDescriptorBufferInfo lightInfo{
            .buffer = frame.lightUBO.getBuffer(),
            .offset = 0,
            .range  = sizeof(LightUBO),
        };
        const VkDescriptorBufferInfo shadowUBOInfo{
            .buffer = m_shadowPass.getShadowUBOBuffer(i),
            .offset = 0,
            .range  = sizeof(ShadowCascadeUBO),
        };
        const VkDescriptorBufferInfo clusterMetadataInfo{
            .buffer = frame.clusterMetadataUBO.getBuffer(),
            .offset = 0,
            .range = sizeof(ClusterMetadataUBO),
        };

        const std::array<VkWriteDescriptorSet, 12> writes{{
            {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = frame.sceneDescriptorSet,
                .dstBinding      = shader_interface::scene_binding::kCamera,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo     = &cameraInfo,
            },
            {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = frame.sceneDescriptorSet,
                .dstBinding      = shader_interface::scene_binding::kLight,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo     = &lightInfo,
            },
            {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = frame.sceneDescriptorSet,
                .dstBinding      = shader_interface::scene_binding::kShadow,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo     = &shadowUBOInfo,
            },
            {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = frame.sceneDescriptorSet,
                .dstBinding      = shader_interface::scene_binding::kShadowMap,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo      = &shadowMapInfo,
            },
            {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = frame.sceneDescriptorSet,
                .dstBinding      = shader_interface::scene_binding::kShadowMoments,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo      = &momentsInfo,
            },
            {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = frame.sceneDescriptorSet,
                .dstBinding      = shader_interface::scene_binding::kPointLights,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo     = &pointLightInfo,
            },
            {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = frame.sceneDescriptorSet,
                .dstBinding      = shader_interface::scene_binding::kClusterGrid,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo     = &clusterGridInfo,
            },
            {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = frame.sceneDescriptorSet,
                .dstBinding      = shader_interface::scene_binding::kClusterLightIndices,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo     = &clusterIndicesInfo,
            },
            {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = frame.sceneDescriptorSet,
                .dstBinding      = shader_interface::scene_binding::kClusterMetadata,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo     = &clusterMetadataInfo,
            },
            {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = frame.sceneDescriptorSet,
                .dstBinding      = shader_interface::scene_binding::kIblIrradiance,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo      = &iblIrradianceInfo,
            },
            {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = frame.sceneDescriptorSet,
                .dstBinding      = shader_interface::scene_binding::kIblPrefiltered,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo      = &iblPrefilteredInfo,
            },
            {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = frame.sceneDescriptorSet,
                .dstBinding      = shader_interface::scene_binding::kIblBrdfLut,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo      = &iblBrdfLutInfo,
            },
        }};
        vkUpdateDescriptorSets(m_ctx.getDevice(),
                               static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    spdlog::info("Frame scene descriptor sets allocated for {} frames in flight", kFramesInFlight);
}

void Renderer::rewriteFrameSceneShadowBindings()
{
    if (m_frameResources.empty()) return;

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

    for (uint32_t i = 0; i < m_frameResources.size(); ++i) {
        auto& frame = m_frameResources[i];
        if (frame.sceneDescriptorSet == VK_NULL_HANDLE) {
            continue;
        }

        const VkDescriptorBufferInfo shadowUBOInfo{
            .buffer = m_shadowPass.getShadowUBOBuffer(i),
            .offset = 0,
            .range = sizeof(ShadowCascadeUBO),
        };

        const std::array<VkWriteDescriptorSet, 3> writes{{
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = frame.sceneDescriptorSet,
                .dstBinding = shader_interface::scene_binding::kShadow,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                .pBufferInfo = &shadowUBOInfo,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = frame.sceneDescriptorSet,
                .dstBinding = shader_interface::scene_binding::kShadowMap,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &shadowMapInfo,
            },
            {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = frame.sceneDescriptorSet,
                .dstBinding = shader_interface::scene_binding::kShadowMoments,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo = &momentsInfo,
            },
        }};
        vkUpdateDescriptorSets(m_ctx.getDevice(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void Renderer::createMaterialDescriptorPool()
{
    // Renderer owns material texture descriptors alongside persistent uploaded materials.
    const uint32_t materialCount = m_resources.materialCount();
    if (materialCount == 0) {
        return;
    }

    const VkDescriptorPoolSize poolSize{
        .type            = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .descriptorCount = 3 * materialCount,
    };

    const VkDescriptorPoolCreateInfo poolCI{
        .sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets       = materialCount,
        .poolSizeCount = 1,
        .pPoolSizes    = &poolSize,
    };
    VK_CHECK(vkCreateDescriptorPool(m_ctx.getDevice(), &poolCI, nullptr,
                                    &m_materialDescriptorPool));

    spdlog::info("Material descriptor pool created for {} material sets", materialCount);
}

void Renderer::allocateMaterialDescriptorSets()
{
    const auto& materials = m_resources.materials();
    const auto& textures = m_resources.textures();
    const uint32_t materialCount = static_cast<uint32_t>(materials.size());
    if (materialCount == 0) return;

    // Allocate all material sets in one batch — more efficient than N individual calls.
    std::vector<VkDescriptorSetLayout> layouts(materialCount, m_materialSetLayout);
    m_materialDescriptorSets.resize(materialCount);

    const VkDescriptorSetAllocateInfo allocInfo{
        .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool     = m_materialDescriptorPool,
        .descriptorSetCount = materialCount,
        .pSetLayouts        = layouts.data(),
    };
    VK_CHECK(vkAllocateDescriptorSets(m_ctx.getDevice(), &allocInfo, m_materialDescriptorSets.data()));

    // Helpers: resolve a texture index to its view/sampler, falling back to white.
    auto getView = [&](int32_t idx) -> VkImageView {
        if (idx >= 0 && idx < static_cast<int32_t>(textures.size())
            && textures[static_cast<size_t>(idx)].isValid())
            return textures[static_cast<size_t>(idx)].getImageView();
        return m_resources.fallbackWhite().getImageView();
    };
    auto getSampler = [&](int32_t idx) -> VkSampler {
        if (idx >= 0 && idx < static_cast<int32_t>(textures.size())
            && textures[static_cast<size_t>(idx)].isValid())
            return textures[static_cast<size_t>(idx)].getSampler();
        return m_resources.fallbackWhite().getSampler();
    };

    for (uint32_t i = 0; i < materialCount; ++i) {
        const Material& mat = materials[i];

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
                .dstSet          = m_materialDescriptorSets[i],
                .dstBinding      = shader_interface::material_binding::kAlbedo,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo      = &albedoInfo,
            },
            {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = m_materialDescriptorSets[i],
                .dstBinding      = shader_interface::material_binding::kNormal,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo      = &normalInfo,
            },
            {
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = m_materialDescriptorSets[i],
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
    handleResize(oldExt.width, oldExt.height);
}

void Renderer::resize(uint32_t width, uint32_t height)
{
    if (width == 0 || height == 0) {
        return;
    }
    handleResize(width, height);
}

void Renderer::handleResize(uint32_t width, uint32_t height)
{
    const VkExtent2D oldExt = m_swapchain.getExtent();

    // Resize contract:
    // wait idle -> recreate swapchain -> recreate renderer resize resources
    // -> refresh bindings that point at recreated resources -> keep scene/frame state coherent.
    vkDeviceWaitIdle(m_ctx.getDevice());

    // Swapchain::createSwapchain() reads surface capabilities and uses
    // cap.currentExtent when available, falling back to clamped requested
    // dimensions otherwise.
    m_swapchain.recreate(width, height);

    const VkExtent2D newExt = m_swapchain.getExtent();
    if (newExt.width > 0 && newExt.height > 0) {
        recreateResizeDependentResources();
        refreshResizeDependentBindings();
    } else {
        destroyResizeDependentResources();
    }

    spdlog::info("Renderer resized: {}x{} -> {}x{}", oldExt.width, oldExt.height,
                 newExt.width, newExt.height);
}

bool Renderer::beginFrame()
{
    m_frameSync.waitForFrame();
    // Fence has signaled — previous frame's GPU work is complete; safe to read timestamps.
    m_gpuTimer.collectResults();
    refreshTimingStats();

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

void Renderer::recordDepthPrepass(VkCommandBuffer cmd, const FrameResources& frame)
{
    const VkExtent2D ext = m_swapchain.getExtent();
    const auto& meshes = m_resources.meshes();
    const auto& materials = m_resources.materials();
    const bool msaa = isMsaaEnabled();

    // Reverse-Z: clear to 0 (far). The resolve produces the single-sample depth AO reads;
    // MAX keeps the closest sample, falling back to SAMPLE_ZERO when unsupported.
    const VkResolveModeFlags supportedResolves =
        m_ctx.getDeviceFeatures().supportedDepthResolveModes;
    const VkResolveModeFlagBits resolveMode =
        (supportedResolves & VK_RESOLVE_MODE_MAX_BIT) ? VK_RESOLVE_MODE_MAX_BIT
                                                      : VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
    const bool resolveDepth = msaa && m_resolvedDepthTarget.getImageView() != VK_NULL_HANDLE;
    if (resolveDepth) {
        vkutil::transitionImage(cmd, m_resolvedDepthTarget.getImage(),
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    const VkRenderingAttachmentInfo depthAttachment{
        .sType              = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView          = m_sceneDepthTarget.getImageView(),
        .imageLayout        = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        .resolveMode        = resolveDepth ? resolveMode : VK_RESOLVE_MODE_NONE,
        .resolveImageView   = resolveDepth ? m_resolvedDepthTarget.getImageView() : VK_NULL_HANDLE,
        .resolveImageLayout = resolveDepth ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
                                           : VK_IMAGE_LAYOUT_UNDEFINED,
        .loadOp             = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp            = VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue         = { .depthStencil = { 0.0f, 0 } },
    };
    const VkRenderingInfo renderingInfo{
        .sType                = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea           = { .offset = { 0, 0 }, .extent = ext },
        .layerCount           = 1,
        .colorAttachmentCount = 0,
        .pDepthAttachment     = &depthAttachment,
    };

    m_gpuTimer.writeTimestamp(cmd, "DepthPrepass_Begin");
    vkCmdBeginRendering(cmd, &renderingInfo);

    const VkViewport viewport{
        .x = 0.0f, .y = 0.0f,
        .width = static_cast<float>(ext.width), .height = static_cast<float>(ext.height),
        .minDepth = 0.0f, .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    const VkRect2D scissor{ .offset = { 0, 0 }, .extent = ext };
    vkCmdSetScissor(cmd, 0, 1, &scissor);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                            shader_interface::kSceneSet, 1, &frame.sceneDescriptorSet, 0, nullptr);

    const VkDeviceSize vertexOffset = 0;
    const VkBuffer vertexBuf = m_resources.vertexBuffer().getBuffer();
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuf, &vertexOffset);
    vkCmdBindIndexBuffer(cmd, m_resources.indexBuffer().getBuffer(), 0, VK_INDEX_TYPE_UINT32);

    VkPipeline boundPipeline = VK_NULL_HANDLE;
    VkCullModeFlags boundCullMode = VK_CULL_MODE_FRONT_AND_BACK;
    const auto& draws = m_activeScene.submission.draws;
    for (const uint32_t drawIndex : m_visibleDrawOrder) {
        const DrawCommand& draw = draws[drawIndex];
        if (!draw.mesh.isValid() || draw.mesh.index >= meshes.size()) continue;
        const bool alphaMasked = isDrawAlphaMasked(draw);
        // Alpha-to-coverage draws resolve their own coverage in the scene pass, so they
        // are left out of the prepass and keep writing depth there.
        if (alphaMasked && isAlphaToCoverageActive()) continue;

        const VkPipeline pipeline = alphaMasked ? m_depthPrepassMaskedPipeline
                                                : m_depthPrepassPipeline;
        if (pipeline != boundPipeline) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            boundPipeline = pipeline;
        }
        const bool doubleSided =
            !draw.material.isValid() || draw.material.index >= materials.size() ||
            materials[draw.material.index].doubleSided;
        const VkCullModeFlags cullMode = doubleSided ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT;
        if (cullMode != boundCullMode) {
            vkCmdSetCullMode(cmd, cullMode);
            boundCullMode = cullMode;
        }

        vkCmdPushConstants(cmd, m_pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           shader_interface::kModelMatrixPcOffset, sizeof(glm::mat4), &draw.transform);
        if (alphaMasked && draw.material.index < m_materialDescriptorSets.size()) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                                    shader_interface::kMaterialSet, 1,
                                    &m_materialDescriptorSets[draw.material.index], 0, nullptr);
            const Material& mat = materials[draw.material.index];
            const MaterialPushConstants matPC{
                .baseColorFactor  = mat.baseColorFactor,
                .metallicFactor   = mat.metallicFactor,
                .roughnessFactor  = mat.roughnessFactor,
                .alphaCutoff      = mat.alphaCutoff,
                .alphaCoverageMode = 0.0f,
            };
            vkCmdPushConstants(cmd, m_pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               shader_interface::kMaterialPcOffset, sizeof(matPC), &matPC);
        }
        vkCmdDrawIndexed(cmd, meshes[draw.mesh.index].indexCount, 1,
                         meshes[draw.mesh.index].firstIndex, 0, 0);
    }

    vkCmdEndRendering(cmd);
    m_gpuTimer.writeTimestamp(cmd, "DepthPrepass_End");
}

void Renderer::render(RendererOverlay* overlay)
{
    VkCommandBuffer cmd = m_commandBuffer.getFrameCommandBuffer();
    FrameResources& frame = currentFrameResources();

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
    m_renderStats.opaqueDrawCalls = 0;
    m_renderStats.maskedDrawCalls = 0;
    m_renderStats.culledDrawCalls = 0;
    m_renderStats.aaMode = m_antiAliasingStatus.mode;
    m_renderStats.activeSampleCount = sampleCountValue(m_antiAliasingStatus.activeSampleCount);
    m_renderStats.sampleShadingEnabled = m_antiAliasingStatus.sampleShadingEnabled;
    m_renderStats.minSampleShading = m_antiAliasingStatus.minSampleShading;
    m_renderStats.alphaToCoverageEnabled = m_antiAliasingStatus.alphaToCoverageEnabled;
    m_renderStats.estimatedMsaaAttachmentMemoryBytes = estimateMsaaAttachmentMemoryBytes();

    const auto& meshes = m_resources.meshes();
    const auto& materials = m_resources.materials();
    const bool hasSceneGeometry =
        !meshes.empty() &&
        m_resources.vertexBuffer().getBuffer() != VK_NULL_HANDLE &&
        m_resources.indexBuffer().getBuffer() != VK_NULL_HANDLE;

    // Update shadow matrices each frame so the shadow follows the camera.
    if (hasSceneGeometry) {
        m_shadowPass.record(cmd,
                            frame.sceneDescriptorSet,
                            m_frameSync.getCurrentFrame(),
                            m_resources.vertexBuffer(),
                            m_resources.indexBuffer(),
                            m_activeScene.submission.draws,
                            m_activeScene.drawBounds,
                            meshes,
                            materials,
                            m_materialDescriptorSets,
                            m_gpuTimer);
        if (m_shadowPass.consumeDescriptorDirty()) {
            rewriteFrameSceneShadowBindings();
        }
    }

    m_gpuTimer.writeTimestamp(cmd, "ClusterCull_Begin");
    m_clusteredLightCullingPass.record(cmd, frame.sceneDescriptorSet,
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

    // ── IBL environment re-bake (compute, only when the sky source changed) ──
    if (m_iblPass.needsBake()) {
        m_iblPass.bake(cmd, m_skyPass.panoramaView(), m_skyPass.panoramaSampler(),
                       m_skyPass.panoramaWidth(), &m_gpuTimer);
    }

    const bool sceneMsaaEnabled = isMsaaEnabled();
    Image& sceneColorTarget = sceneMsaaEnabled ? m_msaaHdrTarget : m_resolvedHdrTarget;

    vkutil::transitionImage(cmd, sceneColorTarget.getImage(),
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,              0,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

    if (sceneMsaaEnabled) {
        vkutil::transitionImage(cmd, m_resolvedHdrTarget.getImage(),
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,              0,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED,                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }

    // Scene depth transitions from UNDEFINED each frame — the first pass to touch it
    // clears it, so UNDEFINED→DEPTH_ATTACHMENT_OPTIMAL is valid and avoids layout tracking.
    vkutil::transitionImage(cmd, m_sceneDepthTarget.getImage(),
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,              0,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,                         VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    // Visibility is shared by the prepass and the scene pass.
    if (hasSceneGeometry) {
        buildVisibleDrawOrder();
    }

    // ── Depth prepass + screen-space AO ──────────────────────────────────────
    const bool prepassActive = hasSceneGeometry && isDepthPrepassActive();
    if (prepassActive) {
        recordDepthPrepass(cmd, frame);

        if (m_aoSettings.enabled && m_gtaoPass.isReady()) {
            const bool msaaDepth = isMsaaEnabled() &&
                                   m_resolvedDepthTarget.getImageView() != VK_NULL_HANDLE;
            Image& aoDepthImage = msaaDepth ? m_resolvedDepthTarget : m_sceneDepthTarget;
            vkutil::transitionImage(cmd, aoDepthImage.getImage(),
                VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_ASPECT_DEPTH_BIT);

            m_gtaoPass.record(cmd, glm::inverse(m_projMatrix),
                              std::max(m_aoSettings.radius, 0.01f), 1.0f, &m_gpuTimer);

            if (!msaaDepth) {
                // The scene pass keeps rendering into this image, so hand it back.
                vkutil::transitionImage(cmd, aoDepthImage.getImage(),
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                    VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,    VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                    VK_IMAGE_ASPECT_DEPTH_BIT);
            }
        }
    }

    // ── Scene pass → HDR offscreen target ────────────────────────────────────
    // Clear to black in linear space (sRGB 0.01 ≈ linear ~0.001; using 0 is visually identical).
    const VkRenderingAttachmentInfo colorAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = getActiveSceneColorAttachmentView(),
        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        .resolveMode = sceneMsaaEnabled ? VK_RESOLVE_MODE_AVERAGE_BIT : VK_RESOLVE_MODE_NONE,
        .resolveImageView = sceneMsaaEnabled ? getResolvedHdrView() : VK_NULL_HANDLE,
        .resolveImageLayout = sceneMsaaEnabled
            ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
            : VK_IMAGE_LAYOUT_UNDEFINED,
        .loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp     = sceneMsaaEnabled ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE,
        .clearValue  = { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } } },
    };

    // Reverse-Z: clear depth to 0.0 (far plane); closer fragments have larger depth values
    const VkRenderingAttachmentInfo depthAttachment{
        .sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView   = m_sceneDepthTarget.getImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        // With a prepass the depth is already complete: load it and test EQUAL.
        .loadOp      = prepassActive ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR,
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


    // ── PBR geometry pass ─────────────────────────────────────────────────────
    VkPipeline boundScenePipeline = VK_NULL_HANDLE;
    auto bindScenePipeline = [&](VkPipeline pipeline) {
        if (pipeline != VK_NULL_HANDLE && boundScenePipeline != pipeline) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            boundScenePipeline = pipeline;
        }
    };
    // Dynamic cull mode must be set before the first PBR draw; the sentinel forces it.
    VkCullModeFlags boundCullMode = VK_CULL_MODE_FRONT_AND_BACK;
    VkCompareOp boundDepthCompare = VK_COMPARE_OP_MAX_ENUM;

    // Bind camera UBO descriptor set (set=0, binding=0 — view/projection matrices)
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                            shader_interface::kSceneSet, 1, &frame.sceneDescriptorSet, 0, nullptr);

    if (hasSceneGeometry) {
        // Bind vertex and index buffers
        const VkDeviceSize vertexOffset = 0;
        const VkBuffer vertexBuf = m_resources.vertexBuffer().getBuffer();
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuf, &vertexOffset);
        vkCmdBindIndexBuffer(cmd, m_resources.indexBuffer().getBuffer(), 0, VK_INDEX_TYPE_UINT32);

        // Push normalization model matrix (bytes 0–63): same transform as shadow pass.
        const auto& draws = m_activeScene.submission.draws;
        for (const uint32_t drawIndex : m_visibleDrawOrder) {
            const DrawCommand& draw = draws[drawIndex];
            if (!draw.mesh.isValid() || draw.mesh.index >= meshes.size())
                continue;

            const MeshRenderData& mesh = meshes[draw.mesh.index];
            const bool alphaMasked = isDrawAlphaMasked(draw);

            VkPipeline drawPipeline = selectSolidPipelineForDraw(draw);
            if (m_showNormals) {
                drawPipeline = m_normalsPipeline;
            } else if (m_wireframe) {
                drawPipeline = m_wireframePipeline;
            }
            const bool drawUsesAlphaCoverage =
                alphaMasked && drawPipeline == m_maskedA2cPipeline;
            bindScenePipeline(drawPipeline);

            // Single-sided materials cull back faces; debug views show every face.
            const bool doubleSided =
                !draw.material.isValid() ||
                draw.material.index >= materials.size() ||
                materials[draw.material.index].doubleSided;
            const VkCullModeFlags cullMode = (doubleSided || m_showNormals || m_wireframe)
                ? VK_CULL_MODE_NONE
                : VK_CULL_MODE_BACK_BIT;
            if (cullMode != boundCullMode) {
                vkCmdSetCullMode(cmd, cullMode);
                boundCullMode = cullMode;
            }

            // Draws covered by the prepass test EQUAL and write no depth; A2C draws were
            // skipped there, so they keep the regular GREATER test with depth writes.
            const bool coveredByPrepass = prepassActive && !drawUsesAlphaCoverage;
            const VkCompareOp depthCompare = coveredByPrepass ? VK_COMPARE_OP_EQUAL
                                                              : VK_COMPARE_OP_GREATER;
            if (depthCompare != boundDepthCompare) {
                vkCmdSetDepthCompareOp(cmd, depthCompare);
                vkCmdSetDepthWriteEnable(cmd, coveredByPrepass ? VK_FALSE : VK_TRUE);
                boundDepthCompare = depthCompare;
            }

            vkCmdPushConstants(cmd, m_pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               shader_interface::kModelMatrixPcOffset, sizeof(glm::mat4), &draw.transform);

            if (draw.material.isValid() &&
                draw.material.index < m_materialDescriptorSets.size() &&
                draw.material.index < materials.size()) {

                // Bind material texture set (set=1) — set=0 remains bound and unaffected.
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        m_pipelineLayout,
                                        shader_interface::kMaterialSet, 1, &m_materialDescriptorSets[draw.material.index],
                                        0, nullptr);

                // Push material factors to fragment stage (offset 64, after model matrix).
                const Material& mat = materials[draw.material.index];
                const MaterialPushConstants matPC{
                    .baseColorFactor  = mat.baseColorFactor,
                    .metallicFactor   = mat.metallicFactor,
                    .roughnessFactor  = mat.roughnessFactor,
                    .alphaCutoff      = alphaMasked ? mat.alphaCutoff : 0.0f,
                    .alphaCoverageMode = drawUsesAlphaCoverage ? 1.0f : 0.0f,
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
            if (alphaMasked) {
                m_renderStats.maskedDrawCalls++;
            } else {
                m_renderStats.opaqueDrawCalls++;
            }
            m_renderStats.triangles += mesh.indexCount / 3;
        }
    }

    // ── Sky pass: fullscreen triangle at reverse-Z far plane (z = 0) ─────────
    // GREATER_OR_EQUAL depth only passes where depth is still the 0.0 clear value, so
    // drawing the sky after the geometry lets early-Z skip the atmosphere shader on
    // every covered pixel (the result is identical to drawing it first).
    m_skyPass.record(cmd, frame.sceneDescriptorSet, ext, m_skyEnabled, m_skyMode);

    vkCmdEndRendering(cmd);

    m_gpuTimer.writeTimestamp(cmd, "ScenePass_End");

    // ── Transition resolved HDR target: attachment/resolve write → fragment shader read ─
    vkutil::transitionImage(cmd, m_resolvedHdrTarget.getImage(),
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

    // Optional viewer/debug overlay pass (LOAD_OP_LOAD preserves the tone-mapped image).
    if (overlay) {
        overlay->record(RendererOverlayRenderInfo{
            .commandBuffer = cmd,
            .swapchainView = m_swapchain.getCurrentImageView(),
            .extent = m_swapchain.getExtent(),
        });
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

void Renderer::submitScene(RenderScenePacket scene)
{
    // Scene submission is sticky until the caller resubmits a new packet.
    // Renderer takes ownership of the moved packet and rebuilds scene-derived caches here.
    m_activeScene.submission = std::move(scene);
    rebuildActiveSceneDrawBounds();
    uploadActiveScenePointLights();
    uploadClusterMetadataForAllFrames();
}

void Renderer::submitFrame(const RenderFramePacket& packet)
{
    // Normal rendering uses a single per-frame packet handoff after beginFrame().
    applyFrameSubmission(packet);
}

void Renderer::endFrame(RendererOverlay* overlay)
{
    render(overlay);

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
