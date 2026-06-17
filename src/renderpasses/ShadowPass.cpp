#include "renderpasses/ShadowPass.h"

#include "core/VulkanUtil.h"
#include "resource/Vertex.h"

#include <spdlog/spdlog.h>
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {

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

uint32_t resolutionForPreset(int32_t preset)
{
    switch (preset) {
        case 0: return 1024;
        case 1: return 1024;
        case 2: return 2048;
        case 3: return 2048;
        default: return 1024;
    }
}

float viewDepthToReverseZNdc(float d, float n, float f)
{
    d = glm::clamp(d, n, f);
    return (n * (f - d)) / (d * (f - n));
}

std::array<glm::vec3, 8> frustumCornersWorld(
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

    std::array<glm::vec3, 8> world{};
    for (int i = 0; i < 8; ++i) {
        const glm::vec4 w = invVP * ndc[i];
        world[i] = glm::vec3(w) / w.w;
    }
    return world;
}

std::pair<glm::vec3, float> boundingSphere(const std::array<glm::vec3, 8>& pts)
{
    glm::vec3 center(0.0f);
    for (const auto& p : pts) center += p;
    center /= 8.0f;

    float radius = 0.0f;
    for (const auto& p : pts) radius = std::max(radius, glm::length(p - center));
    return { center, radius };
}

struct FrustumPlane {
    glm::vec3 normal = glm::vec3(0.0f);
    float d = 0.0f;
};

std::array<FrustumPlane, 6> extractFrustumPlanes(const glm::mat4& vp)
{
    auto row = [&](int i) -> glm::vec4 {
        return { vp[0][i], vp[1][i], vp[2][i], vp[3][i] };
    };
    const glm::vec4 r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3);

    std::array<FrustumPlane, 6> planes{};
    auto set = [&](int idx, glm::vec4 v) {
        const float len = glm::length(glm::vec3(v));
        if (len > 1e-8f) v /= len;
        planes[idx] = { glm::vec3(v), v.w };
    };
    set(0, r3 + r0);
    set(1, r3 - r0);
    set(2, r3 + r1);
    set(3, r3 - r1);
    set(4, r3 + r2);
    set(5, r3 - r2);
    return planes;
}

constexpr std::array<std::pair<int, int>, 12> kFrustumEdges{{
    {0, 2}, {0, 3}, {0, 4}, {0, 5},
    {1, 2}, {1, 3}, {1, 4}, {1, 5},
    {2, 4}, {2, 5},
    {3, 4}, {3, 5},
}};

std::vector<ShadowPass::CullingPlane> computeShadowCullPlanes(
    const glm::mat4& cascadeVP,
    const glm::vec3& lightDir)
{
    const auto frustum = extractFrustumPlanes(cascadeVP);

    std::array<bool, 6> isFrontFacing{};
    for (int i = 0; i < 6; ++i) {
        isFrontFacing[i] = glm::dot(frustum[i].normal, lightDir) > 0.0f;
    }

    std::vector<ShadowPass::CullingPlane> planes;
    planes.reserve(8);

    for (int i = 0; i < 6; ++i) {
        if (!isFrontFacing[i]) {
            planes.push_back({ frustum[i].normal, frustum[i].d });
        }
    }

    for (const auto& [fA, fB] : kFrustumEdges) {
        if (isFrontFacing[fA] == isFrontFacing[fB]) continue;

        glm::vec3 edgeDir = glm::cross(frustum[fA].normal, frustum[fB].normal);
        const float edgeLen = glm::length(edgeDir);
        if (edgeLen < 1e-8f) continue;
        edgeDir /= edgeLen;

        glm::vec3 planeNormal = glm::cross(edgeDir, lightDir);
        const float pnLen = glm::length(planeNormal);
        if (pnLen < 1e-8f) continue;
        planeNormal /= pnLen;

        int bestAxis = 0;
        float bestDot = 0.0f;
        for (int a = 0; a < 3; ++a) {
            const float ad = std::abs(edgeDir[a]);
            if (ad > bestDot) {
                bestDot = ad;
                bestAxis = a;
            }
        }

        const int c0 = (bestAxis + 1) % 3;
        const int c1 = (bestAxis + 2) % 3;
        const glm::vec3& nA = frustum[fA].normal;
        const glm::vec3& nB = frustum[fB].normal;
        const float a00 = nA[c0], a01 = nA[c1], b0 = -frustum[fA].d;
        const float a10 = nB[c0], a11 = nB[c1], b1 = -frustum[fB].d;
        glm::vec3 q(0.0f);
        const float det = a00 * a11 - a01 * a10;
        if (std::abs(det) > 1e-10f) {
            q[c0] = (b0 * a11 - b1 * a01) / det;
            q[c1] = (a00 * b1 - a10 * b0) / det;
        }

        const int frontFace = isFrontFacing[fA] ? fA : fB;
        if (glm::dot(planeNormal, frustum[frontFace].normal) < 0.0f) {
            planeNormal = -planeNormal;
        }
        planes.push_back({ planeNormal, -glm::dot(planeNormal, q) });
    }

    return planes;
}

bool aabbOutsidePlane(const glm::vec3& bmin, const glm::vec3& bmax,
                      const glm::vec3& normal, float d)
{
    const glm::vec3 pVertex(
        (normal.x >= 0.0f) ? bmax.x : bmin.x,
        (normal.y >= 0.0f) ? bmax.y : bmin.y,
        (normal.z >= 0.0f) ? bmax.z : bmin.z);
    return (glm::dot(normal, pVertex) + d) < 0.0f;
}

bool aabbSurvivesCulling(const glm::vec3& bmin, const glm::vec3& bmax,
                         const std::vector<ShadowPass::CullingPlane>& planes)
{
    for (const auto& p : planes) {
        if (aabbOutsidePlane(bmin, bmax, p.normal, p.d)) return false;
    }
    return true;
}

VkCullModeFlags toVkCullMode(int32_t mode)
{
    switch (mode) {
        case 1: return VK_CULL_MODE_FRONT_BIT;
        case 2: return VK_CULL_MODE_BACK_BIT;
        default: return VK_CULL_MODE_NONE;
    }
}

} // namespace

std::vector<uint32_t> ShadowPass::loadSpv(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open shader: " + path);
    const auto size = static_cast<size_t>(file.tellg());
    if (size == 0 || size % 4 != 0) {
        throw std::runtime_error("Invalid SPIR-V: " + path);
    }
    file.seekg(0);
    std::vector<uint32_t> code(size / 4);
    file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(size));
    return code;
}

glm::vec3 ShadowPass::safeNormalizeDirection(glm::vec3 v, glm::vec3 fallback)
{
    const float len2 = glm::dot(v, v);
    if (!std::isfinite(len2) || len2 <= 1e-8f) return glm::normalize(fallback);
    return v * glm::inversesqrt(len2);
}

ShadowPass::ShadowPass(VulkanContext& ctx)
    : m_ctx(ctx)
    , m_shadowMap(ctx)
    , m_fallbackMoments(ctx)
    , m_shadowMoments(ctx)
    , m_shadowMomentsTemp(ctx)
{
}

ShadowPass::~ShadowPass()
{
    shutdown(m_ctx.getDevice());
}

void ShadowPass::init(VkPipelineLayout sharedPipelineLayout,
                      uint32_t framesInFlight,
                      const std::string& shaderDir)
{
    m_pipelineLayout = sharedPipelineLayout;
    m_shaderDir = shaderDir;
    m_cascadeSize = resolutionForPreset(m_settings.qualityPreset);
    m_frameResources.clear();
    m_frameResources.reserve(framesInFlight);
    for (uint32_t i = 0; i < framesInFlight; ++i) {
        m_frameResources.emplace_back(m_ctx);
        m_frameResources.back().shadowUBO.createHostVisible(
            sizeof(shader_interface::ShadowCascadeUBO),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    }

    createDepthResources();
    createFallbackMoments();
    createPipelines();
    updateMatrices(0);
    updateDebugMemoryEstimate();
}

void ShadowPass::shutdown(VkDevice device)
{
    if (device == VK_NULL_HANDLE) return;

    if (m_blurPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, m_blurPipeline, nullptr);
        m_blurPipeline = VK_NULL_HANDLE;
    }
    if (m_blurPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, m_blurPipelineLayout, nullptr);
        m_blurPipelineLayout = VK_NULL_HANDLE;
    }
    if (m_blurDescriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, m_blurDescriptorPool, nullptr);
        m_blurDescriptorPool = VK_NULL_HANDLE;
        m_blurSetHorizontal = VK_NULL_HANDLE;
        m_blurSetVertical = VK_NULL_HANDLE;
    }
    if (m_blurSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, m_blurSetLayout, nullptr);
        m_blurSetLayout = VK_NULL_HANDLE;
    }

    if (m_shadowPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, m_shadowPipeline, nullptr);
        m_shadowPipeline = VK_NULL_HANDLE;
    }
    if (m_shadowAlphaPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, m_shadowAlphaPipeline, nullptr);
        m_shadowAlphaPipeline = VK_NULL_HANDLE;
    }
    if (m_shadowVsmPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, m_shadowVsmPipeline, nullptr);
        m_shadowVsmPipeline = VK_NULL_HANDLE;
    }
    if (m_shadowVsmAlphaPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, m_shadowVsmAlphaPipeline, nullptr);
        m_shadowVsmAlphaPipeline = VK_NULL_HANDLE;
    }

    destroyVsmResources(device);
    destroyDepthResources(device);

    if (m_momentsSampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, m_momentsSampler, nullptr);
        m_momentsSampler = VK_NULL_HANDLE;
    }
    m_fallbackMoments.destroy();
    m_frameResources.clear();
}

bool ShadowPass::updateSettings(const ShadowSettings& settings)
{
    ShadowSettings sanitized = settings;
    sanitized.csmLambda = glm::clamp(sanitized.csmLambda, 0.0f, 1.0f);
    sanitized.maxDistance = glm::clamp(sanitized.maxDistance, 10.0f, 300.0f);
    sanitized.filterMode = glm::clamp(sanitized.filterMode, 0, 2);
    sanitized.pcfSpreadRadius = glm::max(sanitized.pcfSpreadRadius, 0.25f);
    sanitized.vsmBleedReduction = glm::clamp(sanitized.vsmBleedReduction, 0.0f, 0.95f);
    sanitized.depthBiasConstant = glm::max(sanitized.depthBiasConstant, 0.0f);
    sanitized.depthBiasSlope = glm::max(sanitized.depthBiasSlope, 0.0f);
    sanitized.qualityPreset = glm::clamp(sanitized.qualityPreset, 0, 3);
    sanitized.cullMode = glm::clamp(sanitized.cullMode, 0, 2);

    const ShadowSettings previous = m_settings;
    const uint32_t requestedSize = resolutionForPreset(sanitized.qualityPreset);
    const bool sizeChanged = requestedSize != m_cascadeSize;
    const bool needsVsm = sanitized.filterMode == 2 && !m_vsmAllocated;
    const bool cullModeChanged = sanitized.cullMode != previous.cullMode;
    m_settings = sanitized;

    if (sizeChanged) {
        recreateForResolution(requestedSize);
    } else if (needsVsm) {
        createVsmResources();
        m_descriptorDirty = true;
    }

    if (cullModeChanged) {
        const VkDevice device = m_ctx.getDevice();
        vkDeviceWaitIdle(device);
        if (m_shadowPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, m_shadowPipeline, nullptr);
            m_shadowPipeline = VK_NULL_HANDLE;
        }
        if (m_shadowAlphaPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, m_shadowAlphaPipeline, nullptr);
            m_shadowAlphaPipeline = VK_NULL_HANDLE;
        }
        if (m_shadowVsmPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, m_shadowVsmPipeline, nullptr);
            m_shadowVsmPipeline = VK_NULL_HANDLE;
        }
        if (m_shadowVsmAlphaPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, m_shadowVsmAlphaPipeline, nullptr);
            m_shadowVsmAlphaPipeline = VK_NULL_HANDLE;
        }
        createPipelines();
    }

    updateDebugMemoryEstimate();
    return sizeChanged || needsVsm;
}

void ShadowPass::updateLightDirection(glm::vec3 lightDirection)
{
    m_lightDirection = safeNormalizeDirection(lightDirection, glm::vec3(0.577f));
}

void ShadowPass::updateCamera(const glm::mat4& view,
                              const glm::mat4& projection,
                              float nearZ,
                              float farZ,
                              const glm::vec3& cameraPos)
{
    m_viewMatrix = view;
    m_projMatrix = projection;
    m_cameraNearZ = nearZ;
    m_cameraFarZ = farZ;
    m_cameraPos = cameraPos;
}

void ShadowPass::createDepthResources()
{
    const VkDevice device = m_ctx.getDevice();
    m_shadowMap.create(m_cascadeSize, m_cascadeSize, 1,
                       VK_FORMAT_D32_SFLOAT,
                       VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                       VK_IMAGE_ASPECT_DEPTH_BIT,
                       CASCADE_COUNT);
    for (uint32_t c = 0; c < CASCADE_COUNT; ++c) {
        m_shadowLayerViews[c] = m_shadowMap.createSingleLayerView(c, VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    if (m_shadowSampler == VK_NULL_HANDLE) {
        const VkSamplerCreateInfo samplerCI{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_NEAREST,
            .minFilter = VK_FILTER_NEAREST,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
            .compareEnable = VK_FALSE,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
        };
        VK_CHECK(vkCreateSampler(device, &samplerCI, nullptr, &m_shadowSampler));
    }
}

void ShadowPass::destroyDepthResources(VkDevice device)
{
    if (m_shadowSampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, m_shadowSampler, nullptr);
        m_shadowSampler = VK_NULL_HANDLE;
    }
    for (auto& view : m_shadowLayerViews) {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, view, nullptr);
            view = VK_NULL_HANDLE;
        }
    }
    m_shadowMap.destroy();
}

void ShadowPass::createFallbackMoments()
{
    if (m_momentsSampler == VK_NULL_HANDLE) {
        const VkSamplerCreateInfo momentsSamplerCI{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = VK_FILTER_LINEAR,
            .minFilter = VK_FILTER_LINEAR,
            .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .anisotropyEnable = VK_FALSE,
            .compareEnable = VK_FALSE,
            .minLod = 0.0f,
            .maxLod = 0.0f,
            .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
        };
        VK_CHECK(vkCreateSampler(m_ctx.getDevice(), &momentsSamplerCI, nullptr, &m_momentsSampler));
    }

    if (m_fallbackMoments.getImage() != VK_NULL_HANDLE) return;

    m_fallbackMoments.create(1, 1, 1,
        VK_FORMAT_R32G32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        CASCADE_COUNT);

    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = vkutil::beginSingleUseCommands(m_ctx, pool);
    vkutil::transitionImage(cmd, m_fallbackMoments.getImage(),
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_ASPECT_COLOR_BIT);
    vkutil::endSingleUseCommands(m_ctx, pool, cmd);
}

void ShadowPass::createPipelines()
{
    const VkDevice device = m_ctx.getDevice();
    const auto shadowSpv = loadSpv(m_shaderDir + "/shadow.vert.spv");
    const auto alphaVertSpv = loadSpv(m_shaderDir + "/shadow_alpha.vert.spv");
    const auto alphaFragSpv = loadSpv(m_shaderDir + "/shadow_alpha.frag.spv");
    const auto vsmFragSpv = loadSpv(m_shaderDir + "/shadow_vsm.frag.spv");
    const auto vsmAlphaFragSpv = loadSpv(m_shaderDir + "/shadow_vsm_alpha.frag.spv");

    VkShaderModule shadowVert = makeShaderModule(device, shadowSpv);
    VkShaderModule alphaVert = makeShaderModule(device, alphaVertSpv);
    VkShaderModule alphaFrag = makeShaderModule(device, alphaFragSpv);
    VkShaderModule vsmFrag = makeShaderModule(device, vsmFragSpv);
    VkShaderModule vsmAlphaFrag = makeShaderModule(device, vsmAlphaFragSpv);

    const auto binding = Vertex::getBindingDescription();
    const auto attrs = Vertex::getAttributeDescriptions();
    const VkPipelineVertexInputStateCreateInfo vertexInput{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size()),
        .pVertexAttributeDescriptions = attrs.data(),
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
    VkPipelineRasterizationStateCreateInfo rasterization{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = toVkCullMode(m_settings.cullMode),
        .frontFace = VK_FRONT_FACE_CLOCKWISE,
        .depthBiasEnable = VK_TRUE,
        .depthBiasConstantFactor = m_settings.depthBiasConstant,
        .depthBiasClamp = 0.0f,
        .depthBiasSlopeFactor = m_settings.depthBiasSlope,
        .lineWidth = 1.0f,
    };
    const VkPipelineMultisampleStateCreateInfo multisample{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
    };
    const VkPipelineDepthStencilStateCreateInfo depthStencil{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
    };
    const VkPipelineColorBlendStateCreateInfo depthOnlyBlend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 0,
    };
    const std::array<VkDynamicState, 3> dynamicStates{
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_DEPTH_BIAS,
    };
    const VkPipelineDynamicStateCreateInfo dynamicState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
        .pDynamicStates = dynamicStates.data(),
    };
    const VkPipelineRenderingCreateInfo depthRenderingInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 0,
        .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
    };

    auto makePipeline = [&](const VkPipelineShaderStageCreateInfo* stages,
                            uint32_t stageCount,
                            const VkPipelineRenderingCreateInfo& rendering,
                            const VkPipelineColorBlendStateCreateInfo& blend,
                            VkPipeline* outPipeline) {
        const VkGraphicsPipelineCreateInfo ci{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &rendering,
            .stageCount = stageCount,
            .pStages = stages,
            .pVertexInputState = &vertexInput,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterization,
            .pMultisampleState = &multisample,
            .pDepthStencilState = &depthStencil,
            .pColorBlendState = &blend,
            .pDynamicState = &dynamicState,
            .layout = m_pipelineLayout,
            .renderPass = VK_NULL_HANDLE,
        };
        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, outPipeline));
    };

    const VkPipelineShaderStageCreateInfo shadowStage{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = shadowVert,
        .pName = "main",
    };
    makePipeline(&shadowStage, 1, depthRenderingInfo, depthOnlyBlend, &m_shadowPipeline);

    const std::array<VkPipelineShaderStageCreateInfo, 2> alphaStages{{
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = alphaVert, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = alphaFrag, .pName = "main" },
    }};
    makePipeline(alphaStages.data(), static_cast<uint32_t>(alphaStages.size()),
                 depthRenderingInfo, depthOnlyBlend, &m_shadowAlphaPipeline);

    const VkPipelineColorBlendAttachmentState vsmAttachment{
        .blendEnable = VK_FALSE,
        .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT,
    };
    const VkPipelineColorBlendStateCreateInfo vsmBlend{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &vsmAttachment,
    };
    const VkFormat momentsFormat = VK_FORMAT_R32G32_SFLOAT;
    const VkPipelineRenderingCreateInfo vsmRenderingInfo{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
        .colorAttachmentCount = 1,
        .pColorAttachmentFormats = &momentsFormat,
        .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
    };
    const std::array<VkPipelineShaderStageCreateInfo, 2> vsmStages{{
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = shadowVert, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = vsmFrag, .pName = "main" },
    }};
    makePipeline(vsmStages.data(), static_cast<uint32_t>(vsmStages.size()),
                 vsmRenderingInfo, vsmBlend, &m_shadowVsmPipeline);

    const std::array<VkPipelineShaderStageCreateInfo, 2> vsmAlphaStages{{
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = alphaVert, .pName = "main" },
        { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = vsmAlphaFrag, .pName = "main" },
    }};
    makePipeline(vsmAlphaStages.data(), static_cast<uint32_t>(vsmAlphaStages.size()),
                 vsmRenderingInfo, vsmBlend, &m_shadowVsmAlphaPipeline);

    vkDestroyShaderModule(device, shadowVert, nullptr);
    vkDestroyShaderModule(device, alphaVert, nullptr);
    vkDestroyShaderModule(device, alphaFrag, nullptr);
    vkDestroyShaderModule(device, vsmFrag, nullptr);
    vkDestroyShaderModule(device, vsmAlphaFrag, nullptr);
}

void ShadowPass::createVsmResources()
{
    if (m_vsmAllocated) return;
    const VkDevice device = m_ctx.getDevice();

    m_shadowMoments.create(m_cascadeSize, m_cascadeSize, 1,
        VK_FORMAT_R32G32_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_STORAGE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        CASCADE_COUNT);
    m_shadowMomentsTemp.create(m_cascadeSize, m_cascadeSize, 1,
        VK_FORMAT_R32G32_SFLOAT,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        CASCADE_COUNT);
    for (uint32_t c = 0; c < CASCADE_COUNT; ++c) {
        m_momentsLayerViews[c] = m_shadowMoments.createSingleLayerView(c, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = vkutil::beginSingleUseCommands(m_ctx, pool);
    vkutil::transitionImage(cmd, m_shadowMoments.getImage(),
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_ASPECT_COLOR_BIT);
    vkutil::endSingleUseCommands(m_ctx, pool, cmd);

    const std::array<VkDescriptorSetLayoutBinding, 2> blurBindings{{
        { .binding = shader_interface::shadow_blur_binding::kInputImage,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
        { .binding = shader_interface::shadow_blur_binding::kOutputImage,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .descriptorCount = 1,
          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
    }};
    const VkDescriptorSetLayoutCreateInfo blurSetLayoutCI{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = static_cast<uint32_t>(blurBindings.size()),
        .pBindings = blurBindings.data(),
    };
    VK_CHECK(vkCreateDescriptorSetLayout(device, &blurSetLayoutCI, nullptr, &m_blurSetLayout));

    const VkPushConstantRange blurPushRange{
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = shader_interface::kShadowBlurPcOffset,
        .size = 2 * sizeof(int32_t),
    };
    const VkPipelineLayoutCreateInfo blurLayoutCI{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &m_blurSetLayout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &blurPushRange,
    };
    VK_CHECK(vkCreatePipelineLayout(device, &blurLayoutCI, nullptr, &m_blurPipelineLayout));

    VkShaderModule blurModule = makeShaderModule(device, loadSpv(m_shaderDir + "/shadow_blur.comp.spv"));
    const VkComputePipelineCreateInfo blurPipelineCI{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = blurModule,
            .pName = "main",
        },
        .layout = m_blurPipelineLayout,
    };
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &blurPipelineCI,
                                      nullptr, &m_blurPipeline));
    vkDestroyShaderModule(device, blurModule, nullptr);

    const VkDescriptorPoolSize blurPoolSize{
        .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        .descriptorCount = 4,
    };
    const VkDescriptorPoolCreateInfo blurPoolCI{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 2,
        .poolSizeCount = 1,
        .pPoolSizes = &blurPoolSize,
    };
    VK_CHECK(vkCreateDescriptorPool(device, &blurPoolCI, nullptr, &m_blurDescriptorPool));

    const std::array<VkDescriptorSetLayout, 2> blurLayouts = { m_blurSetLayout, m_blurSetLayout };
    const VkDescriptorSetAllocateInfo blurAllocInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = m_blurDescriptorPool,
        .descriptorSetCount = 2,
        .pSetLayouts = blurLayouts.data(),
    };
    std::array<VkDescriptorSet, 2> blurSets{};
    VK_CHECK(vkAllocateDescriptorSets(device, &blurAllocInfo, blurSets.data()));
    m_blurSetHorizontal = blurSets[0];
    m_blurSetVertical = blurSets[1];

    const VkDescriptorImageInfo momentsInfo{
        .imageView = m_shadowMoments.getImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const VkDescriptorImageInfo tempInfo{
        .imageView = m_shadowMomentsTemp.getImageView(),
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    const std::array<VkWriteDescriptorSet, 4> writes{{
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = m_blurSetHorizontal,
          .dstBinding = shader_interface::shadow_blur_binding::kInputImage,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .pImageInfo = &momentsInfo },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = m_blurSetHorizontal,
          .dstBinding = shader_interface::shadow_blur_binding::kOutputImage,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .pImageInfo = &tempInfo },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = m_blurSetVertical,
          .dstBinding = shader_interface::shadow_blur_binding::kInputImage,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .pImageInfo = &tempInfo },
        { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = m_blurSetVertical,
          .dstBinding = shader_interface::shadow_blur_binding::kOutputImage,
          .descriptorCount = 1,
          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          .pImageInfo = &momentsInfo },
    }};
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    m_vsmAllocated = true;
    m_debugInfo.vsmAllocated = true;
    spdlog::info("VSM shadow resources allocated lazily: {}x{}x{}", m_cascadeSize, m_cascadeSize, CASCADE_COUNT);
}

void ShadowPass::destroyVsmResources(VkDevice device)
{
    for (auto& view : m_momentsLayerViews) {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, view, nullptr);
            view = VK_NULL_HANDLE;
        }
    }
    m_shadowMomentsTemp.destroy();
    m_shadowMoments.destroy();
    m_vsmAllocated = false;
    m_debugInfo.vsmAllocated = false;
}

void ShadowPass::recreateForResolution(uint32_t newSize)
{
    vkDeviceWaitIdle(m_ctx.getDevice());
    const VkDevice device = m_ctx.getDevice();

    destroyVsmResources(device);
    destroyDepthResources(device);
    m_cascadeSize = newSize;
    createDepthResources();
    if (m_settings.filterMode == 2) createVsmResources();
    m_descriptorDirty = true;
}

VkImageView ShadowPass::getMomentsImageView() const
{
    return m_vsmAllocated ? m_shadowMoments.getImageView() : m_fallbackMoments.getImageView();
}

bool ShadowPass::consumeDescriptorDirty()
{
    const bool dirty = m_descriptorDirty;
    m_descriptorDirty = false;
    return dirty;
}

void ShadowPass::updateDebugMemoryEstimate()
{
    const float cascadePixels = static_cast<float>(m_cascadeSize) * static_cast<float>(m_cascadeSize) *
                                static_cast<float>(CASCADE_COUNT);
    float bytes = cascadePixels * 4.0f;
    if (m_vsmAllocated) bytes += cascadePixels * 8.0f * 2.0f;
    m_debugInfo.estimatedMemoryMB = bytes / (1024.0f * 1024.0f);
    m_debugInfo.cascadeResolution = m_cascadeSize;
    m_debugInfo.cascadeCount = CASCADE_COUNT;
    m_debugInfo.vsmAllocated = m_vsmAllocated;
    m_debugInfo.maxDistance = m_settings.maxDistance;
}

VkBuffer ShadowPass::getShadowUBOBuffer(uint32_t frameIndex) const
{
    if (frameIndex >= m_frameResources.size()) {
        return VK_NULL_HANDLE;
    }
    return m_frameResources[frameIndex].shadowUBO.getBuffer();
}

void ShadowPass::updateMatrices(uint32_t frameIndex)
{
    if (frameIndex >= m_frameResources.size()) {
        return;
    }

    const glm::vec3 lightDir = safeNormalizeDirection(m_lightDirection, glm::vec3(0.577f));
    const glm::vec3 up = (std::abs(lightDir.y) > 0.9f)
        ? glm::vec3(1.0f, 0.0f, 0.0f)
        : glm::vec3(0.0f, 1.0f, 0.0f);

    const float n = glm::max(m_cameraNearZ, 0.05f);
    float f = glm::min(m_cameraFarZ, m_settings.maxDistance);
    f = glm::max(f, n + 1.0f);
    const float lambda = glm::clamp(m_settings.csmLambda, 0.0f, 1.0f);

    std::array<float, CASCADE_COUNT + 1> splitDepths{};
    splitDepths[0] = n;
    splitDepths[CASCADE_COUNT] = f;
    for (uint32_t i = 1; i < CASCADE_COUNT; ++i) {
        const float t = static_cast<float>(i) / CASCADE_COUNT;
        const float logSplit = n * std::pow(f / n, t);
        const float uniformSplit = n + (f - n) * t;
        splitDepths[i] = lambda * logSplit + (1.0f - lambda) * uniformSplit;
    }

    shader_interface::ShadowCascadeUBO ubo{};
    ubo.splitDepths = { splitDepths[1], splitDepths[2], splitDepths[3], splitDepths[4] };

    for (uint32_t c = 0; c < CASCADE_COUNT; ++c) {
        const float cNear = splitDepths[c];
        const float cFar = splitDepths[c + 1];
        const float cNearNDC = viewDepthToReverseZNdc(cNear, n, f);
        const float cFarNDC = viewDepthToReverseZNdc(cFar, n, f);
        const auto corners = frustumCornersWorld(m_viewMatrix, m_projMatrix, cNearNDC, cFarNDC);
        auto [center, radius] = boundingSphere(corners);
        radius = glm::max(radius, 0.01f);

        const glm::mat4 lightViewSnap = glm::lookAt(center + lightDir, center, up);
        const float texelSize = glm::max((2.0f * radius) / static_cast<float>(m_cascadeSize), 1e-6f);
        glm::vec4 centerLS = lightViewSnap * glm::vec4(center, 1.0f);
        centerLS.x = std::floor(centerLS.x / texelSize) * texelSize;
        centerLS.y = std::floor(centerLS.y / texelSize) * texelSize;
        const glm::vec3 snappedCenter = glm::vec3(glm::inverse(lightViewSnap) * centerLS);

        const glm::mat4 lightView = glm::lookAt(snappedCenter + lightDir * radius,
                                                snappedCenter,
                                                up);
        glm::mat4 lightProj = glm::orthoRH_ZO(-radius, radius, -radius, radius, 0.0f, 2.0f * radius);
        lightProj[1][1] *= -1.0f;
        ubo.lightViewProj[c] = lightProj * lightView;

        if (m_settings.enableCasterCulling) {
            const float tanHalfFov = 1.0f / std::abs(m_projMatrix[1][1]);
            const float aspect = std::abs(m_projMatrix[1][1]) / std::abs(m_projMatrix[0][0]);
            glm::mat4 stdProj = glm::perspectiveRH_NO(2.0f * std::atan(tanHalfFov), aspect, cNear, cFar);
            stdProj[1][1] *= -1.0f;
            m_shadowCullPlanes[c] = computeShadowCullPlanes(stdProj * m_viewMatrix, lightDir);
        } else {
            m_shadowCullPlanes[c].clear();
        }
    }

    m_lastShadowUBO = ubo;
    m_debugInfo.splitDepths = ubo.splitDepths;
    m_debugInfo.maxDistance = f;
    m_frameResources[frameIndex].shadowUBO.upload(&ubo, sizeof(ubo));
}

void ShadowPass::record(VkCommandBuffer cmd,
                        VkDescriptorSet sceneDescriptorSet,
                        uint32_t frameIndex,
                        const Buffer& vertexBuffer,
                        const Buffer& indexBuffer,
                        const std::vector<DrawCommand>& drawCommands,
                        const std::vector<SceneDrawBounds>& drawBounds,
                        const std::vector<MeshRenderData>& meshes,
                        const std::vector<Material>& materials,
                        const std::vector<VkDescriptorSet>& materialSets,
                        GPUTimer& gpuTimer)
{
    if (sceneDescriptorSet == VK_NULL_HANDLE || frameIndex >= m_frameResources.size()) {
        return;
    }

    updateMatrices(frameIndex);
    if (m_settings.filterMode == 2 && !m_vsmAllocated) {
        createVsmResources();
        m_descriptorDirty = true;
    }

    vkutil::transitionImage(cmd, m_shadowMap.getImage(),
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    const bool isVsm = (m_settings.filterMode == 2 && m_vsmAllocated);
    if (isVsm) {
        vkutil::transitionImage(cmd, m_shadowMoments.getImage(),
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_ASPECT_COLOR_BIT);
    }

    gpuTimer.writeTimestamp(cmd, "ShadowPass_Begin");

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout,
                            shader_interface::kSceneSet, 1, &sceneDescriptorSet, 0, nullptr);
    const VkBuffer vertexBuf = vertexBuffer.getBuffer();
    const VkDeviceSize zeroOffset = 0;
    vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuf, &zeroOffset);
    vkCmdBindIndexBuffer(cmd, indexBuffer.getBuffer(), 0, VK_INDEX_TYPE_UINT32);

    const VkExtent2D shadowExtent{ m_cascadeSize, m_cascadeSize };
    const VkViewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(m_cascadeSize),
        .height = static_cast<float>(m_cascadeSize),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    const VkRect2D scissor{ .offset = { 0, 0 }, .extent = shadowExtent };

    for (uint32_t c = 0; c < CASCADE_COUNT; ++c) {
        const VkRenderingAttachmentInfo depthAtt{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = m_shadowLayerViews[c],
            .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = { .depthStencil = { 1.0f, 0 } },
        };
        const VkRenderingAttachmentInfo momentsAtt{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = isVsm ? m_momentsLayerViews[c] : VK_NULL_HANDLE,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = { .color = { .float32 = { 0.0f, 0.0f, 0.0f, 0.0f } } },
        };
        const VkRenderingInfo renderInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = { .offset = { 0, 0 }, .extent = shadowExtent },
            .layerCount = 1,
            .colorAttachmentCount = isVsm ? 1u : 0u,
            .pColorAttachments = isVsm ? &momentsAtt : nullptr,
            .pDepthAttachment = &depthAtt,
        };

        vkCmdBeginRendering(cmd, &renderInfo);
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        vkCmdSetScissor(cmd, 0, 1, &scissor);
        vkCmdSetDepthBias(cmd, m_settings.depthBiasConstant, 0.0f, m_settings.depthBiasSlope);
        vkCmdPushConstants(cmd, m_pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           shader_interface::kCascadeIndexPcOffset,
                           sizeof(uint32_t), &c);

        uint32_t culled = 0;
        uint32_t drawn = 0;
        VkPipeline boundPipeline = VK_NULL_HANDLE;
        auto bindPipeline = [&](VkPipeline pipeline) {
            if (boundPipeline != pipeline) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                boundPipeline = pipeline;
            }
        };

        for (size_t drawIndex = 0; drawIndex < drawCommands.size(); ++drawIndex) {
            const DrawCommand& draw = drawCommands[drawIndex];
            if (!draw.mesh.isValid() || draw.mesh.index >= meshes.size()) continue;
            const MeshRenderData& mesh = meshes[draw.mesh.index];
            const SceneDrawBounds* bounds =
                (drawIndex < drawBounds.size()) ? &drawBounds[drawIndex] : nullptr;
            if (m_settings.enableCasterCulling &&
                bounds != nullptr &&
                !m_shadowCullPlanes[c].empty() &&
                !aabbSurvivesCulling(bounds->worldMin, bounds->worldMax, m_shadowCullPlanes[c])) {
                ++culled;
                continue;
            }

            const bool alphaMasked =
                draw.material.isValid() &&
                draw.material.index < materials.size() &&
                materials[draw.material.index].alphaMode == Material::AlphaMode::Mask &&
                draw.material.index < materialSets.size();

            bindPipeline(isVsm
                ? (alphaMasked ? m_shadowVsmAlphaPipeline : m_shadowVsmPipeline)
                : (alphaMasked ? m_shadowAlphaPipeline : m_shadowPipeline));

            vkCmdPushConstants(cmd, m_pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                               shader_interface::kModelMatrixPcOffset, sizeof(glm::mat4), &draw.transform);

            if (alphaMasked) {
                const VkDescriptorSet materialSet = materialSets[draw.material.index];
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                        m_pipelineLayout,
                                        shader_interface::kMaterialSet, 1, &materialSet, 0, nullptr);
                const Material& mat = materials[draw.material.index];
                const shader_interface::MaterialPushConstants matPC{
                    .baseColorFactor = mat.baseColorFactor,
                    .metallicFactor = mat.metallicFactor,
                    .roughnessFactor = mat.roughnessFactor,
                    .alphaCutoff = mat.alphaCutoff,
                    .alphaCoverageMode = 0.0f,
                };
                vkCmdPushConstants(cmd, m_pipelineLayout,
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                                   shader_interface::kMaterialPcOffset, sizeof(matPC), &matPC);
            }

            vkCmdDrawIndexed(cmd, mesh.indexCount, 1, mesh.firstIndex, 0, 0);
            ++drawn;
        }

        m_debugInfo.culledMeshes[c] = culled;
        m_debugInfo.totalMeshes[c] = static_cast<uint32_t>(drawCommands.size());
        m_debugInfo.drawnMeshes[c] = drawn;
        vkCmdEndRendering(cmd);
    }

    gpuTimer.writeTimestamp(cmd, "ShadowPass_End");

    vkutil::transitionImage(cmd, m_shadowMap.getImage(),
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL,
        VK_IMAGE_ASPECT_DEPTH_BIT);

    if (!isVsm) return;

    vkutil::transitionImage(cmd, m_shadowMoments.getImage(),
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_ASPECT_COLOR_BIT);
    vkutil::transitionImage(cmd, m_shadowMomentsTemp.getImage(),
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_IMAGE_ASPECT_COLOR_BIT);

    gpuTimer.writeTimestamp(cmd, "BlurPass_Begin");
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_blurPipeline);

    struct BlurPC { int32_t direction; int32_t radius; };
    BlurPC blurPC{ 0, 3 };
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_blurPipelineLayout, shader_interface::kShadowBlurSet,
                            1, &m_blurSetHorizontal, 0, nullptr);
    vkCmdPushConstants(cmd, m_blurPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       shader_interface::kShadowBlurPcOffset, sizeof(BlurPC), &blurPC);
    vkCmdDispatch(cmd, (m_cascadeSize + 15) / 16, (m_cascadeSize + 15) / 16, CASCADE_COUNT);

    const VkMemoryBarrier2 computeBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT,
    };
    const VkDependencyInfo depInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &computeBarrier,
    };
    vkCmdPipelineBarrier2(cmd, &depInfo);

    blurPC.direction = 1;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            m_blurPipelineLayout, shader_interface::kShadowBlurSet,
                            1, &m_blurSetVertical, 0, nullptr);
    vkCmdPushConstants(cmd, m_blurPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       shader_interface::kShadowBlurPcOffset, sizeof(BlurPC), &blurPC);
    vkCmdDispatch(cmd, (m_cascadeSize + 15) / 16, (m_cascadeSize + 15) / 16, CASCADE_COUNT);
    gpuTimer.writeTimestamp(cmd, "BlurPass_End");

    vkutil::transitionImage(cmd, m_shadowMoments.getImage(),
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_IMAGE_ASPECT_COLOR_BIT);
}
