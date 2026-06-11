#include "renderpasses/ClusteredLightCullingPass.h"

#include "core/ShaderInterface.h"

#include <spdlog/spdlog.h>

#include <fstream>
#include <stdexcept>

namespace {

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

} // namespace

ClusteredLightCullingPass::ClusteredLightCullingPass(VulkanContext& ctx)
    : m_ctx(ctx)
{
}

ClusteredLightCullingPass::~ClusteredLightCullingPass()
{
    shutdown(m_ctx.getDevice());
}

std::vector<uint32_t> ClusteredLightCullingPass::loadSpv(const std::string& path)
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

void ClusteredLightCullingPass::init(VkDescriptorSetLayout sceneSetLayout,
                                     const std::string& shaderDir)
{
    const VkPipelineLayoutCreateInfo layoutCI{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &sceneSetLayout,
    };
    VK_CHECK(vkCreatePipelineLayout(m_ctx.getDevice(), &layoutCI, nullptr, &m_pipelineLayout));

    VkShaderModule computeModule = makeShaderModule(
        m_ctx.getDevice(),
        loadSpv(shaderDir + "/cluster_cull.comp.spv"));

    const VkComputePipelineCreateInfo pipelineCI{
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = computeModule,
            .pName = "main",
        },
        .layout = m_pipelineLayout,
    };
    VK_CHECK(vkCreateComputePipelines(m_ctx.getDevice(), VK_NULL_HANDLE, 1,
                                      &pipelineCI, nullptr, &m_pipeline));
    vkDestroyShaderModule(m_ctx.getDevice(), computeModule, nullptr);

    spdlog::info("Clustered light culling pass created");
}

void ClusteredLightCullingPass::record(VkCommandBuffer cmd,
                                       VkDescriptorSet sceneDescriptorSet,
                                       uint32_t clusterCountX,
                                       uint32_t clusterCountY,
                                       uint32_t clusterCountZ) const
{
    if (m_pipeline == VK_NULL_HANDLE || sceneDescriptorSet == VK_NULL_HANDLE ||
        clusterCountX == 0 || clusterCountY == 0 || clusterCountZ == 0) {
        return;
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout,
                            shader_interface::kSceneSet, 1, &sceneDescriptorSet, 0, nullptr);
    vkCmdDispatch(cmd, clusterCountX, clusterCountY, clusterCountZ);
}

void ClusteredLightCullingPass::shutdown(VkDevice device)
{
    if (device == VK_NULL_HANDLE) return;

    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
}
