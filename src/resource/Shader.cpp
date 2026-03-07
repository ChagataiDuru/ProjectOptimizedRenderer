#include "resource/Shader.h"

#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/SPIRV/GlslangToSpv.h>
#include <spdlog/spdlog.h>

#include <fstream>
#include <stdexcept>
#include <string>

// ── glslang stage mapping ─────────────────────────────────────────────────────

static EShLanguage toGlslangStage(VkShaderStageFlagBits stage)
{
    switch (stage) {
        case VK_SHADER_STAGE_VERTEX_BIT:                  return EShLangVertex;
        case VK_SHADER_STAGE_FRAGMENT_BIT:                return EShLangFragment;
        case VK_SHADER_STAGE_COMPUTE_BIT:                 return EShLangCompute;
        case VK_SHADER_STAGE_GEOMETRY_BIT:                return EShLangGeometry;
        case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:    return EShLangTessControl;
        case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT: return EShLangTessEvaluation;
        default:
            throw std::runtime_error("ShaderCompiler: unsupported VkShaderStageFlagBits");
    }
}

// glslang process-level init is idempotent but must be called before any compilation.
static void ensureGlslangInitialized()
{
    static bool initialized = false;
    if (!initialized) {
        glslang::InitializeProcess();
        initialized = true;
    }
}

// ── Compilation ───────────────────────────────────────────────────────────────

std::vector<uint32_t> ShaderCompiler::compileGLSL(const char* glslSource,
                                                   VkShaderStageFlagBits stage,
                                                   const char* /*entryPoint*/)
{
    ensureGlslangInitialized();

    const EShLanguage glslangStage = toGlslangStage(stage);
    glslang::TShader shader(glslangStage);

    shader.setStrings(&glslSource, 1);

    // Target Vulkan 1.4 / SPIR-V 1.6
    shader.setEnvInput(glslang::EShSourceGlsl, glslangStage,
                       glslang::EShClientVulkan, 460);
    shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_3);
    shader.setEnvTarget(glslang::EShTargetSpv,    glslang::EShTargetSpv_1_6);

    const EShMessages messages = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
    if (!shader.parse(GetDefaultResources(), 460, false, messages)) {
        throw std::runtime_error(std::string("GLSL parse error:\n") + shader.getInfoLog());
    }

    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(messages)) {
        throw std::runtime_error(std::string("GLSL link error:\n") + program.getInfoLog());
    }

    std::vector<uint32_t> spirv;
    glslang::SpvOptions spvOptions{};
    spvOptions.generateDebugInfo = false;
    spvOptions.disableOptimizer  = false;
    glslang::GlslangToSpv(*program.getIntermediate(glslangStage), spirv, &spvOptions);

    spdlog::debug("ShaderCompiler: compiled {} bytes of GLSL → {} SPIR-V words",
                  std::strlen(glslSource), spirv.size());
    return spirv;
}

// ── Module creation ───────────────────────────────────────────────────────────

VkShaderModule ShaderCompiler::createModule(VulkanContext& ctx,
                                            const std::vector<uint32_t>& spirv)
{
    const VkShaderModuleCreateInfo ci{
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirv.size() * sizeof(uint32_t),
        .pCode    = spirv.data(),
    };
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(ctx.getDevice(), &ci, nullptr, &mod));
    return mod;
}

ShaderModule ShaderCompiler::loadFromFile(VulkanContext& ctx,
                                          const char* spirvPath,
                                          VkShaderStageFlagBits stage)
{
    std::ifstream file(spirvPath, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error(std::string("Cannot open SPIR-V: ") + spirvPath);

    const auto byteSize = static_cast<size_t>(file.tellg());
    if (byteSize == 0 || byteSize % 4 != 0)
        throw std::runtime_error(std::string("Invalid SPIR-V (size not multiple of 4): ") + spirvPath);

    file.seekg(0);
    std::vector<uint32_t> code(byteSize / 4);
    file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(byteSize));

    spdlog::debug("ShaderCompiler: loaded {} ({} bytes)", spirvPath, byteSize);
    return { createModule(ctx, code), stage };
}

void ShaderCompiler::destroyModule(VulkanContext& ctx, ShaderModule& mod)
{
    if (mod.module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(ctx.getDevice(), mod.module, nullptr);
        mod.module = VK_NULL_HANDLE;
    }
}
