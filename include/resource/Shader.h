#pragma once

#include "core/VulkanContext.h"
#include <vector>
#include <string>

struct ShaderModule {
    VkShaderModule        module = VK_NULL_HANDLE;
    VkShaderStageFlagBits stage  = VK_SHADER_STAGE_VERTEX_BIT;
};

class ShaderCompiler {
public:
    ShaderCompiler() = delete; // static-only utility class

    // Compile GLSL source string to SPIR-V words targeting Vulkan 1.4 / SPIR-V 1.6.
    static std::vector<uint32_t> compileGLSL(const char* glslSource,
                                             VkShaderStageFlagBits stage,
                                             const char* entryPoint = "main");

    // Create a VkShaderModule from pre-compiled SPIR-V words.
    static VkShaderModule createModule(VulkanContext& ctx,
                                       const std::vector<uint32_t>& spirv);

    // Load a pre-compiled .spv file and wrap it in a ShaderModule.
    static ShaderModule loadFromFile(VulkanContext& ctx,
                                     const char* spirvPath,
                                     VkShaderStageFlagBits stage);

    // Destroy a shader module when it is no longer needed (after pipeline creation).
    static void destroyModule(VulkanContext& ctx, ShaderModule& mod);
};
