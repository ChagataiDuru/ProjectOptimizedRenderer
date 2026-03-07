#pragma once

#include "core/VulkanContext.h"
#include "core/Swapchain.h"
#include "core/CommandBuffer.h"
#include "core/FrameSync.h"
#include <vector>
#include <string>

class Renderer {
public:
    Renderer(VulkanContext& ctx, Swapchain& swapchain);
    ~Renderer();

    Renderer(const Renderer&)            = delete;
    Renderer& operator=(const Renderer&) = delete;

    void init();
    void shutdown();

    // Wait for frame slot, reset command buffer, acquire swapchain image.
    void beginFrame();

    // Record commands, submit, present, advance frame indices.
    void endFrame();

private:
    void render();
    void createTrianglePipeline();
    void destroyPipeline();

    static std::vector<uint32_t> loadSpv(const std::string& path);

    VulkanContext& m_ctx;
    Swapchain&     m_swapchain;
    CommandBuffer  m_commandBuffer;
    FrameSync      m_frameSync;

    VkPipeline       m_trianglePipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout   = VK_NULL_HANDLE;
    // Shader modules are destroyed after pipeline creation; kept as VK_NULL_HANDLE thereafter.
    VkShaderModule   m_vertModule       = VK_NULL_HANDLE;
    VkShaderModule   m_fragModule       = VK_NULL_HANDLE;
};
