#pragma once

#include "core/VulkanContext.h"
#include "core/Swapchain.h"
#include "core/CommandBuffer.h"
#include "core/FrameSync.h"
#include "resource/Buffer.h"
#include <glm/glm.hpp>
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

    // Placeholder: Phase 1.2 will wire this to a per-frame UBO upload.
    void setCameraMatrices(const glm::mat4& view, const glm::mat4& projection);

private:
    void render();
    void createPbrPipeline();
    void createGeometry();
    void destroyPipeline();

    static std::vector<uint32_t> loadSpv(const std::string& path);

    VulkanContext& m_ctx;
    Swapchain&     m_swapchain;
    CommandBuffer  m_commandBuffer;
    FrameSync      m_frameSync;

    // Phase 1.1: geometry buffers (initialized in constructor list — Buffer has no default ctor)
    Buffer   m_vertexBuffer;
    Buffer   m_indexBuffer;
    uint32_t m_indexCount = 0;

    // Phase 1.1: pipeline resources
    VkPipeline            m_pipeline        = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout  = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_cameraSetLayout = VK_NULL_HANDLE;
    // Shader modules are destroyed after pipeline creation; kept as VK_NULL_HANDLE thereafter.
    VkShaderModule        m_vertModule      = VK_NULL_HANDLE;
    VkShaderModule        m_fragModule      = VK_NULL_HANDLE;

    // Phase 1.2: camera UBO — host-visible so we can memcpy each frame without staging
    struct CameraUBO {
        glm::mat4 view;
        glm::mat4 projection;
    };

    Buffer           m_cameraUBOBuffer;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet  m_descriptorSet  = VK_NULL_HANDLE;

    void createCameraUBO();
    void createDescriptorPool();
    void createDescriptorSet();
};
