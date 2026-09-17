#pragma once

#include "core/VulkanContext.h"
#include "resource/Image.h"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

class GPUTimer;

// Image-based lighting from the current sky (ART-LGT-004).
//
// Owns three sampled images that the PBR shader reads through the scene set:
//   - irradiance: 32x16 equirect, cosine-convolved (diffuse = albedo * sample)
//   - prefiltered: 256x128 equirect with 6 mips, GGX roughness = mip / 5
//   - BRDF LUT: 128x128 split-sum (scale, bias), baked once at init
// The environment is re-captured and re-filtered on the GPU whenever the sky source
// changes (sky mode, panorama, or procedural sun direction/intensity).
class IblPass {
public:
    static constexpr uint32_t kEnvironmentWidth = 256;
    static constexpr uint32_t kEnvironmentHeight = 128;
    static constexpr uint32_t kIrradianceWidth = 32;
    static constexpr uint32_t kIrradianceHeight = 16;
    static constexpr uint32_t kPrefilteredMips = 6;
    static constexpr uint32_t kBrdfLutSize = 128;

    explicit IblPass(VulkanContext& ctx);
    ~IblPass();

    IblPass(const IblPass&) = delete;
    IblPass& operator=(const IblPass&) = delete;

    // Creates images and pipelines, bakes the BRDF LUT and an initial procedural
    // environment so the sampled images are valid before the first frame.
    void init(const std::string& shaderDir,
              VkImageView panoramaView, VkSampler panoramaSampler,
              uint32_t panoramaWidth);
    void shutdown(VkDevice device);

    // Marks the environment dirty when the sky source differs from the last bake.
    void updateSource(int32_t skyMode, const glm::vec3& sunDirection, float sunIntensity,
                      uint64_t panoramaGeneration);
    bool needsBake() const { return m_dirty; }

    // Records the environment capture + filtering into cmd (outside any render pass).
    void bake(VkCommandBuffer cmd, VkImageView panoramaView, VkSampler panoramaSampler,
              uint32_t panoramaWidth, GPUTimer* timer);

    VkImageView irradianceView() const { return m_irradiance.getImageView(); }
    VkImageView prefilteredView() const { return m_prefiltered.getImageView(); }
    VkImageView brdfLutView() const { return m_brdfLut.getImageView(); }
    VkSampler   environmentSampler() const { return m_environmentSampler; }
    VkSampler   lutSampler() const { return m_lutSampler; }
    float       prefilteredMaxLod() const { return static_cast<float>(kPrefilteredMips - 1); }
    uint32_t    bakeCount() const { return m_bakeCount; }

private:
    struct SourceState {
        int32_t skyMode = -1;
        glm::vec3 sunDirection = glm::vec3(0.0f);
        float sunIntensity = -1.0f;
        uint64_t panoramaGeneration = ~0ull;
    };

    VkPipeline createComputePipeline(const std::string& path, uint32_t pushConstantSize,
                                     VkPipelineLayout& outLayout);
    void writeSet(VkDescriptorSet set, VkImageView storageView, VkImageView sampledView,
                  VkSampler sampler);
    void bakeBrdfLut(VkCommandBuffer cmd);
    static std::vector<uint32_t> loadSpv(const std::string& path);

    VulkanContext& m_ctx;
    std::string m_shaderDir;

    Image m_environment;
    Image m_irradiance;
    Image m_prefiltered;
    Image m_brdfLut;
    VkImageView m_environmentLevel0View = VK_NULL_HANDLE;
    std::array<VkImageView, kPrefilteredMips> m_prefilteredLevelViews{};

    VkSampler m_environmentSampler = VK_NULL_HANDLE;
    VkSampler m_lutSampler = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_setLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_captureSet = VK_NULL_HANDLE;
    VkDescriptorSet m_irradianceSet = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kPrefilteredMips> m_prefilterSets{};
    VkDescriptorSet m_lutSet = VK_NULL_HANDLE;

    VkPipelineLayout m_captureLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_irradianceLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_prefilterLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_lutLayout = VK_NULL_HANDLE;
    VkPipeline m_capturePipeline = VK_NULL_HANDLE;
    VkPipeline m_irradiancePipeline = VK_NULL_HANDLE;
    VkPipeline m_prefilterPipeline = VK_NULL_HANDLE;
    VkPipeline m_lutPipeline = VK_NULL_HANDLE;

    SourceState m_requested;
    SourceState m_baked;
    bool m_dirty = true;
    uint32_t m_bakeCount = 0;
};
