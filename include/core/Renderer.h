#pragma once

#include "core/VulkanContext.h"
#include "core/Swapchain.h"
#include "core/CommandBuffer.h"
#include "core/DrawCommand.h"
#include "core/FrameSync.h"
#include "core/MeshRenderData.h"
#include "core/RenderFramePacket.h"
#include "core/RenderScenePacket.h"
#include "core/RenderSettings.h"
#include "core/RendererOverlay.h"
#include "core/ShaderInterface.h"
#include "debug/GPUTimer.h"
#include "debug/Screenshot.h"
#include "renderpasses/ShadowPass.h"
#include "renderpasses/SkyPass.h"
#include "renderpasses/TonemapPass.h"
#include "renderpasses/ClusteredLightCullingPass.h"
#include "resource/Buffer.h"
#include "resource/Image.h"
#include "resource/Model.h"
#include "resource/RendererResourceManager.h"
#include <glm/glm.hpp>
#include <array>
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

    // Wait for frame slot, collect GPU timer results, reset command buffer, acquire image.
    // Returns false if the frame should be skipped (e.g. window minimized).
    // Caller must NOT call endFrame() when beginFrame() returns false.
    bool beginFrame();

    // Record commands, submit, present, advance frame indices.
    void endFrame(RendererOverlay* overlay = nullptr);

    void submitFrame(const RenderFramePacket& packet);
    // Sticky scene submission remains active until the caller submits a replacement packet.
    void submitScene(RenderScenePacket scene);

    // Upload CPU import data into renderer-owned GPU resources. Import/reload
    // policy remains application-side; Renderer only owns GPU resource state.
    void uploadModelResources(const Model& model);

    void resize(uint32_t width, uint32_t height);

    // Phase 4.3: cascade shadow map constants
    static constexpr uint32_t CASCADE_COUNT = shader_interface::kCascadeCount;
    static constexpr uint32_t CASCADE_SIZE  = 1024;

    // Phase 4.2.5: shadow culling stats
    const std::array<uint32_t, CASCADE_COUNT>& getShadowCulledMeshes() const { return m_shadowPass.getDebugInfo().culledMeshes; }
    const std::array<uint32_t, CASCADE_COUNT>& getShadowTotalMeshes()  const { return m_shadowPass.getDebugInfo().totalMeshes; }
    const ShadowDebugInfo& getShadowDebugInfo() const { return m_shadowPass.getDebugInfo(); }
    void loadHdrPanorama(const std::string& path);

    // Phase 2.6: render statistics — populated each frame in render()
    struct RenderStats {
        uint32_t drawCalls          = 0;
        uint32_t triangles          = 0;
        uint32_t opaqueDrawCalls    = 0;
        uint32_t maskedDrawCalls    = 0;
        uint32_t meshCount          = 0;
        uint32_t materialCount      = 0;
        uint32_t textureCount       = 0;
        size_t   textureMemoryBytes = 0;   // approximate GPU texture memory usage
        AntiAliasingMode aaMode = AntiAliasingMode::None;
        uint32_t activeSampleCount = 1;
        bool     sampleShadingEnabled = false;
        float    minSampleShading = 0.0f;
        bool     alphaToCoverageEnabled = false;
        size_t   estimatedMsaaAttachmentMemoryBytes = 0;
        float    sceneGpuMs = 0.0f;
        float    tonemapGpuMs = 0.0f;
        float    totalGpuFrameMs = 0.0f;
    };
    const RenderStats& getRenderStats() const { return m_renderStats; }

    // Phase 2.6: GPU timer — read timing results from the previous completed frame
    const GPUTimer& getGPUTimer() const { return m_gpuTimer; }

    // Phase 2.6: queue a screenshot capture at the end of the current frame
    void requestScreenshot(const std::string& filename = "");

    // Persistent renderer configuration: AA is not frame-local because sample
    // count affects attachments, pipelines, resolves, and resize resources.
    void setAntiAliasingSettings(const AntiAliasingSettings& settings);
    const AntiAliasingSettings& getAntiAliasingSettings() const { return m_antiAliasingSettings; }
    const AntiAliasingStatus& getAntiAliasingStatus() const { return m_antiAliasingStatus; }

private:
    static constexpr uint32_t kFramesInFlight = 3;

    // Push constant: bytes 64-95 in the pipeline layout (after 64-byte model matrix)
    using MaterialPushConstants = shader_interface::MaterialPushConstants;

    // Camera UBO (host-visible, updated every frame)
    // Phase 6.5: added inverseVP between projection and cameraPos for sky ray reconstruction.
    using CameraUBO = shader_interface::CameraUBO;

    // Directional light UBO (host-visible, std140: 48 bytes = 3×vec4)
    using LightUBO = shader_interface::LightUBO;
    using ShadowCascadeUBO = shader_interface::ShadowCascadeUBO;
    using ClusterMetadataUBO = shader_interface::ClusterMetadataUBO;

    // Frame-owned resources that are rewritten as the current slot advances.
    struct FrameResources {
        explicit FrameResources(VulkanContext& ctx)
            : cameraUBO(ctx)
            , lightUBO(ctx)
            , clusterMetadataUBO(ctx)
        {
        }

        FrameResources(FrameResources&&) noexcept = default;
        FrameResources& operator=(FrameResources&&) noexcept = default;
        FrameResources(const FrameResources&) = delete;
        FrameResources& operator=(const FrameResources&) = delete;

        Buffer cameraUBO;
        Buffer lightUBO;
        Buffer clusterMetadataUBO;
        // Frame-owned scene descriptor set binds this slot's UBOs plus shared scene inputs.
        VkDescriptorSet sceneDescriptorSet = VK_NULL_HANDLE;
    };

    struct ActiveSceneState {
        RenderScenePacket submission;
        std::vector<SceneDrawBounds> drawBounds;
    };

    void render(RendererOverlay* overlay);
    void handleResize();
    void handleResize(uint32_t width, uint32_t height);
    void createResizeDependentResources();
    void destroyResizeDependentResources();
    void recreateResizeDependentResources();
    void refreshResizeDependentBindings();
    void createPbrPipeline();
    void createPbrGraphicsPipelines();
    void destroyPbrGraphicsPipelines();
    void destroyPipeline();
    void destroyRendererDescriptorPools();
    AntiAliasingStatus resolveAntiAliasingSettings(const AntiAliasingSettings& settings) const;
    void logAntiAliasingStatus() const;
    bool isMsaaEnabled() const;
    bool isAlphaToCoverageActive() const;
    VkSampleCountFlagBits getActiveSceneSampleCount() const;
    VkImageView getActiveSceneColorAttachmentView() const;
    VkImageView getResolvedHdrView() const;
    bool isDrawAlphaMasked(const DrawCommand& draw) const;
    VkPipeline selectSolidPipelineForDraw(const DrawCommand& draw) const;
    size_t estimateMsaaAttachmentMemoryBytes() const;
    void refreshTimingStats();
    void refreshRenderStats();
    void createFrameResources();
    void applyFrameSubmission(const RenderFramePacket& packet);
    void rebuildActiveSceneDrawBounds();
    void uploadCurrentFrameCameraState(const CameraData& camera);
    void uploadCurrentFrameLightState();
    void uploadActiveScenePointLights();
    void uploadCurrentClusterMetadata();
    void uploadClusterMetadataForAllFrames();

    static std::vector<uint32_t> loadSpv(const std::string& path);

    FrameResources& currentFrameResources();
    const FrameResources& currentFrameResources() const;
    CameraUBO buildCurrentCameraUBO() const;
    LightUBO buildCurrentLightUBO() const;
    ClusterMetadataUBO buildCurrentClusterMetadata() const;

    VulkanContext& m_ctx;
    Swapchain&     m_swapchain;
    CommandBuffer  m_commandBuffer;
    FrameSync      m_frameSync;

    // Renderer-persistent uploaded geometry/resources.
    RendererResourceManager m_resources;

    // Sticky scene submission state derived from caller-owned scene packets.
    ActiveSceneState m_activeScene;

    // Frame-slot resources.
    std::vector<FrameResources> m_frameResources;

    // Renderer-shared descriptor layouts and PBR pipeline state.
    VkPipeline            m_pipeline           = VK_NULL_HANDLE;
    VkPipeline            m_maskedPipeline     = VK_NULL_HANDLE;
    VkPipeline            m_maskedA2cPipeline  = VK_NULL_HANDLE;
    VkPipeline            m_wireframePipeline  = VK_NULL_HANDLE;
    VkPipeline            m_normalsPipeline    = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout     = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_sceneSetLayout     = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_materialSetLayout  = VK_NULL_HANDLE;
    VkShaderModule        m_vertModule         = VK_NULL_HANDLE;
    VkShaderModule        m_fragModule         = VK_NULL_HANDLE;
    VkShaderModule        m_normalsFragModule  = VK_NULL_HANDLE;

    // Renderer-owned descriptor pools/sets.
    VkDescriptorPool m_frameSceneDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorPool m_materialDescriptorPool   = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> m_materialDescriptorSets;

    // Scene-submission and clustered-light GPU resources.
    Buffer           m_pointLightBuffer;
    Buffer           m_clusterGridBuffer;
    Buffer           m_clusterLightIndexBuffer;
    uint32_t         m_maxPointLights = 1024;
    uint32_t         m_clusterCountX = 0;
    uint32_t         m_clusterCountY = 0;
    uint32_t         m_clusterCountZ = shader_interface::kClusterZSlices;
    uint32_t         m_clusterCount = 0;

    // Resize-dependent scene attachments. Tonemap always samples m_resolvedHdrTarget.
    Image            m_sceneDepthTarget;
    Image            m_msaaHdrTarget;
    Image            m_resolvedHdrTarget;
    VkSampler        m_hdrSampler = VK_NULL_HANDLE;

    // Extracted passes keep their pass-local descriptors/pipelines internally.
    TonemapPass               m_tonemapPass;
    ShadowPass                m_shadowPass;
    ClusteredLightCullingPass m_clusteredLightCullingPass;
    SkyPass                   m_skyPass;

    // Cached from the last submitFrame() so per-frame uploads stay explicit and traceable.
    ShadowSettings  m_shadowSettings;
    glm::vec3       m_lightDirection   = glm::vec3(1.0f, 1.0f, 1.0f);
    glm::vec3       m_lightColor       = glm::vec3(1.0f);
    float           m_lightIntensity   = 1.0f;
    float           m_ambientIntensity = 0.1f;
    glm::mat4       m_viewMatrix       = glm::mat4(1.0f);
    glm::mat4       m_projMatrix       = glm::mat4(1.0f);
    float           m_cameraNearZ      = 0.01f;
    float           m_cameraFarZ       = 1000.0f;
    glm::vec3       m_cameraPos        = glm::vec3(0.0f);
    float   m_exposure         = 0.0f;  // EV offset; 0 = no change
    int32_t m_tonemapMode      = 0;     // 0=Reinhard, 1=AgX, 2=PBR Neutral
    int32_t m_splitScreenMode  = 0;     // 0=off, 1=on
    int32_t m_splitRightMode   = 1;     // comparison operator for right half (default AgX)
    bool    m_skyEnabled = true;   // sky drawn by default
    int32_t m_skyMode    = 0;      // 0 = Procedural (Rayleigh+Mie), 1 = HDR Panorama
    bool    m_wireframe   = false;
    bool    m_showNormals = false;

    AntiAliasingSettings m_antiAliasingSettings;
    AntiAliasingStatus   m_antiAliasingStatus;
    bool                 m_loggedSampleShadingFallback = false;

    // Phase 2.6: profiling and stats
    GPUTimer     m_gpuTimer;
    Screenshot   m_screenshot;
    RenderStats  m_renderStats;

    bool        m_screenshotRequested = false;
    std::string m_screenshotFilename;

    void createClusteredLightResources();
    void destroyClusteredLightResources();
    void resizeClusteredLightResources();
    void createFrameSceneDescriptorPool();
    void allocateFrameSceneDescriptorSets();
    void rewriteFrameSceneShadowBindings();
    void rewriteFrameSceneClusterBindings();
    void createMaterialDescriptorPool();
    void allocateMaterialDescriptorSets();
    void createSceneDepthTarget();
    void createSceneHdrTargets();
};
