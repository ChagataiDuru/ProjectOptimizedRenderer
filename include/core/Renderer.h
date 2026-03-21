#pragma once

#include "core/VulkanContext.h"
#include "core/Swapchain.h"
#include "core/CommandBuffer.h"
#include "core/FrameSync.h"
#include "debug/GPUTimer.h"
#include "debug/Screenshot.h"
#include "resource/Buffer.h"
#include "resource/Image.h"
#include "resource/Model.h"
#include "resource/SceneInfo.h"
#include "resource/Texture.h"
#include "resource/SamplerCache.h"
#include <glm/glm.hpp>
#include <array>
#include <string>
#include <vector>

class ImGuiManager;  // forward declare — full header pulled in by Renderer.cpp

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
    void endFrame();

    void setCameraMatrices(const glm::mat4& view, const glm::mat4& projection,
                           const glm::vec3& cameraPos);

    // Phase 4.3: expose camera frustum planes so CSM splits stay in sync
    void setCameraFrustum(float nearZ, float farZ);

    void setLightParameters(const glm::vec3& direction, const glm::vec3& color,
                            float intensity, float ambient);

    // Phase 4.3: CSM controls
    void setCsmLambda(float lambda)            { m_csmLambda = lambda; }
    void setCascadeDebugEnabled(bool enabled);
    bool isCascadeDebugEnabled() const         { return m_showCascadeDebug; }

    // Phase 4.4-4.6: shadow filter mode and per-mode settings
    void setShadowFilterMode(int32_t mode);    // 0=None, 1=PCF, 2=VSM
    void setPcfSpreadRadius(float radius);
    void setVsmBleedReduction(float reduction);
    int32_t getShadowFilterMode() const        { return m_shadowFilterMode; }

    // Unload current model and load a new glTF at the given path.
    // Destroys model-dependent GPU resources, loads new model, recreates descriptors.
    // Must be called on the main thread while no frames are in flight.
    void reloadModel(const std::string& modelPath);

    // Phase 2.5: wire an ImGuiManager to receive an overlay render pass each frame
    void setImGuiManager(ImGuiManager* mgr) { m_imguiManager = mgr; }

    // Read-only access for UI panels
    const Model&     getModel()     const { return m_model; }
    const SceneInfo& getSceneInfo() const { return m_sceneInfo; }

    // Rendering toggles — stored here, applied when pipeline variants are added
    void setWireframeEnabled(bool enabled) { m_wireframe   = enabled; }
    void setNormalVisualization(bool enabled) { m_showNormals = enabled; }
    bool isWireframeEnabled()     const { return m_wireframe; }
    bool isNormalVisualization()  const { return m_showNormals; }

    // Phase 4.3: cascade shadow map constants
    static constexpr uint32_t CASCADE_COUNT = 4;
    static constexpr uint32_t CASCADE_SIZE  = 2048;

    // Phase 5/6: tone mapping controls
    void setExposure(float ev)           { m_exposure = ev; }
    float getExposure()   const          { return m_exposure; }
    int32_t getTonemapMode() const       { return m_tonemapMode; }

    // Phase 6: unified setter for all tonemap params (pushed as one PC struct each frame)
    void setTonemapParams(int32_t mode, float exposure, bool splitScreen, int32_t splitRightMode) {
        m_tonemapMode      = mode;
        m_exposure         = exposure;
        m_splitScreenMode  = splitScreen ? 1 : 0;
        m_splitRightMode   = splitRightMode;
    }

    // Phase 2.6: render statistics — populated each frame in render()
    struct RenderStats {
        uint32_t drawCalls          = 0;
        uint32_t triangles          = 0;
        uint32_t meshCount          = 0;
        uint32_t materialCount      = 0;
        uint32_t textureCount       = 0;
        size_t   textureMemoryBytes = 0;   // approximate GPU texture memory usage
    };
    const RenderStats& getRenderStats() const { return m_renderStats; }

    // Phase 2.6: GPU timer — read timing results from the previous completed frame
    const GPUTimer& getGPUTimer() const { return m_gpuTimer; }

    // Phase 2.6: queue a screenshot capture at the end of the current frame
    void requestScreenshot(const std::string& filename = "");

private:
    void render();
    void handleResize();
    void createPbrPipeline();
    void loadModel(const std::string& modelPath);
    void destroyPipeline();

    static std::vector<uint32_t> loadSpv(const std::string& path);

    VulkanContext& m_ctx;
    Swapchain&     m_swapchain;
    CommandBuffer  m_commandBuffer;
    FrameSync      m_frameSync;

    // Geometry buffers
    Buffer m_vertexBuffer;
    Buffer m_indexBuffer;

    // Phase 1.3: model and per-mesh draw data
    struct MeshRenderData {
        uint32_t firstIndex;
        uint32_t indexCount;
        int32_t  materialIndex = -1;
    };
    Model                       m_model;
    SceneInfo                   m_sceneInfo;    // Phase 3.7: normalization transform
    std::vector<MeshRenderData> m_meshRenderData;

    // Pipeline resources
    VkPipeline            m_pipeline           = VK_NULL_HANDLE;
    VkPipeline            m_wireframePipeline  = VK_NULL_HANDLE;
    VkPipeline            m_normalsPipeline    = VK_NULL_HANDLE;
    VkPipelineLayout      m_pipelineLayout     = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_cameraSetLayout    = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_materialSetLayout  = VK_NULL_HANDLE;
    VkShaderModule        m_vertModule         = VK_NULL_HANDLE;
    VkShaderModule        m_fragModule         = VK_NULL_HANDLE;
    VkShaderModule        m_normalsFragModule  = VK_NULL_HANDLE;

    // Material descriptor sets (one per Model::materials entry)
    std::vector<VkDescriptorSet> m_materialSets;

    // Push constant: bytes 64-95 in the pipeline layout (after 64-byte model matrix)
    struct MaterialPushConstants {
        glm::vec4 baseColorFactor = glm::vec4(1.0f);
        float     metallicFactor  = 1.0f;
        float     roughnessFactor = 1.0f;
        float     alphaCutoff     = 0.0f;   // 0 = no cutoff (opaque); >0 = discard below this
        float     _pad            = 0.0f;
    };

    // Camera UBO (host-visible, updated every frame)
    struct CameraUBO {
        glm::mat4 view;
        glm::mat4 projection;
        glm::vec3 cameraPos;
        float     _pad = 0.0f;  // std140: vec3 pads to 16 bytes
    };

    // Directional light UBO (host-visible, std140: 48 bytes = 3×vec4)
    struct LightUBO {
        glm::vec3 lightDirection;
        float     lightIntensity;
        glm::vec3 lightColor;
        float     ambientIntensity;
        uint32_t  debugCascades;     // 0 = off, 1 = false-color overlay
        uint32_t  shadowFilterMode;  // 0=None, 1=PCF, 2=VSM
        float     pcfSpreadRadius;   // PCF sample spread in texels
        float     vsmBleedReduction; // VSM light bleeding reduction factor
    };  // 48 bytes: 2×(vec3+float) + 4×uint32/float

    Buffer           m_cameraUBOBuffer;
    Buffer           m_lightUBOBuffer;

    // Texture infrastructure
    SamplerCache         m_samplerCache;
    std::vector<Texture> m_textures;
    Texture              m_fallbackWhite;

    // Depth buffer (D32_SFLOAT, reverse-Z)
    Image            m_depthImage;

    // Phase 5: HDR offscreen target (R16G16B16A16_SFLOAT, swapchain-sized)
    Image            m_hdrTarget;
    VkSampler        m_hdrSampler           = VK_NULL_HANDLE;

    // Phase 5: Tone map pass resources
    VkDescriptorSetLayout m_tonemapSetLayout      = VK_NULL_HANDLE;
    VkDescriptorPool      m_tonemapPool            = VK_NULL_HANDLE;
    VkDescriptorSet       m_tonemapSet             = VK_NULL_HANDLE;
    VkPipelineLayout      m_tonemapPipelineLayout  = VK_NULL_HANDLE;
    VkPipeline            m_tonemapPipeline        = VK_NULL_HANDLE;

    // Phase 5/6: tone map state (uploaded as TonemapPC push constant each frame)
    float   m_exposure         = 0.0f;  // EV offset; 0 = no change
    int32_t m_tonemapMode      = 0;     // 0=Reinhard, 1=AgX, 2=PBR Neutral
    int32_t m_splitScreenMode  = 0;     // 0=off, 1=on
    int32_t m_splitRightMode   = 1;     // comparison operator for right half (default AgX)

    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet  m_descriptorSet  = VK_NULL_HANDLE;

    // Phase 2.5: optional ImGui overlay
    ImGuiManager*    m_imguiManager   = nullptr;

    // Phase 2.6: profiling and stats
    GPUTimer     m_gpuTimer;
    Screenshot   m_screenshot;
    RenderStats  m_renderStats;

    // Phase 4.3: cascaded shadow map resources
    struct ShadowCascadeUBO {
        glm::mat4 lightViewProj[CASCADE_COUNT];
        glm::vec4 splitDepths;   // x=c0, y=c1, z=c2, w=c3 (view-space |Z|)
    };
    Image            m_shadowMap;           // D32_SFLOAT 2D_ARRAY, CASCADE_COUNT layers
    std::array<VkImageView, CASCADE_COUNT> m_shadowLayerViews = {};  // per-layer attachment views
    VkSampler        m_shadowSampler    = VK_NULL_HANDLE;
    Buffer           m_shadowUBOBuffer;
    VkPipeline       m_shadowPipeline   = VK_NULL_HANDLE;
    VkShaderModule   m_shadowVertModule = VK_NULL_HANDLE;
    glm::vec3        m_lightDirection   = glm::vec3(1.0f, 1.0f, 1.0f);

    // Stored to re-upload LightUBO when debug toggle changes
    glm::vec3 m_lightColor        = glm::vec3(1.0f);
    float     m_lightIntensity    = 1.0f;
    float     m_ambientIntensity  = 0.1f;

    // Camera frustum — updated by setCameraFrustum() for CSM split computation
    glm::mat4 m_viewMatrix  = glm::mat4(1.0f);
    glm::mat4 m_projMatrix  = glm::mat4(1.0f);
    float     m_cameraNearZ = 0.01f;
    float     m_cameraFarZ  = 1000.0f;

    // CSM UI state
    float m_csmLambda        = 0.5f;
    bool  m_showCascadeDebug = false;

    // Phase 4.4-4.6: VSM moment images (RG32_SFLOAT, CASCADE_COUNT layers each)
    Image m_shadowMoments;       // color attachment (shadow pass) + sampled (PBR pass)
    Image m_shadowMomentsTemp;   // blur intermediate (compute only)
    std::array<VkImageView, CASCADE_COUNT> m_momentsLayerViews = {};  // per-cascade color attach
    VkSampler  m_momentsSampler    = VK_NULL_HANDLE;  // LINEAR filter for bilinear VSM
    VkPipeline m_shadowVsmPipeline = VK_NULL_HANDLE;  // shadow pass: depth + moments output

    // Separable Gaussian blur (compute)
    VkPipeline            m_blurPipeline       = VK_NULL_HANDLE;
    VkPipelineLayout      m_blurPipelineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_blurSetLayout      = VK_NULL_HANDLE;
    VkDescriptorPool      m_blurDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet       m_blurSetHorizontal  = VK_NULL_HANDLE;  // moments→temp
    VkDescriptorSet       m_blurSetVertical    = VK_NULL_HANDLE;  // temp→moments

    // Shadow filter state (uploaded to LightUBO, used by pbr.frag)
    int32_t m_shadowFilterMode    = 0;    // 0=None, 1=PCF, 2=VSM
    float   m_pcfSpreadRadius     = 2.0f; // PCF sample spread in texels
    float   m_vsmBleedReduction   = 0.2f; // VSM light bleeding reduction

    bool        m_screenshotRequested = false;
    std::string m_screenshotFilename;

    // Rendering toggles (UI-driven; applied when pipeline variants are added)
    bool m_wireframe   = false;
    bool m_showNormals = false;

    glm::vec3 m_cameraPos = glm::vec3(0.0f);

    void createCameraUBO();
    void createLightUBO();
    void uploadLightUBO();   // re-upload m_lightUBOBuffer from stored params
    void createDepthImage();
    void createHdrTarget();  // Phase 5: create/recreate HDR offscreen target + update tonemap set
    void createTonemapPipeline();
    void updateTonemapDescriptorSet();  // called by createHdrTarget() after image (re)creation
    void createDescriptorPool();
    void createDescriptorSet();
    void createMaterialDescriptorSets();
    void createShadowResources();
    void updateShadowMatrices();
};
