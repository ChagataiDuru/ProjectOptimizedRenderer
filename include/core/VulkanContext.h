#pragma once

#include <volk.h>
#include <vk_mem_alloc.h>
#include <cstdint>
#include <vector>

// Throws std::runtime_error with call site on Vulkan failure.
#define VK_CHECK(call)                                                              \
    do {                                                                            \
        VkResult _vkr = (call);                                                     \
        if (_vkr != VK_SUCCESS) {                                                   \
            throw std::runtime_error(                                               \
                std::string("VkResult ") + std::to_string(static_cast<int>(_vkr))  \
                + " from " #call " at " __FILE__ ":" + std::to_string(__LINE__));   \
        }                                                                           \
    } while (0)

struct RendererDeviceFeatures {
    bool dynamicRendering = false;
    bool dynamicRenderingLocalRead = false;
    bool synchronization2 = false;
    bool pushDescriptor = false;

    bool fragmentShadingRate = false;
    bool meshShader = false;
    bool taskShader = false;
    bool descriptorIndexing = false;
    bool timelineSemaphore = false;
};

struct RendererMeshShaderProperties {
    uint32_t maxTaskWorkGroupInvocations = 0;
    uint32_t maxMeshWorkGroupInvocations = 0;
    uint32_t maxMeshOutputVertices = 0;
    uint32_t maxMeshOutputPrimitives = 0;
    uint32_t maxPreferredTaskWorkGroupInvocations = 0;
    uint32_t maxPreferredMeshWorkGroupInvocations = 0;
    uint32_t maxMeshMultiviewViewCount = 0;
};

class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();

    VulkanContext(const VulkanContext&)            = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;
    VulkanContext(VulkanContext&&)                 = delete;
    VulkanContext& operator=(VulkanContext&&)      = delete;

    void init();
    void shutdown();

    VkInstance       getInstance()            const { return m_instance; }
    VkDevice         getDevice()              const { return m_device; }
    VkPhysicalDevice getPhysicalDevice()      const { return m_physicalDevice; }
    VkQueue          getGraphicsQueue()       const { return m_graphicsQueue; }
    uint32_t         getGraphicsQueueFamily() const { return m_graphicsQueueFamily; }
    VmaAllocator     getAllocator()           const { return m_allocator; }

    const RendererDeviceFeatures& getDeviceFeatures() const { return m_deviceFeatures; }
    const RendererMeshShaderProperties& getMeshShaderProperties() const { return m_meshShaderProperties; }

    bool hasFeature_DynamicRenderingLocalRead() const;
    bool hasFeature_PushDescriptor()            const;
    void logDeviceInfo()                        const;

private:
    void createInstance();
    void selectPhysicalDevice();
    void createLogicalDevice();

    bool                               isDeviceSuitable(VkPhysicalDevice device);
    bool                               checkPortabilitySubset(VkPhysicalDevice device);
    std::vector<VkExtensionProperties> getDeviceExtensions(VkPhysicalDevice device) const;
    RendererDeviceFeatures             queryRendererDeviceFeatures(VkPhysicalDevice device) const;
    RendererMeshShaderProperties       queryMeshShaderProperties(VkPhysicalDevice device,
                                                                 const RendererDeviceFeatures& features) const;

    VkInstance               m_instance      = VK_NULL_HANDLE;
    VkPhysicalDevice         m_physicalDevice = VK_NULL_HANDLE;
    VkDevice                 m_device         = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VmaAllocator             m_allocator      = nullptr;

    VkQueue  m_graphicsQueue = VK_NULL_HANDLE;
    VkQueue  m_computeQueue  = VK_NULL_HANDLE;
    VkQueue  m_transferQueue = VK_NULL_HANDLE;

    uint32_t m_graphicsQueueFamily = UINT32_MAX;
    uint32_t m_computeQueueFamily  = UINT32_MAX;
    uint32_t m_transferQueueFamily = UINT32_MAX;

    // Feature chain: Features2 -> Vk14 -> Vk13 -> Vk12
    VkPhysicalDeviceProperties2      m_deviceProps{};
    VkPhysicalDeviceVulkan12Features m_vulkan12Features{};
    VkPhysicalDeviceVulkan13Features m_vulkan13Features{};
    VkPhysicalDeviceVulkan14Features m_vulkan14Features{};
    VkPhysicalDeviceFeatures2        m_enabledFeatures{};

    RendererDeviceFeatures       m_deviceFeatures{};
    RendererMeshShaderProperties m_meshShaderProperties{};
};
