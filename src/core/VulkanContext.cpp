#include "core/VulkanContext.h"

#include <SDL3/SDL_vulkan.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr VkFormat kSceneHdrFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat kSceneDepthFormat = VK_FORMAT_D32_SFLOAT;
constexpr VkSampleCountFlags kSceneSampleCountMask =
    VK_SAMPLE_COUNT_1_BIT |
    VK_SAMPLE_COUNT_2_BIT |
    VK_SAMPLE_COUNT_4_BIT |
    VK_SAMPLE_COUNT_8_BIT;

VkSampleCountFlags queryImageFormatSampleCounts(VkPhysicalDevice device,
                                                VkFormat format,
                                                VkImageUsageFlags usage)
{
    VkImageFormatProperties props{};
    const VkResult result = vkGetPhysicalDeviceImageFormatProperties(
        device,
        format,
        VK_IMAGE_TYPE_2D,
        VK_IMAGE_TILING_OPTIMAL,
        usage,
        0,
        &props);
    if (result != VK_SUCCESS) {
        return 0;
    }
    return props.sampleCounts;
}

std::array<bool, 4> sampleCountFlagsToArray(VkSampleCountFlags counts)
{
    return {
        (counts & VK_SAMPLE_COUNT_1_BIT) != 0,
        (counts & VK_SAMPLE_COUNT_2_BIT) != 0,
        (counts & VK_SAMPLE_COUNT_4_BIT) != 0,
        (counts & VK_SAMPLE_COUNT_8_BIT) != 0,
    };
}

std::string sampleCountArrayToString(const std::array<bool, 4>& counts)
{
    const std::array<const char*, 4> labels = { "1x", "2x", "4x", "8x" };
    std::string result;
    for (size_t i = 0; i < counts.size(); ++i) {
        if (!counts[i]) {
            continue;
        }
        if (!result.empty()) {
            result += ", ";
        }
        result += labels[i];
    }
    return result.empty() ? "none" : result;
}

} // namespace

static bool hasExtension(const std::vector<VkExtensionProperties>& exts, const char* name)
{
    return std::any_of(exts.begin(), exts.end(), [name](const VkExtensionProperties& e) {
        return strcmp(e.extensionName, name) == 0;
    });
}

static bool anyFragmentShadingRateFeatureEnabled(
    const VkPhysicalDeviceFragmentShadingRateFeaturesKHR& features)
{
    return features.pipelineFragmentShadingRate == VK_TRUE ||
           features.primitiveFragmentShadingRate == VK_TRUE ||
           features.attachmentFragmentShadingRate == VK_TRUE;
}

static void appendDeviceExtension(std::vector<const char*>& extensions, const char* name)
{
    if (std::find(extensions.begin(), extensions.end(), name) == extensions.end()) {
        extensions.push_back(name);
    }
}

// ── Debug messenger callback ──────────────────────────────────────────────────

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
    VkDebugUtilsMessageTypeFlagsEXT             /*type*/,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*                                       /*userData*/)
{
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
        spdlog::error("[Vulkan] {}", data->pMessage);
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        spdlog::warn("[Vulkan] {}", data->pMessage);
    else
        spdlog::debug("[Vulkan] {}", data->pMessage);
    return VK_FALSE;
}

// ── Lifecycle ─────────────────────────────────────────────────────────────────

VulkanContext::VulkanContext() = default;

VulkanContext::~VulkanContext()
{
    shutdown();
}

void VulkanContext::init()
{
    createInstance();
    selectPhysicalDevice();
    createLogicalDevice();
    logDeviceInfo();
}

void VulkanContext::shutdown()
{
    if (m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
    }
    if (m_allocator != nullptr) {
        vmaDestroyAllocator(m_allocator);
        m_allocator = nullptr;
    }
    if (m_device != VK_NULL_HANDLE) {
        vkDestroyDevice(m_device, nullptr);
        m_device = VK_NULL_HANDLE;
    }
    if (m_debugMessenger != VK_NULL_HANDLE) {
        vkDestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
        m_debugMessenger = VK_NULL_HANDLE;
    }
    if (m_instance != VK_NULL_HANDLE) {
        vkDestroyInstance(m_instance, nullptr);
        m_instance = VK_NULL_HANDLE;
    }
}

// ── Instance ──────────────────────────────────────────────────────────────────

void VulkanContext::createInstance()
{
    VK_CHECK(volkInitialize());

    const VkApplicationInfo appInfo{
        .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName   = "ProjectOptimizedRenderer",
        .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
        .pEngineName        = "POR",
        .engineVersion      = VK_MAKE_VERSION(0, 1, 0),
        .apiVersion         = VK_API_VERSION_1_4,
    };

    // ── Build instance extensions list (platform-dependent) ──────────────────
    // SDL3 can tell us which surface extensions it needs, but we also add
    // debug utils and any portability extensions explicitly.
    std::vector<const char*> extensions = {
        VK_KHR_SURFACE_EXTENSION_NAME,
    };

#ifndef NDEBUG
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

#if defined(__APPLE__)
    // MoltenVK requires portability enumeration and the Metal surface extension
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    extensions.push_back("VK_EXT_metal_surface");
#elif defined(_WIN32)
    // Native Vulkan on Windows — Win32 surface extension
    extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#else
    // Linux / other — SDL will pick wayland or xcb at runtime,
    // but we still need the surface extension which is already added above.
    // SDL_Vulkan_GetInstanceExtensions can be used for dynamic query.
#endif

    // Build layer list: enumerate what's actually available to avoid VK_ERROR_LAYER_NOT_PRESENT.
    std::vector<const char*> validationLayers;
#ifndef NDEBUG
    {
        uint32_t layerCount = 0;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
        std::vector<VkLayerProperties> available(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, available.data());

        const char* khronosValidation = "VK_LAYER_KHRONOS_validation";
        bool found = std::any_of(available.begin(), available.end(),
            [&](const VkLayerProperties& l) {
                return strcmp(l.layerName, khronosValidation) == 0;
            });

        if (found) {
            validationLayers.push_back(khronosValidation);
        } else {
            spdlog::warn("VK_LAYER_KHRONOS_validation not available — "
                         "install the LunarG Vulkan SDK for GPU validation. "
                         "Continuing without validation.");
        }
    }
#endif

#ifndef NDEBUG
    const VkDebugUtilsMessengerCreateInfoEXT debugInfo{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType =
            VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = debugCallback,
    };
#endif

    VkInstanceCreateFlags instanceFlags = 0;
#if defined(__APPLE__)
    // Required for MoltenVK portability enumeration
    instanceFlags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#endif

    const VkInstanceCreateInfo createInfo{
        .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
#ifndef NDEBUG
        // Chain debug messenger so validation covers instance creation/destruction too
        .pNext                   = &debugInfo,
#else
        .pNext                   = nullptr,
#endif
        .flags                   = instanceFlags,
        .pApplicationInfo        = &appInfo,
        .enabledLayerCount       = static_cast<uint32_t>(validationLayers.size()),
        .ppEnabledLayerNames     = validationLayers.data(),
        .enabledExtensionCount   = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &m_instance));
    volkLoadInstance(m_instance);

#ifndef NDEBUG
    VK_CHECK(vkCreateDebugUtilsMessengerEXT(m_instance, &debugInfo, nullptr, &m_debugMessenger));
#endif
}

// ── Physical device selection ─────────────────────────────────────────────────

std::vector<VkExtensionProperties> VulkanContext::getDeviceExtensions(VkPhysicalDevice device) const
{
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, exts.data());
    return exts;
}

RendererDeviceFeatures VulkanContext::queryRendererDeviceFeatures(VkPhysicalDevice device) const
{
    // Capabilities are detected once up front and then cached in
    // RendererDeviceFeatures so optional systems query centralized state instead
    // of chaining new Vulkan feature probes later.
    const std::vector<VkExtensionProperties> exts = getDeviceExtensions(device);
    const bool hasFragmentShadingRateExt = hasExtension(exts, VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);
    const bool hasMeshShaderExt = hasExtension(exts, VK_EXT_MESH_SHADER_EXTENSION_NAME);

    VkPhysicalDeviceFragmentShadingRateFeaturesKHR fragmentShadingRateFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR
    };
    VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT
    };
    VkPhysicalDeviceVulkan12Features vk12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features vk13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &vk12};
    VkPhysicalDeviceVulkan14Features vk14{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES, &vk13};

    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &vk14};

    if (hasFragmentShadingRateExt) {
        fragmentShadingRateFeatures.pNext = features.pNext;
        features.pNext = &fragmentShadingRateFeatures;
    }
    if (hasMeshShaderExt) {
        meshShaderFeatures.pNext = features.pNext;
        features.pNext = &meshShaderFeatures;
    }

    vkGetPhysicalDeviceFeatures2(device, &features);

    RendererDeviceFeatures result{};
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(device, &props);

    const VkSampleCountFlags colorFormatCounts = queryImageFormatSampleCounts(
        device,
        kSceneHdrFormat,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
    const VkSampleCountFlags depthFormatCounts = queryImageFormatSampleCounts(
        device,
        kSceneDepthFormat,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
    const VkSampleCountFlags framebufferCounts =
        props.limits.framebufferColorSampleCounts &
        props.limits.framebufferDepthSampleCounts;
    const VkSampleCountFlags sceneCounts =
        colorFormatCounts &
        depthFormatCounts &
        framebufferCounts &
        kSceneSampleCountMask;

    result.dynamicRendering = vk13.dynamicRendering == VK_TRUE;
    result.dynamicRenderingLocalRead = vk14.dynamicRenderingLocalRead == VK_TRUE;
    result.synchronization2 = vk13.synchronization2 == VK_TRUE;
    result.pushDescriptor = vk14.pushDescriptor == VK_TRUE;
    result.fragmentShadingRate = hasFragmentShadingRateExt &&
        anyFragmentShadingRateFeatureEnabled(fragmentShadingRateFeatures);
    result.meshShader = hasMeshShaderExt && meshShaderFeatures.meshShader == VK_TRUE;
    result.taskShader = hasMeshShaderExt && meshShaderFeatures.taskShader == VK_TRUE;
    result.descriptorIndexing = vk12.descriptorIndexing == VK_TRUE;
    result.timelineSemaphore = vk12.timelineSemaphore == VK_TRUE;
    result.sampleRateShading = features.features.sampleRateShading == VK_TRUE;
    result.supportedSceneSampleCounts = sampleCountFlagsToArray(sceneCounts);
    return result;
}

RendererMeshShaderProperties VulkanContext::queryMeshShaderProperties(
    VkPhysicalDevice device,
    const RendererDeviceFeatures& features) const
{
    RendererMeshShaderProperties result{};
    if (!features.meshShader) {
        return result;
    }

    VkPhysicalDeviceMeshShaderPropertiesEXT meshProps{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT
    };
    VkPhysicalDeviceProperties2 props{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        &meshProps
    };
    vkGetPhysicalDeviceProperties2(device, &props);

    result.maxTaskWorkGroupInvocations = meshProps.maxTaskWorkGroupInvocations;
    result.maxMeshWorkGroupInvocations = meshProps.maxMeshWorkGroupInvocations;
    result.maxMeshOutputVertices = meshProps.maxMeshOutputVertices;
    result.maxMeshOutputPrimitives = meshProps.maxMeshOutputPrimitives;
    result.maxPreferredTaskWorkGroupInvocations = meshProps.maxPreferredTaskWorkGroupInvocations;
    result.maxPreferredMeshWorkGroupInvocations = meshProps.maxPreferredMeshWorkGroupInvocations;
    result.maxMeshMultiviewViewCount = meshProps.maxMeshMultiviewViewCount;
    return result;
}

bool VulkanContext::checkPortabilitySubset(VkPhysicalDevice device)
{
    // VK_KHR_portability_subset is mandatory on MoltenVK but doesn't exist
    // on native Vulkan drivers (NVIDIA, AMD, Intel on Windows/Linux).
    auto exts = getDeviceExtensions(device);
    return hasExtension(exts, "VK_KHR_portability_subset");
}

bool VulkanContext::isDeviceSuitable(VkPhysicalDevice device)
{
    // Must support a graphics queue
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &qfCount, qfs.data());

    bool hasGraphics = std::any_of(qfs.begin(), qfs.end(), [](const VkQueueFamilyProperties& q) {
        return (q.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
    });
    if (!hasGraphics) return false;

    const RendererDeviceFeatures features = queryRendererDeviceFeatures(device);

    return features.dynamicRendering &&
           features.synchronization2;
}

void VulkanContext::selectPhysicalDevice()
{
    uint32_t deviceCount = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr));
    if (deviceCount == 0)
        throw std::runtime_error("No Vulkan-capable GPU found");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    VK_CHECK(vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data()));

    VkPhysicalDevice best  = VK_NULL_HANDLE;
    int              bestScore = -1;

    for (auto dev : devices) {
        if (!isDeviceSuitable(dev)) continue;

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(dev, &props);

        int score = 0;
        // Discrete GPUs (NVIDIA, AMD) score highest
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            score += 1000;
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
            score += 500;

        const RendererDeviceFeatures features = queryRendererDeviceFeatures(dev);

        if (features.pushDescriptor) score += 100;
        if (features.fragmentShadingRate) score += 50;
        if (features.meshShader) score += 25;

        spdlog::info("  GPU candidate: {} | type {} | score {}",
            props.deviceName, static_cast<int>(props.deviceType), score);

        if (score > bestScore) {
            bestScore = score;
            best      = dev;
        }
    }

    if (best == VK_NULL_HANDLE)
        throw std::runtime_error(
            "No suitable GPU found: Vulkan 1.3+ dynamicRendering and synchronization2 "
            "are required by the current renderer baseline");

    m_physicalDevice = best;
    m_deviceFeatures = queryRendererDeviceFeatures(m_physicalDevice);
    m_meshShaderProperties = queryMeshShaderProperties(m_physicalDevice, m_deviceFeatures);

    m_deviceProps = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    vkGetPhysicalDeviceProperties2(m_physicalDevice, &m_deviceProps);
}

// ── Logical device ────────────────────────────────────────────────────────────

void VulkanContext::createLogicalDevice()
{
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &qfCount, qfs.data());

    for (uint32_t i = 0; i < qfCount; ++i) {
        const auto flags = qfs[i].queueFlags;

        if (m_graphicsQueueFamily == UINT32_MAX && (flags & VK_QUEUE_GRAPHICS_BIT))
            m_graphicsQueueFamily = i;

        // Prefer a queue family that is compute-only (no graphics).
        // NVIDIA GPUs typically expose dedicated async compute families.
        if (m_computeQueueFamily == UINT32_MAX &&
            (flags & VK_QUEUE_COMPUTE_BIT) && !(flags & VK_QUEUE_GRAPHICS_BIT))
            m_computeQueueFamily = i;

        // Prefer a queue family that is transfer-only.
        // NVIDIA GPUs expose a dedicated transfer/copy engine family.
        if (m_transferQueueFamily == UINT32_MAX &&
            (flags & VK_QUEUE_TRANSFER_BIT) &&
            !(flags & VK_QUEUE_GRAPHICS_BIT) &&
            !(flags & VK_QUEUE_COMPUTE_BIT))
            m_transferQueueFamily = i;
    }

    // Fall back to graphics family if dedicated queues aren't available
    // (e.g. MoltenVK on Apple Silicon exposes one unified family)
    if (m_computeQueueFamily  == UINT32_MAX) m_computeQueueFamily  = m_graphicsQueueFamily;
    if (m_transferQueueFamily == UINT32_MAX) m_transferQueueFamily = m_graphicsQueueFamily;

    const float priority = 1.0f;
    const std::set<uint32_t> uniqueFamilies = {
        m_graphicsQueueFamily, m_computeQueueFamily, m_transferQueueFamily
    };

    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    queueInfos.reserve(uniqueFamilies.size());
    for (uint32_t family : uniqueFamilies) {
        queueInfos.push_back({
            .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = family,
            .queueCount       = 1,
            .pQueuePriorities = &priority,
        });
    }

    // Build enabled feature chain: Features2 -> Vk14 -> Vk13 -> Vk12
    m_vulkan12Features = {
        .sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
    };
    m_vulkan12Features.descriptorIndexing = m_deviceFeatures.descriptorIndexing ? VK_TRUE : VK_FALSE;
    m_vulkan12Features.scalarBlockLayout = VK_TRUE;   // Needed for std430 in GLSL.
    m_vulkan12Features.timelineSemaphore = m_deviceFeatures.timelineSemaphore ? VK_TRUE : VK_FALSE;
    m_vulkan12Features.bufferDeviceAddress = VK_TRUE; // Required for VMA buffer device address mode.

    m_vulkan13Features = {
        .sType                          = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext                          = &m_vulkan12Features,
        .shaderDemoteToHelperInvocation = VK_TRUE,  // Required for SPIR-V 1.6 OpDemoteToHelperInvocation (discard)
        .synchronization2               = VK_TRUE,
        .dynamicRendering               = VK_TRUE,
    };

    m_vulkan14Features = {
        .sType                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
        .pNext                    = &m_vulkan13Features,
        .dynamicRenderingLocalRead = m_deviceFeatures.dynamicRenderingLocalRead ? VK_TRUE : VK_FALSE,
        .pushDescriptor           = m_deviceFeatures.pushDescriptor ? VK_TRUE : VK_FALSE,
    };

    m_enabledFeatures = {
        .sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext    = &m_vulkan14Features,
        .features = {
            .sampleRateShading = m_deviceFeatures.sampleRateShading ? VK_TRUE : VK_FALSE,
            .fillModeNonSolid = VK_TRUE,  // Required for VK_POLYGON_MODE_LINE (wireframe)
            .samplerAnisotropy = VK_TRUE,
        },
    };

    // VK_KHR_swapchain is not core; presentation always requires it.
    // Optional renderer features below add their device extensions only when supported.
    VkPhysicalDeviceFragmentShadingRateFeaturesKHR fragmentShadingRateFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR
    };
    if (m_deviceFeatures.fragmentShadingRate) {
        VkPhysicalDeviceFeatures2 supportedFeatures{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            &fragmentShadingRateFeatures
        };
        vkGetPhysicalDeviceFeatures2(m_physicalDevice, &supportedFeatures);
        fragmentShadingRateFeatures.pNext = m_enabledFeatures.pNext;
        m_enabledFeatures.pNext = &fragmentShadingRateFeatures;
    }

    VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT
    };
    if (m_deviceFeatures.meshShader) {
        meshShaderFeatures.taskShader = m_deviceFeatures.taskShader ? VK_TRUE : VK_FALSE;
        meshShaderFeatures.meshShader = VK_TRUE;
        meshShaderFeatures.pNext = m_enabledFeatures.pNext;
        m_enabledFeatures.pNext = &meshShaderFeatures;
    }

    std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
    };

    // VK_KHR_portability_subset is mandatory on MoltenVK; doesn't exist on native drivers.
    if (checkPortabilitySubset(m_physicalDevice))
        appendDeviceExtension(deviceExtensions, "VK_KHR_portability_subset");

    if (m_deviceFeatures.fragmentShadingRate)
        appendDeviceExtension(deviceExtensions, VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME);

    if (m_deviceFeatures.meshShader)
        appendDeviceExtension(deviceExtensions, VK_EXT_MESH_SHADER_EXTENSION_NAME);

    const VkDeviceCreateInfo deviceInfo{
        .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext                   = &m_enabledFeatures,
        .queueCreateInfoCount    = static_cast<uint32_t>(queueInfos.size()),
        .pQueueCreateInfos       = queueInfos.data(),
        .enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data(),
    };

    VK_CHECK(vkCreateDevice(m_physicalDevice, &deviceInfo, nullptr, &m_device));
    volkLoadDevice(m_device);

    vkGetDeviceQueue(m_device, m_graphicsQueueFamily, 0, &m_graphicsQueue);
    vkGetDeviceQueue(m_device, m_computeQueueFamily,  0, &m_computeQueue);
    vkGetDeviceQueue(m_device, m_transferQueueFamily, 0, &m_transferQueue);

    // VMA needs vkGetInstanceProcAddr/vkGetDeviceProcAddr to resolve all other
    // function pointers at runtime, since we use volk (VMA_DYNAMIC_VULKAN_FUNCTIONS=1).
    VmaVulkanFunctions vmaFunctions{};
    vmaFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vmaFunctions.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;

    const VmaAllocatorCreateInfo allocCI{
        // Matches the bufferDeviceAddress feature we enabled in Vk12Features
        .flags            = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
        .physicalDevice   = m_physicalDevice,
        .device           = m_device,
        .pVulkanFunctions = &vmaFunctions,
        .instance         = m_instance,
        .vulkanApiVersion = VK_API_VERSION_1_4,
    };
    VK_CHECK(vmaCreateAllocator(&allocCI, &m_allocator));

    spdlog::info("Logical device + VMA allocator created successfully");
}

// ── Feature queries ───────────────────────────────────────────────────────────

bool VulkanContext::hasFeature_DynamicRenderingLocalRead() const
{
    return m_deviceFeatures.dynamicRenderingLocalRead;
}

bool VulkanContext::hasFeature_PushDescriptor() const
{
    return m_deviceFeatures.pushDescriptor;
}

// ── Diagnostics ───────────────────────────────────────────────────────────────

void VulkanContext::logDeviceInfo() const
{
    const auto& props = m_deviceProps.properties;

    const char* deviceTypeName = [&] {
        switch (props.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "Discrete GPU";
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "Integrated GPU";
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "Virtual GPU";
            case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "CPU";
            default:                                     return "Unknown";
        }
    }();

    const uint32_t apiMaj = VK_API_VERSION_MAJOR(props.apiVersion);
    const uint32_t apiMin = VK_API_VERSION_MINOR(props.apiVersion);
    const uint32_t apiPat = VK_API_VERSION_PATCH(props.apiVersion);
    const uint32_t drv    = props.driverVersion;

    spdlog::info("──── Vulkan Device ──────────────────────────────");
    spdlog::info("  Name:         {}", props.deviceName);
    spdlog::info("  Type:         {}", deviceTypeName);
    spdlog::info("  Vendor ID:    0x{:04X}", props.vendorID);
    spdlog::info("  API version:  {}.{}.{}", apiMaj, apiMin, apiPat);
#ifdef _WIN32
    // NVIDIA encodes driver version as (major << 22) | (minor << 14) | (subminor << 6) | patch
    if (props.vendorID == 0x10DE) {
        spdlog::info("  Driver ver:   {}.{}.{}.{} (NVIDIA encoding)",
            (drv >> 22) & 0x3FF,
            (drv >> 14) & 0xFF,
            (drv >> 6)  & 0xFF,
            drv & 0x3F);
    } else {
        spdlog::info("  Driver ver:   {}.{}.{}", VK_API_VERSION_MAJOR(drv),
                                                 VK_API_VERSION_MINOR(drv),
                                                 VK_API_VERSION_PATCH(drv));
    }
#else
    spdlog::info("  Driver ver:   {}.{}.{}", VK_API_VERSION_MAJOR(drv),
                                             VK_API_VERSION_MINOR(drv),
                                             VK_API_VERSION_PATCH(drv));
#endif
    spdlog::info("  Baseline renderer requirements:");
    spdlog::info("    dynamicRendering          : {}",
        m_deviceFeatures.dynamicRendering ? "YES" : "NO");
    spdlog::info("    synchronization2          : {}",
        m_deviceFeatures.synchronization2 ? "YES" : "NO");
    spdlog::info("    scalarBlockLayout         : {}",
        m_vulkan12Features.scalarBlockLayout == VK_TRUE ? "YES" : "NO");
    spdlog::info("  Scene multisampling support:");
    spdlog::info("    HDR format                : R16G16B16A16_SFLOAT");
    spdlog::info("    Depth format              : D32_SFLOAT");
    spdlog::info("    Supported counts          : {}",
        sampleCountArrayToString(m_deviceFeatures.supportedSceneSampleCounts));
    spdlog::info("    Sample-rate shading       : {}",
        m_deviceFeatures.sampleRateShading ? "supported" : "unsupported");
    spdlog::info("  Optional features enabled on this device:");
    if (m_deviceFeatures.sampleRateShading) {
        spdlog::info("    sampleRateShading");
    }
    if (m_deviceFeatures.dynamicRenderingLocalRead) {
        spdlog::info("    dynamicRenderingLocalRead");
    }
    if (m_deviceFeatures.pushDescriptor) {
        spdlog::info("    pushDescriptor");
    }
    if (m_deviceFeatures.descriptorIndexing) {
        spdlog::info("    descriptorIndexing");
    }
    if (m_deviceFeatures.timelineSemaphore) {
        spdlog::info("    timelineSemaphore");
    }
    if (m_deviceFeatures.fragmentShadingRate) {
        spdlog::info("    fragmentShadingRate");
    }
    if (m_deviceFeatures.meshShader) {
        spdlog::info("    meshShader");
    }
    if (m_deviceFeatures.taskShader) {
        spdlog::info("    taskShader");
    }
    spdlog::info("  Optional features unavailable on this device:");
    if (!m_deviceFeatures.sampleRateShading) {
        spdlog::info("    sampleRateShading");
    }
    if (!m_deviceFeatures.dynamicRenderingLocalRead) {
        spdlog::info("    dynamicRenderingLocalRead");
    }
    if (!m_deviceFeatures.pushDescriptor) {
        spdlog::info("    pushDescriptor");
    }
    if (!m_deviceFeatures.descriptorIndexing) {
        spdlog::info("    descriptorIndexing");
    }
    if (!m_deviceFeatures.timelineSemaphore) {
        spdlog::info("    timelineSemaphore");
    }
    if (!m_deviceFeatures.fragmentShadingRate) {
        spdlog::info("    fragmentShadingRate");
    }
    if (!m_deviceFeatures.meshShader) {
        spdlog::info("    meshShader");
    }
    if (!m_deviceFeatures.taskShader) {
        spdlog::info("    taskShader");
    }
    spdlog::info("  Future-use capability queries should use RendererDeviceFeatures:");
    spdlog::info("    sampleRateShading         : {}",
        m_deviceFeatures.sampleRateShading ? "YES" : "NO");
    spdlog::info("    sceneSampleCounts         : {}",
        sampleCountArrayToString(m_deviceFeatures.supportedSceneSampleCounts));
    spdlog::info("    dynamicRenderingLocalRead : {}",
        hasFeature_DynamicRenderingLocalRead() ? "YES" : "NO");
    spdlog::info("    fragmentShadingRate       : {}",
        m_deviceFeatures.fragmentShadingRate ? "YES" : "NO");
    spdlog::info("    meshShader                : {}",
        m_deviceFeatures.meshShader ? "YES" : "NO");
    spdlog::info("    taskShader                : {}",
        m_deviceFeatures.taskShader ? "YES" : "NO");
    if (m_deviceFeatures.meshShader) {
        spdlog::info("  Mesh shader properties:");
        spdlog::info("    maxTaskWorkGroupInvocations      : {}",
            m_meshShaderProperties.maxTaskWorkGroupInvocations);
        spdlog::info("    maxMeshWorkGroupInvocations      : {}",
            m_meshShaderProperties.maxMeshWorkGroupInvocations);
        spdlog::info("    maxMeshOutputVertices            : {}",
            m_meshShaderProperties.maxMeshOutputVertices);
        spdlog::info("    maxMeshOutputPrimitives          : {}",
            m_meshShaderProperties.maxMeshOutputPrimitives);
        spdlog::info("    maxPreferredTaskWorkGroupInvoc.  : {}",
            m_meshShaderProperties.maxPreferredTaskWorkGroupInvocations);
        spdlog::info("    maxPreferredMeshWorkGroupInvoc.  : {}",
            m_meshShaderProperties.maxPreferredMeshWorkGroupInvocations);
        spdlog::info("    maxMeshMultiviewViewCount        : {}",
            m_meshShaderProperties.maxMeshMultiviewViewCount);
    }
    spdlog::info("  Queue families:");
    spdlog::info("    Graphics  : {}", m_graphicsQueueFamily);
    spdlog::info("    Compute   : {}", m_computeQueueFamily);
    spdlog::info("    Transfer  : {}", m_transferQueueFamily);
    spdlog::info("────────────────────────────────────────────────");
}
