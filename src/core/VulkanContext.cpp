#include "core/VulkanContext.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

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

    const std::vector<const char*> extensions = {
        VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME,
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };

    // Build layer list: enumerate what's actually available to avoid VK_ERROR_LAYER_NOT_PRESENT.
    // Validation layers are only present when the LunarG Vulkan SDK is installed separately
    // from MoltenVK — they are not bundled with the MoltenVK Homebrew formula.
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

    const VkInstanceCreateInfo createInfo{
        .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        // Chain debug messenger so validation covers instance creation/destruction too
        .pNext                   = &debugInfo,
        // Required for MoltenVK portability enumeration
        .flags                   = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
        .pApplicationInfo        = &appInfo,
        .enabledLayerCount       = static_cast<uint32_t>(validationLayers.size()),
        .ppEnabledLayerNames     = validationLayers.data(),
        .enabledExtensionCount   = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &m_instance));
    volkLoadInstance(m_instance);

    VK_CHECK(vkCreateDebugUtilsMessengerEXT(m_instance, &debugInfo, nullptr, &m_debugMessenger));
}

// ── Physical device selection ─────────────────────────────────────────────────

std::vector<VkExtensionProperties> VulkanContext::getDeviceExtensions(VkPhysicalDevice device)
{
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, exts.data());
    return exts;
}

bool VulkanContext::checkRequiredExtensions(VkPhysicalDevice device)
{
    // VK_KHR_portability_subset is mandatory on MoltenVK
    auto exts = getDeviceExtensions(device);
    return std::any_of(exts.begin(), exts.end(), [](const VkExtensionProperties& e) {
        return strcmp(e.extensionName, "VK_KHR_portability_subset") == 0;
    });
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

    // Mandatory Vulkan 1.4 features
    VkPhysicalDeviceVulkan14Features vk14{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES};
    VkPhysicalDeviceVulkan13Features vk13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &vk14};
    VkPhysicalDeviceFeatures2 feats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &vk13};
    vkGetPhysicalDeviceFeatures2(device, &feats);

    return vk13.dynamicRendering == VK_TRUE &&
           vk14.dynamicRenderingLocalRead == VK_TRUE;
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
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            score += 1000;
        else if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
            // Apple Silicon UMA: integrated is the right classification
            score += 500;

        VkPhysicalDeviceVulkan14Features vk14{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES};
        VkPhysicalDeviceFeatures2 feats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &vk14};
        vkGetPhysicalDeviceFeatures2(dev, &feats);

        if (vk14.pushDescriptor) score += 100;

        auto exts = getDeviceExtensions(dev);
        bool hasFragShading = std::any_of(exts.begin(), exts.end(),
            [](const VkExtensionProperties& e) {
                return strcmp(e.extensionName, VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME) == 0;
            });
        if (hasFragShading) score += 50;

        spdlog::info("  GPU candidate: {} | type {} | score {}",
            props.deviceName, static_cast<int>(props.deviceType), score);

        if (score > bestScore) {
            bestScore = score;
            best      = dev;
        }
    }

    if (best == VK_NULL_HANDLE)
        throw std::runtime_error(
            "No suitable GPU found: Vulkan 1.4 + dynamicRendering + "
            "dynamicRenderingLocalRead are required");

    m_physicalDevice = best;

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

        // Prefer a queue family that is compute-only (no graphics)
        if (m_computeQueueFamily == UINT32_MAX &&
            (flags & VK_QUEUE_COMPUTE_BIT) && !(flags & VK_QUEUE_GRAPHICS_BIT))
            m_computeQueueFamily = i;

        // Prefer a queue family that is transfer-only
        if (m_transferQueueFamily == UINT32_MAX &&
            (flags & VK_QUEUE_TRANSFER_BIT) &&
            !(flags & VK_QUEUE_GRAPHICS_BIT) &&
            !(flags & VK_QUEUE_COMPUTE_BIT))
            m_transferQueueFamily = i;
    }

    // MoltenVK on Apple Silicon exposes one unified family — fall back gracefully
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

    // Query which optional features are actually supported before enabling them
    VkPhysicalDeviceVulkan14Features queryVk14{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES};
    VkPhysicalDeviceFeatures2 queryFeats{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &queryVk14};
    vkGetPhysicalDeviceFeatures2(m_physicalDevice, &queryFeats);

    // Build enabled feature chain: Features2 -> Vk14 -> Vk13 -> Vk12
    m_vulkan12Features = {
        .sType              = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .scalarBlockLayout  = VK_TRUE,   // core since 1.2; needed for std430 in GLSL
        .bufferDeviceAddress = VK_TRUE,  // required for VMA buffer device address mode
    };

    m_vulkan13Features = {
        .sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext            = &m_vulkan12Features,
        .synchronization2 = VK_TRUE,
        .dynamicRendering = VK_TRUE,    // core since 1.3; no VkRenderPass
    };

    m_vulkan14Features = {
        .sType                    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES,
        .pNext                    = &m_vulkan13Features,
        .dynamicRenderingLocalRead = VK_TRUE,                         // SMAA tile-local
        .pushDescriptor           = queryVk14.pushDescriptor,         // core in 1.4; conditional
    };

    m_enabledFeatures = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &m_vulkan14Features,
    };

    // Device extensions: only portability_subset is needed on MoltenVK
    // (dynamic_rendering, push_descriptors, sync2 are all core in Vulkan 1.4)
    std::vector<const char*> deviceExtensions;
    if (checkRequiredExtensions(m_physicalDevice))
        deviceExtensions.push_back("VK_KHR_portability_subset");

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

    spdlog::info("Logical device created successfully");
}

// ── Feature queries ───────────────────────────────────────────────────────────

bool VulkanContext::hasFeature_DynamicRenderingLocalRead() const
{
    return m_vulkan14Features.dynamicRenderingLocalRead == VK_TRUE;
}

bool VulkanContext::hasFeature_PushDescriptor() const
{
    return m_vulkan14Features.pushDescriptor == VK_TRUE;
}

// ── Diagnostics ───────────────────────────────────────────────────────────────

void VulkanContext::logDeviceInfo() const
{
    const auto& props = m_deviceProps.properties;

    const char* deviceTypeName = [&] {
        switch (props.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "Discrete GPU";
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "Integrated GPU (Apple Silicon UMA)";
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
    spdlog::info("  Driver ver:   {}.{}.{}", VK_API_VERSION_MAJOR(drv),
                                             VK_API_VERSION_MINOR(drv),
                                             VK_API_VERSION_PATCH(drv));
    spdlog::info("  Features:");
    spdlog::info("    dynamicRenderingLocalRead : {}",
        hasFeature_DynamicRenderingLocalRead() ? "YES" : "NO");
    spdlog::info("    pushDescriptor            : {}",
        hasFeature_PushDescriptor() ? "YES" : "NO");
    spdlog::info("    dynamicRendering          : {}",
        m_vulkan13Features.dynamicRendering == VK_TRUE ? "YES" : "NO");
    spdlog::info("    synchronization2          : {}",
        m_vulkan13Features.synchronization2 == VK_TRUE ? "YES" : "NO");
    spdlog::info("    scalarBlockLayout         : {}",
        m_vulkan12Features.scalarBlockLayout == VK_TRUE ? "YES" : "NO");
    spdlog::info("  Queue families:");
    spdlog::info("    Graphics  : {}", m_graphicsQueueFamily);
    spdlog::info("    Compute   : {}", m_computeQueueFamily);
    spdlog::info("    Transfer  : {}", m_transferQueueFamily);
    spdlog::info("────────────────────────────────────────────────");
}
