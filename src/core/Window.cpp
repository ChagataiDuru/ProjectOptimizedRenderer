#include "core/Window.h"

#include <SDL3/SDL_vulkan.h>
#include <spdlog/spdlog.h>
#include <cstdlib>
#include <stdexcept>
#include <string_view>
#include <string>

#if defined(__APPLE__)
#include <array>
#include <filesystem>
#endif

namespace {

#if defined(__APPLE__)
const char* selectMacVulkanLoaderPath()
{
#ifdef POR_VULKAN_LOADER_PATH
    constexpr std::string_view configuredPath = POR_VULKAN_LOADER_PATH;
    if (!configuredPath.empty() && std::filesystem::exists(configuredPath)) {
        return POR_VULKAN_LOADER_PATH;
    }
#endif

    static constexpr std::array<const char*, 4> kFallbackPaths{{
        "/opt/homebrew/lib/libvulkan.1.dylib",
        "/opt/homebrew/lib/libvulkan.dylib",
        "/usr/local/lib/libvulkan.1.dylib",
        "/usr/local/lib/libvulkan.dylib",
    }};

    for (const char* candidate : kFallbackPaths) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    return nullptr;
}

const char* selectMacMoltenVkIcdPath()
{
#ifdef POR_MOLTENVK_ICD_JSON_PATH
    constexpr std::string_view configuredPath = POR_MOLTENVK_ICD_JSON_PATH;
    if (!configuredPath.empty() && std::filesystem::exists(configuredPath)) {
        return POR_MOLTENVK_ICD_JSON_PATH;
    }
#endif

    static constexpr std::array<const char*, 3> kFallbackPaths{{
        "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json",
        "/opt/homebrew/Cellar/molten-vk/1.4.1/etc/vulkan/icd.d/MoltenVK_icd.json",
        "/usr/local/share/vulkan/icd.d/MoltenVK_icd.json",
    }};

    for (const char* candidate : kFallbackPaths) {
        if (std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    return nullptr;
}

void configureMacVulkanEnvironment()
{
    if (const char* icdPath = selectMacMoltenVkIcdPath()) {
        if (setenv("VK_DRIVER_FILES", icdPath, 1) != 0 ||
            setenv("VK_ICD_FILENAMES", icdPath, 1) != 0) {
            throw std::runtime_error(
                std::string("Failed to pin MoltenVK ICD manifest: ") + icdPath);
        }
        spdlog::info("Vulkan ICD pinned to {}", icdPath);
    }
}
#endif

} // namespace

Window::Window(uint32_t width, uint32_t height, const char* title)
    : m_width(width), m_height(height), m_title(title)
{
}

Window::~Window()
{
    shutdown();
}

void Window::init()
{
#if defined(__APPLE__)
    configureMacVulkanEnvironment();
#endif

    if (!SDL_Init(SDL_INIT_VIDEO))
        throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());

#if defined(__APPLE__)
    const char* vulkanLibraryPath = selectMacVulkanLoaderPath();
    if (!vulkanLibraryPath) {
#ifdef POR_MOLTENVK_LIBRARY_PATH
        vulkanLibraryPath = POR_MOLTENVK_LIBRARY_PATH;
#endif
    }

    if (vulkanLibraryPath) {
        if (!SDL_Vulkan_LoadLibrary(vulkanLibraryPath)) {
            throw std::runtime_error(
                std::string("SDL_Vulkan_LoadLibrary failed for '") +
                vulkanLibraryPath + "': " + SDL_GetError());
        }
        m_vulkanLibraryLoaded = true;
        spdlog::info("SDL Vulkan loader pinned to {}", vulkanLibraryPath);
    }
#endif

    m_window = SDL_CreateWindow(
        m_title,
        static_cast<int>(m_width),
        static_cast<int>(m_height),
        SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY
    );

    if (!m_window)
        throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());

    // Sync stored dimensions to actual pixel size immediately after creation.
    // On macOS HiDPI and Windows with DPI scaling, pixel size may differ from logical size.
    int pw, ph;
    SDL_GetWindowSizeInPixels(m_window, &pw, &ph);
    m_width  = static_cast<uint32_t>(pw);
    m_height = static_cast<uint32_t>(ph);

    spdlog::info("Window created: {}x{} px", m_width, m_height);
}

void Window::shutdown()
{
    if (m_window) {
        SDL_DestroyWindow(m_window);
        m_window = nullptr;
    }
    if (m_vulkanLibraryLoaded) {
        SDL_Vulkan_UnloadLibrary();
        m_vulkanLibraryLoaded = false;
    }
    SDL_Quit();
}

void Window::getExtent(uint32_t& width, uint32_t& height) const
{
    // SDL_GetWindowSizeInPixels returns the correct drawable dimensions on all platforms:
    //   macOS — Metal drawable size (differs from logical points on HiDPI)
    //   Windows — client area pixels (accounts for DPI scaling)
    int pw, ph;
    SDL_GetWindowSizeInPixels(m_window, &pw, &ph);
    width  = static_cast<uint32_t>(pw);
    height = static_cast<uint32_t>(ph);
}

void Window::pollEvents()
{
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
            case SDL_EVENT_QUIT:
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
                m_shouldClose = true;
                break;
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                // Use pixel size changed (not RESIZED) to track DPI-correct dimensions
                m_width  = static_cast<uint32_t>(event.window.data1);
                m_height = static_cast<uint32_t>(event.window.data2);
                break;
            default:
                break;
        }
    }
}
