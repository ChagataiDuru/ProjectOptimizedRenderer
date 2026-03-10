#include "core/Window.h"

#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

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
    if (!SDL_Init(SDL_INIT_VIDEO))
        throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());

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
