#pragma once

#include <SDL3/SDL.h>
#include <cstdint>

class Window {
public:
    Window(uint32_t width, uint32_t height, const char* title);
    ~Window();

    Window(const Window&)            = delete;
    Window& operator=(const Window&) = delete;

    void  init();
    void  shutdown();

    // Returns SDL_Window* as void* to avoid leaking SDL3 into callers that only need the handle.
    void* getHandle() const { return m_window; }

    // Returns drawable pixel dimensions (not logical points — critical for HiDPI on Apple Silicon).
    void getExtent(uint32_t& width, uint32_t& height) const;

    bool shouldClose() const { return m_shouldClose; }
    void pollEvents();

private:
    SDL_Window* m_window      = nullptr;
    uint32_t    m_width;
    uint32_t    m_height;
    const char* m_title;
    bool        m_shouldClose = false;
};
