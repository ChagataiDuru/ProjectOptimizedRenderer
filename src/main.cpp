#include "core/VulkanContext.h"
#include "core/Device.h"
#include "core/Window.h"
#include "core/Swapchain.h"
#include "core/Renderer.h"
#include "core/Camera.h"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <SDL3/SDL.h>
#include <glm/glm.hpp>
#include <chrono>
#include <cmath>
#include <stdexcept>

int main()
{
    auto logger = spdlog::stdout_color_mt("POR");
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");

    spdlog::info("ProjectOptimizedRenderer starting");

    try {
        Window window(1280, 720, "ProjectOptimizedRenderer");
        window.init();

        VulkanContext vulkanContext;
        vulkanContext.init();

        uint32_t w, h;
        window.getExtent(w, h);

        Swapchain swapchain(vulkanContext);
        swapchain.init(window.getHandle(), w, h);

        Device device(vulkanContext);

        Renderer renderer(vulkanContext, swapchain);
        renderer.init();

        // === Camera Setup ===
        Camera camera;
        float aspectRatio = static_cast<float>(w) / static_cast<float>(h);
        camera.setPerspective(45.0f, aspectRatio, 0.01f, 1000.0f);

        // === Frame Timing ===
        auto lastTime = std::chrono::high_resolution_clock::now();
        float deltaTime = 0.016f;  // 16ms default until first real measurement

        spdlog::info("Entering render loop");

        // === Main Render Loop ===
        bool running = true;
        while (running) {

            // ── Frame timing ──────────────────────────────────────────────────
            auto currentTime = std::chrono::high_resolution_clock::now();
            deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
            lastTime = currentTime;

            // Clamp to prevent large jumps during window drag or minimize.
            deltaTime = glm::min(deltaTime, 0.033f);

            spdlog::debug("Frame: {:.2f}ms ({:.1f} FPS)",
                          deltaTime * 1000.0f, 1.0f / deltaTime);

            // ── SDL event processing ──────────────────────────────────────────
            // Keyboard state for smooth per-frame movement (held keys).
            const bool* keys = SDL_GetKeyboardState(nullptr);

            if (keys[SDL_SCANCODE_ESCAPE]) {
                running = false;
            }

            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                switch (event.type) {

                    case SDL_EVENT_QUIT:
                        running = false;
                        break;

                    case SDL_EVENT_MOUSE_MOTION: {
                        float xoffset = static_cast<float>(event.motion.xrel);
                        float yoffset = static_cast<float>(event.motion.yrel);
                        if (std::abs(xoffset) > 0.1f || std::abs(yoffset) > 0.1f) {
                            camera.processMouseMovement(xoffset, yoffset);
                        }
                        break;
                    }

                    case SDL_EVENT_WINDOW_RESIZED: {
                        float newAspect = static_cast<float>(event.window.data1)
                                        / static_cast<float>(event.window.data2);
                        camera.setPerspective(45.0f, newAspect, 0.01f, 1000.0f);
                        spdlog::info("Window resized to {}x{}",
                                     event.window.data1, event.window.data2);
                        break;
                    }

                    default:
                        break;
                }
            }

            // ── Camera update ─────────────────────────────────────────────────
            camera.processKeyboard(keys);
            camera.update(deltaTime);

            // ── Pass camera matrices to renderer ──────────────────────────────
            renderer.setCameraMatrices(camera.getViewMatrix(),
                                       camera.getProjectionMatrix());

            // ── Render ────────────────────────────────────────────────────────
            renderer.beginFrame();
            renderer.endFrame();
        }

        spdlog::info("Render loop ended, shutting down...");

        renderer.shutdown();
        swapchain.shutdown();
        vulkanContext.shutdown();
        window.shutdown();

    } catch (const std::exception& e) {
        spdlog::critical("Fatal: {}", e.what());
        return 1;
    }

    return 0;
}
