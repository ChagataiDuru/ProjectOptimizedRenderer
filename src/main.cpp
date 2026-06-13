#include "core/Camera.h"
#include "core/Device.h"
#include "core/RenderFramePacket.h"
#include "core/Renderer.h"
#include "core/Swapchain.h"
#include "core/VulkanContext.h"
#include "core/Window.h"
#include "debug/ImGuiManager.h"
#include "debug/LogSink.h"
#include "viewer/ViewerPanels.h"
#include "viewer/ViewerState.h"

#include <SDL3/SDL.h>
#include <IconsFontAwesome6.h>
#include <chrono>
#include <cmath>
#include <glm/glm.hpp>
#include <imgui.h>
#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace {

RenderFramePacket buildRenderFramePacket(const Camera& camera, const ViewerState& state)
{
    RenderFramePacket packet{};
    packet.camera.view       = camera.getViewMatrix();
    packet.camera.projection = camera.getProjectionMatrix();
    packet.camera.position   = camera.getPosition();
    packet.camera.nearZ      = camera.getNearZ();
    packet.camera.farZ       = camera.getFarZ();
    packet.light             = state.light;
    packet.shadow            = state.shadow;
    packet.tonemap           = state.tonemap;
    packet.sky               = state.sky;
    packet.debug             = state.debugView;
    return packet;
}

void drawSunDirectionOverlay(const Camera& camera, Window& window, const ViewerState& state)
{
    if (glm::length(state.light.direction) <= 0.0001f) {
        return;
    }

    const glm::vec3 camPos = camera.getPosition();
    const glm::vec3 lightDir = glm::normalize(state.light.direction);
    const glm::vec3 sunWorldPos = camPos + lightDir * 500.0f;

    const glm::vec4 clipPos = camera.getProjectionMatrix() * camera.getViewMatrix()
                            * glm::vec4(sunWorldPos, 1.0f);
    if (clipPos.w <= 0.0f) {
        return;
    }

    const glm::vec3 ndc = glm::vec3(clipPos) / clipPos.w;

    uint32_t fbW, fbH;
    window.getExtent(fbW, fbH);
    const float screenX = (ndc.x * 0.5f + 0.5f) * static_cast<float>(fbW);
    const float screenY = (0.5f - ndc.y * 0.5f) * static_cast<float>(fbH);

    ImDrawList* fg = ImGui::GetForegroundDrawList();
    fg->AddCircleFilled(ImVec2(screenX, screenY), 24.0f,
                        IM_COL32(255, 220, 100, 60), 32);
    fg->AddCircleFilled(ImVec2(screenX, screenY), 16.0f,
                        IM_COL32(255, 220, 100, 120), 32);
    fg->AddCircleFilled(ImVec2(screenX, screenY), 8.0f,
                        IM_COL32(255, 255, 200, 255), 32);
    fg->AddText(ImVec2(screenX + 14.0f, screenY - 8.0f),
                IM_COL32(255, 255, 200, 180),
                ICON_FA_SUN);
}

}  // namespace

int main()
{
    auto stdoutSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto imguiSink  = std::make_shared<ImGuiLogSink>(500);

    auto logger = std::make_shared<spdlog::logger>(
        "POR", spdlog::sinks_init_list{ stdoutSink, imguiSink });
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

        ImGuiManager imguiManager(vulkanContext, swapchain);
        imguiManager.init(window.getHandle());
        renderer.setImGuiManager(&imguiManager);

        ViewerState viewerState;
        ImGuiManager::RenderToggles renderToggles;
        renderToggles.wireframe   = &viewerState.debugView.wireframe;
        renderToggles.showNormals = &viewerState.debugView.showNormals;
        imguiManager.setRenderToggles(renderToggles);

        Camera camera;
        const float aspectRatio = static_cast<float>(w) / static_cast<float>(h);
        camera.setPerspective(45.0f, aspectRatio, 0.01f, 1000.0f);
        camera.fitToScene(renderer.getSceneInfo().normalizedRadius);

        registerViewerPanels(imguiManager,
                             renderer,
                             camera,
                             window,
                             *imguiSink,
                             viewerState);

        auto lastTime = std::chrono::high_resolution_clock::now();

        SDL_SetWindowRelativeMouseMode(
            static_cast<SDL_Window*>(window.getHandle()), viewerState.mouseCaptured);

        spdlog::info("Entering render loop");

        bool running = true;
        while (running) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            viewerState.deltaTime =
                std::chrono::duration<float>(currentTime - lastTime).count();
            lastTime = currentTime;
            viewerState.deltaTime = glm::min(viewerState.deltaTime, 0.033f);

            const bool* keys = SDL_GetKeyboardState(nullptr);
            if (keys[SDL_SCANCODE_ESCAPE]) {
                running = false;
            }

            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                imguiManager.processEvent(&event);

                switch (event.type) {
                    case SDL_EVENT_QUIT:
                        running = false;
                        break;

                    case SDL_EVENT_KEY_DOWN:
                        if (event.key.scancode == SDL_SCANCODE_F1) {
                            viewerState.mouseCaptured = !viewerState.mouseCaptured;
                            SDL_SetWindowRelativeMouseMode(
                                static_cast<SDL_Window*>(window.getHandle()),
                                viewerState.mouseCaptured);
                            spdlog::debug("Mouse capture: {}",
                                          viewerState.mouseCaptured ? "on" : "off");
                        }
                        if (event.key.scancode == SDL_SCANCODE_F2) {
                            renderer.requestScreenshot();
                            spdlog::info("Screenshot requested");
                        }
                        if (event.key.scancode == SDL_SCANCODE_F11) {
                            imguiManager.toggleVisible();
                            spdlog::info("ImGui overlay: {}",
                                         imguiManager.isVisible()
                                             ? "visible"
                                             : "hidden (F11 fullscreen)");
                            if (!imguiManager.isVisible() && !viewerState.mouseCaptured) {
                                viewerState.mouseCaptured = true;
                                SDL_SetWindowRelativeMouseMode(
                                    static_cast<SDL_Window*>(window.getHandle()), true);
                            }
                        }
                        break;

                    case SDL_EVENT_MOUSE_MOTION:
                        if (viewerState.mouseCaptured) {
                            const float xoffset = static_cast<float>(event.motion.xrel);
                            const float yoffset = static_cast<float>(event.motion.yrel);
                            if (std::abs(xoffset) > 0.1f || std::abs(yoffset) > 0.1f) {
                                camera.processMouseMovement(xoffset, yoffset);
                            }
                        }
                        break;

                    case SDL_EVENT_WINDOW_RESIZED: {
                        int pw, ph;
                        SDL_GetWindowSizeInPixels(
                            static_cast<SDL_Window*>(window.getHandle()), &pw, &ph);
                        if (pw > 0 && ph > 0) {
                            const float newAspect =
                                static_cast<float>(pw) / static_cast<float>(ph);
                            camera.setPerspective(45.0f, newAspect, 0.01f, 1000.0f);
                        }
                        spdlog::info("Window resized to {}x{} px", pw, ph);
                        break;
                    }

                    default:
                        break;
                }
            }

            if (imguiManager.shouldQuit()) {
                running = false;
            }

            if (!viewerState.pendingModelPath.empty()) {
                renderer.reloadModel(viewerState.pendingModelPath);
                camera.fitToScene(renderer.getSceneInfo().normalizedRadius);
                viewerState.selectedMeshIndex = -1;
                viewerState.pendingModelPath.clear();
            }

            if (!viewerState.pendingPanoramaPath.empty()) {
                renderer.loadHdrPanorama(viewerState.pendingPanoramaPath);
                viewerState.pendingPanoramaPath.clear();
            }

            if (viewerState.mouseCaptured) {
                camera.processKeyboard(keys);
            }
            camera.update(viewerState.deltaTime);

            imguiManager.beginFrame();
            drawSunDirectionOverlay(camera, window, viewerState);
            imguiManager.endFrame();

            if (!renderer.beginFrame()) {
                continue;
            }
            renderer.submitFrame(buildRenderFramePacket(camera, viewerState));
            renderer.endFrame();
        }

        spdlog::info("Render loop ended, shutting down...");

        imguiManager.shutdown();
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
