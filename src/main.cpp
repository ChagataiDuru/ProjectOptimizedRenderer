#include "core/Camera.h"
#include "core/Device.h"
#include "core/Renderer.h"
#include "core/Swapchain.h"
#include "core/VulkanContext.h"
#include "core/Window.h"
#include "debug/ImGuiManager.h"

#include <SDL3/SDL.h>
#include <array>
#include <chrono>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <imgui.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <stdexcept>

int main() {
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

    // === ImGui ===
    ImGuiManager imguiManager(vulkanContext, swapchain);
    imguiManager.init(window.getHandle());
    renderer.setImGuiManager(&imguiManager);

    // === Camera Setup ===
    Camera camera;
    float aspectRatio = static_cast<float>(w) / static_cast<float>(h);
    camera.setPerspective(45.0f, aspectRatio, 0.01f, 1000.0f);

    // === Light state (live-editable via the ImGui panel) ===
    struct LightState {
      float direction[3] = {0.577f, 0.577f, 0.577f};
      float color[3]     = {1.0f, 1.0f, 1.0f};
      float intensity    = 3.5f;
      float ambient      = 0.3f;
    } lightState;

    // === Frame Timing ===
    auto lastTime = std::chrono::high_resolution_clock::now();
    float deltaTime = 0.016f;

    // === ImGui panel registration ===
    // FPS history ring buffer (last 90 frames)
    std::array<float, 90> fpsHistory{};
    int fpsHistoryOffset = 0;

    imguiManager.registerPanel({"Performance", [&]() {
      float fps = (deltaTime > 0.0f) ? (1.0f / deltaTime) : 0.0f;
      fpsHistory[static_cast<size_t>(fpsHistoryOffset)] = fps;
      fpsHistoryOffset = (fpsHistoryOffset + 1) % static_cast<int>(fpsHistory.size());
      ImGui::Text("FPS:        %.1f", fps);
      ImGui::Text("Frame time: %.2f ms", deltaTime * 1000.0f);
      ImGui::PlotLines("##fps",
        fpsHistory.data(), static_cast<int>(fpsHistory.size()),
        fpsHistoryOffset,
        "FPS",
        0.0f, 300.0f,
        ImVec2(0.0f, 60.0f));
    }});

    imguiManager.registerPanel({"Camera", [&]() {
      const glm::vec3 pos = camera.getPosition();
      ImGui::Text("Position:  (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);

      // Derive yaw/pitch from quaternion for display
      const glm::vec3 euler = glm::degrees(glm::eulerAngles(camera.getOrientation()));
      ImGui::Text("Pitch/Yaw: (%.1f°, %.1f°)", euler.x, euler.y);
      ImGui::Spacing();
      ImGui::TextDisabled("WASD: move  Mouse: look  F1: toggle cursor");
    }});

    imguiManager.registerPanel({"Light", [&]() {
      bool changed = false;
      changed |= ImGui::SliderFloat3("Direction", lightState.direction, -1.0f, 1.0f);
      changed |= ImGui::ColorEdit3("Color",       lightState.color);
      changed |= ImGui::SliderFloat("Intensity",  &lightState.intensity, 0.0f, 10.0f);
      changed |= ImGui::SliderFloat("Ambient",    &lightState.ambient,   0.0f, 1.0f);
      if (changed) {
        renderer.setLightParameters(
          glm::vec3(lightState.direction[0], lightState.direction[1], lightState.direction[2]),
          glm::vec3(lightState.color[0],     lightState.color[1],     lightState.color[2]),
          lightState.intensity,
          lightState.ambient);
      }
    }});

    imguiManager.registerPanel({"Render Stats", [&]() {
      const auto& stats = renderer.getRenderStats();
      ImGui::Text("Draw calls:  %u",   stats.drawCalls);
      ImGui::Text("Triangles:   %u",   stats.triangles);
      ImGui::Text("Meshes:      %u",   stats.meshCount);
      ImGui::Text("Materials:   %u",   stats.materialCount);
      ImGui::Text("Textures:    %u",   stats.textureCount);
      if (stats.textureMemoryBytes > 0) {
        ImGui::Text("Tex memory:  %.1f MB",
                    stats.textureMemoryBytes / (1024.0f * 1024.0f));
      }
      ImGui::Separator();
      ImGui::TextDisabled("F2: save screenshot");
    }});

    imguiManager.registerPanel({"GPU Timing", [&]() {
      const auto& timer = renderer.getGPUTimer();
      if (!timer.isValid()) {
        ImGui::TextDisabled("GPU timestamps not supported on this device");
        return;
      }
      const float sceneMs = timer.getElapsedMs("ScenePass_Begin", "ScenePass_End");
      const float imguiMs = timer.getElapsedMs("ScenePass_End",   "ImGuiPass_End");
      const float totalMs = sceneMs + imguiMs;
      ImGui::Text("Scene pass:  %.3f ms", sceneMs);
      ImGui::Text("ImGui pass:  %.3f ms", imguiMs);
      ImGui::Text("GPU total:   %.3f ms", totalMs);
      ImGui::Separator();
      ImGui::Text("Budget 60 Hz: 16.67 ms");
      ImGui::ProgressBar(totalMs / 16.67f, ImVec2(-1.0f, 0.0f), "");
    }});

    // === Mouse capture state ===
    bool mouseCaptured = true;
    SDL_SetWindowRelativeMouseMode(
        static_cast<SDL_Window*>(window.getHandle()), true);

    spdlog::info("Entering render loop");

    // === Main Render Loop ===
    bool running = true;
    while (running) {

      // ── Frame timing ──────────────────────────────────────────────────
      auto currentTime = std::chrono::high_resolution_clock::now();
      deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
      lastTime = currentTime;
      deltaTime = glm::min(deltaTime, 0.033f);

      // ── SDL event processing ──────────────────────────────────────────
      const bool* keys = SDL_GetKeyboardState(nullptr);

      if (keys[SDL_SCANCODE_ESCAPE]) {
        running = false;
      }

      SDL_Event event;
      while (SDL_PollEvent(&event)) {
        // Forward all events to ImGui first
        imguiManager.processEvent(&event);

        switch (event.type) {

        case SDL_EVENT_QUIT:
          running = false;
          break;

        // F1: toggle between captured (gameplay) and free (UI) mouse mode
        // F2: capture screenshot to screenshots/
        case SDL_EVENT_KEY_DOWN:
          if (event.key.scancode == SDL_SCANCODE_F1) {
            mouseCaptured = !mouseCaptured;
            SDL_SetWindowRelativeMouseMode(
                static_cast<SDL_Window*>(window.getHandle()), mouseCaptured);
            spdlog::debug("Mouse capture: {}", mouseCaptured ? "on" : "off");
          }
          if (event.key.scancode == SDL_SCANCODE_F2) {
            renderer.requestScreenshot();
            spdlog::info("Screenshot requested");
          }
          break;

        case SDL_EVENT_MOUSE_MOTION: {
          // Only forward mouse movement to the camera when captured
          if (mouseCaptured) {
            float xoffset = static_cast<float>(event.motion.xrel);
            float yoffset = static_cast<float>(event.motion.yrel);
            if (std::abs(xoffset) > 0.1f || std::abs(yoffset) > 0.1f) {
              camera.processMouseMovement(xoffset, yoffset);
            }
          }
          break;
        }

        case SDL_EVENT_WINDOW_RESIZED: {
          float newAspect = static_cast<float>(event.window.data1) /
                            static_cast<float>(event.window.data2);
          camera.setPerspective(45.0f, newAspect, 0.01f, 1000.0f);
          spdlog::info("Window resized to {}x{}", event.window.data1,
                       event.window.data2);
          break;
        }

        default:
          break;
        }
      }

      // ── Camera update (only when mouse is captured / not interacting with UI) ──
      if (mouseCaptured) {
        camera.processKeyboard(keys);
      }
      camera.update(deltaTime);

      // ── Pass camera matrices to renderer ──────────────────────────────
      renderer.setCameraMatrices(camera.getViewMatrix(),
                                 camera.getProjectionMatrix(),
                                 camera.getPosition());

      // ── Build ImGui frame (invoke panel callbacks) ─────────────────────
      imguiManager.beginFrame();

      // ── Render ────────────────────────────────────────────────────────
      renderer.beginFrame();
      renderer.endFrame();  // ImGui overlay is embedded inside endFrame → render()
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
