#include "core/Camera.h"
#include "core/Device.h"
#include "core/Renderer.h"
#include "core/Swapchain.h"
#include "core/VulkanContext.h"
#include "core/Window.h"
#include "debug/ImGuiManager.h"
#include "debug/LogSink.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_dialog.h>
#include <array>
#include <chrono>
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <IconsFontAwesome6.h>
#include <imgui.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

int main() {
  // ── Logger: stdout + ImGui ring buffer ──────────────────────────────────────
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

    // === ImGui ===
    ImGuiManager imguiManager(vulkanContext, swapchain);
    imguiManager.init(window.getHandle());
    renderer.setImGuiManager(&imguiManager);

    // === Rendering toggles (menu bar writes, render loop forwards to renderer) ===
    bool wireframeEnabled = false;
    bool showNormals      = false;
    ImGuiManager::RenderToggles renderToggles;
    renderToggles.wireframe   = &wireframeEnabled;
    renderToggles.showNormals = &showNormals;
    imguiManager.setRenderToggles(renderToggles);

    // === Pending model load (set by file dialog callback, consumed by render loop) ===
    std::string pendingModelPath;

    imguiManager.setOpenModelCallback([&]() {
        // SDL3 native file dialog — async with callback.
        // Filter for glTF files (.gltf and .glb).
        static const SDL_DialogFileFilter filters[] = {
            { "glTF files", "gltf;glb" },
            { "All files",  "*" },
        };

        SDL_ShowOpenFileDialog(
            [](void* userdata, const char* const* filelist, int /*filter*/) {
                if (filelist && filelist[0]) {
                    *static_cast<std::string*>(userdata) = filelist[0];
                }
            },
            &pendingModelPath,
            static_cast<SDL_Window*>(window.getHandle()),
            filters,
            static_cast<int>(std::size(filters)),
            nullptr,   // default directory
            false      // single file, not multiple
        );
    });

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

    imguiManager.registerPanel(ICON_FA_CHART_LINE " Performance", [&]() {
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
    }, DockLocation::Bottom);

    imguiManager.registerPanel(ICON_FA_VIDEO " Camera", [&]() {
      const glm::vec3 pos = camera.getPosition();
      ImGui::Text("Position:  (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);

      const glm::vec3 euler = glm::degrees(glm::eulerAngles(camera.getOrientation()));
      ImGui::Text("Pitch/Yaw: (%.1f°, %.1f°)", euler.x, euler.y);
      ImGui::Spacing();
      ImGui::TextDisabled("WASD: move  Mouse: look  F1: toggle cursor");
    }, DockLocation::Right);

    imguiManager.registerPanel(ICON_FA_SUN " Light", [&]() {
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
    }, DockLocation::Right);

    imguiManager.registerPanel(ICON_FA_CUBES " Render Stats", [&]() {
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
    }, DockLocation::Bottom);

    imguiManager.registerPanel(ICON_FA_CLOCK " GPU Timing", [&]() {
      const auto& timer = renderer.getGPUTimer();
      if (!timer.isValid()) {
        ImGui::TextDisabled("GPU timestamps not available");
        return;
      }

      const float shadowMs = timer.getElapsedMs("ShadowPass_Begin", "ShadowPass_End");
      const float sceneMs  = timer.getElapsedMs("ScenePass_Begin",  "ScenePass_End");
      const float imguiMs  = timer.getElapsedMs("ScenePass_End",    "ImGuiPass_End");
      const float totalMs  = shadowMs + sceneMs + imguiMs;
      const float budget   = 16.67f;  // 60 Hz

      ImGui::Text("Budget: %.2f / %.2f ms (60 Hz)", totalMs, budget);

      // ── Stacked bar chart ──────────────────────────────────────────────────
      const float barWidth  = ImGui::GetContentRegionAvail().x;
      const float barHeight = 20.0f;
      ImVec2      barStart  = ImGui::GetCursorScreenPos();
      ImDrawList* draw      = ImGui::GetWindowDrawList();

      // Dark background track
      draw->AddRectFilled(barStart,
          ImVec2(barStart.x + barWidth, barStart.y + barHeight),
          IM_COL32(40, 40, 45, 255), 4.0f);

      // Shadow pass — orange, leftmost
      const float shadowW = glm::min((shadowMs / budget) * barWidth, barWidth);
      draw->AddRectFilled(barStart,
          ImVec2(barStart.x + shadowW, barStart.y + barHeight),
          IM_COL32(255, 165, 0, 200), 4.0f);

      // Scene pass — blue, stacked after shadow
      const float sceneW = glm::min((sceneMs / budget) * barWidth, barWidth - shadowW);
      draw->AddRectFilled(
          ImVec2(barStart.x + shadowW, barStart.y),
          ImVec2(barStart.x + shadowW + sceneW, barStart.y + barHeight),
          IM_COL32(66, 150, 250, 200), 4.0f);

      // ImGui pass — green, stacked after scene
      const float imguiW = glm::min((imguiMs / budget) * barWidth,
                                    barWidth - shadowW - sceneW);
      draw->AddRectFilled(
          ImVec2(barStart.x + shadowW + sceneW, barStart.y),
          ImVec2(barStart.x + shadowW + sceneW + imguiW, barStart.y + barHeight),
          IM_COL32(80, 200, 120, 200), 4.0f);

      // 100% budget marker — red vertical line at right edge
      draw->AddLine(
          ImVec2(barStart.x + barWidth, barStart.y),
          ImVec2(barStart.x + barWidth, barStart.y + barHeight),
          IM_COL32(255, 80, 80, 200), 2.0f);

      // Advance cursor past the manually-drawn bar
      ImGui::Dummy(ImVec2(barWidth, barHeight + 4.0f));

      // Legend
      ImGui::ColorButton("##sh", ImVec4(1.0f, 0.65f, 0.0f, 1.0f),
                         ImGuiColorEditFlags_NoTooltip, ImVec2(10.0f, 10.0f));
      ImGui::SameLine();
      ImGui::Text("Shadow: %.3f ms", shadowMs);

      ImGui::ColorButton("##sc", ImVec4(0.26f, 0.59f, 0.98f, 1.0f),
                         ImGuiColorEditFlags_NoTooltip, ImVec2(10.0f, 10.0f));
      ImGui::SameLine();
      ImGui::Text("Scene: %.3f ms", sceneMs);

      ImGui::ColorButton("##im", ImVec4(0.31f, 0.78f, 0.47f, 1.0f),
                         ImGuiColorEditFlags_NoTooltip, ImVec2(10.0f, 10.0f));
      ImGui::SameLine();
      ImGui::Text("ImGui: %.3f ms", imguiMs);
    }, DockLocation::Bottom);

    // ── Scene Hierarchy ──────────────────────────────────────────────────────
    int selectedMeshIndex = -1;

    imguiManager.registerPanel(ICON_FA_SITEMAP " Scene", [&]() {
      const auto& model = renderer.getModel();
      const auto& stats = renderer.getRenderStats();

      ImGui::Text("%zu meshes, %u materials",
                  model.meshes.size(), stats.materialCount);
      ImGui::Separator();

      if (ImGui::BeginChild("MeshList", ImVec2(0, 0), ImGuiChildFlags_None)) {
        for (int i = 0; i < static_cast<int>(model.meshes.size()); ++i) {
          const auto& mesh = model.meshes[i];
          bool isSelected = (selectedMeshIndex == i);

          // Color indicator from the mesh's base color factor
          if (mesh.materialIndex >= 0 &&
              mesh.materialIndex < static_cast<int>(model.materials.size())) {
            const auto& mat = model.materials[mesh.materialIndex];
            ImVec4 col(mat.baseColorFactor.r, mat.baseColorFactor.g,
                       mat.baseColorFactor.b, 1.0f);
            ImGui::ColorButton("##c", col,
                               ImGuiColorEditFlags_NoTooltip |
                               ImGuiColorEditFlags_NoDragDrop,
                               ImVec2(12.0f, 12.0f));
            ImGui::SameLine();
          }

          if (ImGui::Selectable(mesh.name.c_str(), isSelected))
            selectedMeshIndex = i;
        }
      }
      ImGui::EndChild();
    }, DockLocation::Right);

    // ── Properties ───────────────────────────────────────────────────────────
    imguiManager.registerPanel(ICON_FA_SLIDERS " Properties", [&]() {
      const auto& model = renderer.getModel();

      if (selectedMeshIndex < 0 ||
          selectedMeshIndex >= static_cast<int>(model.meshes.size())) {
        ImGui::TextDisabled("Select a mesh in the Scene panel");
        return;
      }

      const auto& mesh = model.meshes[selectedMeshIndex];

      ImGui::Text(ICON_FA_CUBE " %s", mesh.name.c_str());
      ImGui::Separator();
      ImGui::Text("Vertices:  %zu", mesh.vertices.size());
      ImGui::Text("Triangles: %zu", mesh.indices.size() / 3);

      if (mesh.materialIndex >= 0 &&
          mesh.materialIndex < static_cast<int>(model.materials.size())) {
        const auto& mat = model.materials[mesh.materialIndex];

        ImGui::Separator();
        ImGui::Text(ICON_FA_PALETTE " Material: %s", mat.name.c_str());

        ImVec4 baseColor(mat.baseColorFactor.r, mat.baseColorFactor.g,
                         mat.baseColorFactor.b, mat.baseColorFactor.a);
        ImGui::ColorEdit4("##basecol", &baseColor.x,
                          ImGuiColorEditFlags_NoInputs |
                          ImGuiColorEditFlags_NoLabel);
        ImGui::SameLine();
        ImGui::Text("Base Color Factor");

        ImGui::Text("Metallic:  %.2f", mat.metallicFactor);
        ImGui::Text("Roughness: %.2f", mat.roughnessFactor);
        ImGui::Text("Alpha:     %s",
                    mat.alphaMode == Material::AlphaMode::Opaque ? "Opaque" :
                    mat.alphaMode == Material::AlphaMode::Mask   ? "Mask" : "Blend");
        ImGui::Text("Double-sided: %s", mat.doubleSided ? "Yes" : "No");

        ImGui::Separator();
        ImGui::Text(ICON_FA_IMAGE " Textures");

        // Display just the filename portion of the texture path
        auto showTex = [&](const char* label, int32_t idx) {
          if (idx >= 0 && idx < static_cast<int>(model.textures.size())) {
            std::string path = model.textures[idx].path;
            auto slash = path.find_last_of("/\\");
            if (slash != std::string::npos) path = path.substr(slash + 1);
            ImGui::Text("  %s: %s", label, path.c_str());
          } else {
            ImGui::TextDisabled("  %s: (none)", label);
          }
        };

        showTex("Albedo",    mat.albedoTextureIndex);
        showTex("Normal",    mat.normalTextureIndex);
        showTex("Metal/Rgh", mat.metallicRoughnessTextureIndex);
      } else {
        ImGui::TextDisabled("No material assigned");
      }
    }, DockLocation::Right);

    // ── Log Console ───────────────────────────────────────────────────────────
    imguiManager.registerPanel(ICON_FA_TERMINAL " Console", [&]() {
      static bool showDebug = false;
      static bool showInfo  = true;
      static bool showWarn  = true;
      static bool showError = true;

      ImGui::Checkbox("Debug", &showDebug); ImGui::SameLine();
      ImGui::Checkbox("Info",  &showInfo);  ImGui::SameLine();
      ImGui::Checkbox("Warn",  &showWarn);  ImGui::SameLine();
      ImGui::Checkbox("Error", &showError); ImGui::SameLine();
      if (ImGui::Button(ICON_FA_TRASH " Clear"))
        imguiSink->clear();

      ImGui::Separator();

      if (ImGui::BeginChild("LogScroll", ImVec2(0, 0), ImGuiChildFlags_None)) {
        for (const auto& entry : imguiSink->getEntries()) {
          bool   show  = false;
          ImVec4 color = ImVec4(0.8f, 0.8f, 0.8f, 1.0f);

          switch (entry.level) {
            case spdlog::level::debug:
              show = showDebug;
              color = ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
              break;
            case spdlog::level::info:
              show = showInfo;
              color = ImVec4(0.85f, 0.85f, 0.85f, 1.0f);
              break;
            case spdlog::level::warn:
              show = showWarn;
              color = ImVec4(1.0f, 0.8f, 0.3f, 1.0f);
              break;
            case spdlog::level::err:
            case spdlog::level::critical:
              show = showError;
              color = ImVec4(1.0f, 0.3f, 0.3f, 1.0f);
              break;
            default:
              show  = true;
              break;
          }

          if (show) {
            ImGui::PushStyleColor(ImGuiCol_Text, color);
            ImGui::TextUnformatted(entry.message.c_str());
            ImGui::PopStyleColor();
          }
        }

        // Auto-scroll to bottom when new entries arrive
        static size_t lastCount = 0;
        size_t currentCount = imguiSink->getEntryCount();
        if (currentCount != lastCount) {
          ImGui::SetScrollHereY(1.0f);
          lastCount = currentCount;
        }
      }
      ImGui::EndChild();
    }, DockLocation::Bottom);

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
          if (event.key.scancode == SDL_SCANCODE_F11) {
            imguiManager.toggleVisible();
            spdlog::info("ImGui overlay: {}",
                         imguiManager.isVisible() ? "visible" : "hidden (F11 fullscreen)");
            // When hiding ImGui, ensure mouse is captured for FPS camera
            if (!imguiManager.isVisible() && !mouseCaptured) {
              mouseCaptured = true;
              SDL_SetWindowRelativeMouseMode(
                  static_cast<SDL_Window*>(window.getHandle()), true);
            }
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
          // Use pixel dimensions for correct aspect ratio on HiDPI displays.
          // event.window.data1/data2 are logical points on macOS — the ratio
          // is the same, but logging pixel dimensions is more accurate.
          int pw, ph;
          SDL_GetWindowSizeInPixels(
              static_cast<SDL_Window*>(window.getHandle()), &pw, &ph);
          if (pw > 0 && ph > 0) {
              float newAspect = static_cast<float>(pw) / static_cast<float>(ph);
              camera.setPerspective(45.0f, newAspect, 0.01f, 1000.0f);
          }
          spdlog::info("Window resized to {}x{} px", pw, ph);
          break;
        }

        default:
          break;
        }
      }

      // ── Quit via menu bar ─────────────────────────────────────────────
      if (imguiManager.shouldQuit()) {
        running = false;
      }

      // ── Deferred model reload (from file dialog) ─────────────────────
      if (!pendingModelPath.empty()) {
          renderer.reloadModel(pendingModelPath);
          pendingModelPath.clear();
      }

      // ── Sync rendering toggles to renderer ────────────────────────────
      renderer.setWireframeEnabled(wireframeEnabled);
      renderer.setNormalVisualization(showNormals);

      // ── Camera update (only when mouse is captured / not interacting with UI) ──
      if (mouseCaptured) {
        camera.processKeyboard(keys);
      }
      camera.update(deltaTime);

      // ── Pass camera matrices to renderer ──────────────────────────────
      renderer.setCameraMatrices(camera.getViewMatrix(),
                                 camera.getProjectionMatrix(),
                                 camera.getPosition());

      // ── Build ImGui frame ─────────────────────────────────────────────
      imguiManager.beginFrame();
      // ── Sun direction indicator (3D projected into viewport) ──────────
      {
          // Project light direction onto screen to draw a sun disc.
          // The sun is "infinitely far away" — we project a point along the light direction
          // from the camera position, far enough to be in front of the near plane.
          const glm::vec3 camPos = camera.getPosition();
          const glm::vec3 lightDir = glm::normalize(glm::vec3(
              lightState.direction[0], lightState.direction[1], lightState.direction[2]));

          // Sun position: far along the light direction from camera
          const glm::vec3 sunWorldPos = camPos + lightDir * 500.0f;

          // Project to clip space
          const glm::vec4 clipPos = camera.getProjectionMatrix() * camera.getViewMatrix()
                                    * glm::vec4(sunWorldPos, 1.0f);

          // Only draw if sun is in front of camera (w > 0)
          if (clipPos.w > 0.0f) {
              const glm::vec3 ndc = glm::vec3(clipPos) / clipPos.w;

              // Convert NDC [-1,1] to screen pixels
              // Note: Vulkan NDC has Y flipped, but ImGui uses top-left origin
              uint32_t fbW, fbH;
              window.getExtent(fbW, fbH);
              const float screenX = (ndc.x * 0.5f + 0.5f) * static_cast<float>(fbW);
              const float screenY = (0.5f - ndc.y * 0.5f) * static_cast<float>(fbH);

              // Draw sun disc on the foreground overlay (renders above the viewport)
              ImDrawList* fg = ImGui::GetForegroundDrawList();

              // Outer glow
              fg->AddCircleFilled(
                  ImVec2(screenX, screenY), 24.0f,
                  IM_COL32(255, 220, 100, 60), 32);
              fg->AddCircleFilled(
                  ImVec2(screenX, screenY), 16.0f,
                  IM_COL32(255, 220, 100, 120), 32);

              // Inner bright core
              fg->AddCircleFilled(
                  ImVec2(screenX, screenY), 8.0f,
                  IM_COL32(255, 255, 200, 255), 32);

              // Direction label
              fg->AddText(
                  ImVec2(screenX + 14.0f, screenY - 8.0f),
                  IM_COL32(255, 255, 200, 180),
                  ICON_FA_SUN);
          }
      }
      imguiManager.endFrame();   // Dockspace + panel drawing + ImGui::Render()

      // ── Render ────────────────────────────────────────────────────────
      if (!renderer.beginFrame())
          continue;  // Frame skipped (e.g. window minimized) — do not call endFrame()
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
