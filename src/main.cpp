#include "core/Camera.h"
#include "core/RenderFramePacket.h"
#include "core/RendererInstance.h"
#include "debug/ImGuiManager.h"
#include "debug/LogSink.h"
#include "resource/GLTFLoader.h"
#include "resource/SceneInfo.h"
#include "viewer/CapturePresets.h"
#include "viewer/ViewerPanels.h"
#include "viewer/ViewerState.h"

#include <SDL3/SDL.h>
#include <IconsFontAwesome6.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <glm/glm.hpp>
#include <imgui.h>
#include <memory>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef ASSET_DIR
#define ASSET_DIR "assets"
#endif

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

std::vector<shader_interface::GpuPointLight> buildDefaultScenePointLights(float normalizedRadius)
{
    const float radiusScale = std::max(normalizedRadius, 1.0f);
    const float y = 1.8f;
    const float radius = radiusScale * 0.45f;
    const float intensity = 22.0f;

    return {
        { glm::vec4(-0.45f * radiusScale, y, -0.35f * radiusScale, radius), glm::vec4(1.0f, 0.45f, 0.30f, intensity) },
        { glm::vec4( 0.45f * radiusScale, y, -0.35f * radiusScale, radius), glm::vec4(0.35f, 0.65f, 1.0f, intensity) },
        { glm::vec4(-0.45f * radiusScale, y,  0.35f * radiusScale, radius), glm::vec4(0.45f, 1.0f, 0.55f, intensity) },
        { glm::vec4( 0.45f * radiusScale, y,  0.35f * radiusScale, radius), glm::vec4(1.0f, 0.80f, 0.35f, intensity) },
    };
}

RenderScenePacket buildRenderScenePacket(const Model& model,
                                         const SceneInfo& sceneInfo,
                                         bool includePointLights = true)
{
    RenderScenePacket scene{};

    scene.draws.reserve(model.meshes.size());
    for (size_t i = 0; i < model.meshes.size(); ++i) {
        MaterialHandle material{};
        const Mesh& importedMesh = model.meshes[i];
        if (importedMesh.materialIndex >= 0 &&
            importedMesh.materialIndex < static_cast<int32_t>(model.materials.size())) {
            material.index = static_cast<uint32_t>(importedMesh.materialIndex);
        }

        scene.draws.push_back(DrawCommand{
            .transform = sceneInfo.modelMatrix,
            .mesh = MeshHandle{ static_cast<uint32_t>(i) },
            .material = material,
        });
    }

    if (includePointLights) {
        scene.pointLights = buildDefaultScenePointLights(sceneInfo.normalizedRadius);
    }
    return scene;
}

void submitModel(RendererInstance& renderer,
                 const Model& model,
                 const SceneInfo& sceneInfo,
                 bool includePointLights = true)
{
    if (renderer.uploadModelResources(model) != RendererResult::Success) {
        throw std::runtime_error("Renderer resource upload failed: " + renderer.getLastError());
    }
    if (renderer.submitScene(buildRenderScenePacket(model, sceneInfo, includePointLights))
        != RendererResult::Success) {
        throw std::runtime_error("Renderer scene submission failed: " + renderer.getLastError());
    }
}

bool loadViewerModel(RendererInstance& renderer,
                     const std::string& requestedPath,
                     const std::string& fallbackPath,
                     Model& outModel,
                     SceneInfo& outSceneInfo)
{
    const auto tryLoad = [&](const std::string& path) {
        Model model = GLTFLoader::loadGLTF(path);
        SceneInfo sceneInfo = computeSceneInfo(model.boundsMin, model.boundsMax);
        submitModel(renderer, model, sceneInfo);
        outModel = std::move(model);
        outSceneInfo = sceneInfo;
        spdlog::info("Viewer model active: '{}' (scale {:.4f}, radius {:.2f})",
                     path, outSceneInfo.scaleFactor, outSceneInfo.normalizedRadius);
    };

    try {
        tryLoad(requestedPath);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Failed to load model '{}': {}", requestedPath, e.what());
        if (requestedPath == fallbackPath) {
            return false;
        }
    }

    try {
        spdlog::info("Falling back to default model '{}'", fallbackPath);
        tryLoad(fallbackPath);
        return true;
    } catch (const std::exception& e) {
        spdlog::critical("Failed to load fallback model '{}': {}", fallbackPath, e.what());
        return false;
    }
}

// ── Scripted capture mode ─────────────────────────────────────────────────────
// Additive entry point used by the artifact workstreams: apply one capture preset,
// render a fixed number of frames, save one PNG, and exit. The interactive viewer
// path is unchanged when no capture argument is supplied.

struct CaptureOptions {
    std::string outputDir;
    std::vector<std::string> presetIds;
    std::string scenePath;
    uint32_t frameCount = 30;
    uint32_t width = 1280;
    uint32_t height = 720;
};

struct ParsedArgs {
    CaptureOptions capture;
    bool captureRequested = false;
    bool listPresets = false;
    bool helpRequested = false;
    std::string error;
};

bool parseUnsigned(const std::string& text, uint32_t& out)
{
    try {
        const unsigned long value = std::stoul(text);
        if (value == 0 || value > 0xFFFFFFFFul) {
            return false;
        }
        out = static_cast<uint32_t>(value);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

ParsedArgs parseArgs(int argc, char** argv)
{
    ParsedArgs parsed;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const auto takeValue = [&](std::string& target) {
            if (i + 1 >= argc) {
                parsed.error = "Missing value for " + arg;
                return false;
            }
            target = argv[++i];
            return true;
        };

        if (arg == "-h" || arg == "--help") {
            parsed.helpRequested = true;
        } else if (arg == "--list-presets") {
            parsed.listPresets = true;
        } else if (arg == "--scene") {
            if (!takeValue(parsed.capture.scenePath)) return parsed;
        } else if (arg == "--capture") {
            parsed.captureRequested = true;
            if (!takeValue(parsed.capture.outputDir)) return parsed;
        } else if (arg == "--preset") {
            std::string value;
            if (!takeValue(value)) return parsed;
            parsed.capture.presetIds.push_back(std::move(value));
        } else if (arg == "--frames") {
            std::string value;
            if (!takeValue(value)) return parsed;
            if (!parseUnsigned(value, parsed.capture.frameCount)) {
                parsed.error = "Invalid --frames value: " + value;
                return parsed;
            }
        } else if (arg == "--width") {
            std::string value;
            if (!takeValue(value)) return parsed;
            if (!parseUnsigned(value, parsed.capture.width)) {
                parsed.error = "Invalid --width value: " + value;
                return parsed;
            }
        } else if (arg == "--height") {
            std::string value;
            if (!takeValue(value)) return parsed;
            if (!parseUnsigned(value, parsed.capture.height)) {
                parsed.error = "Invalid --height value: " + value;
                return parsed;
            }
        } else {
            parsed.error = "Unknown argument: " + arg;
            return parsed;
        }
    }

    return parsed;
}

void printUsage()
{
    spdlog::info(
        "Usage: ProjectOptimizedRenderer [options]\n"
        "  (no options)              run the interactive viewer\n"
        "  --capture <dir>           render capture presets to <dir>/<preset>.png, then exit\n"
        "  --preset <id|all>         preset to capture (repeatable, default all)\n"
        "  --scene <path>            model to load (default <assets>/source/Sponza.gltf)\n"
        "  --frames <n>              warmup frames rendered before each capture (default 30)\n"
        "  --width <n> --height <n>  window extent (default 1280x720)\n"
        "  --list-presets            print available preset ids and exit\n"
        "  -h, --help                show this message");
}

std::vector<const CapturePreset*> resolveCapturePresets(const std::vector<std::string>& ids,
                                                        std::string& error)
{
    std::vector<const CapturePreset*> selected;

    if (ids.empty()) {
        for (const CapturePreset& preset : capturePresets()) {
            selected.push_back(&preset);
        }
        return selected;
    }

    for (const std::string& id : ids) {
        if (id == "all") {
            selected.clear();
            for (const CapturePreset& preset : capturePresets()) {
                selected.push_back(&preset);
            }
            return selected;
        }

        const CapturePreset* preset = findCapturePreset(id);
        if (preset == nullptr) {
            error = "Unknown capture preset '" + id + "' (use --list-presets)";
            return {};
        }
        selected.push_back(preset);
    }

    return selected;
}

int runCaptureMode(const CaptureOptions& options)
{
    std::string presetError;
    const std::vector<const CapturePreset*> selected =
        resolveCapturePresets(options.presetIds, presetError);
    if (!presetError.empty()) {
        spdlog::error("{}", presetError);
        return 1;
    }

    RendererInstance renderer;
    const RendererCreateInfo createInfo{
        .width = options.width,
        .height = options.height,
        .title = "ProjectOptimizedRenderer (capture)",
    };
    if (RendererInstance::create(createInfo, renderer) != RendererResult::Success) {
        spdlog::error("Renderer creation failed: {}", renderer.getLastError());
        return 1;
    }

    const std::string scenePath = options.scenePath.empty()
        ? std::string(ASSET_DIR) + "/source/Sponza.gltf"
        : options.scenePath;

    Model activeModel;
    SceneInfo activeSceneInfo;
    if (!loadViewerModel(renderer, scenePath, scenePath, activeModel, activeSceneInfo)) {
        spdlog::error("Could not load capture scene '{}'", scenePath);
        return 1;
    }

    uint32_t width = 0;
    uint32_t height = 0;
    renderer.getWindowExtent(width, height);
    const float aspect = height > 0
        ? static_cast<float>(width) / static_cast<float>(height)
        : 16.0f / 9.0f;

    Camera camera;
    camera.setPerspective(45.0f, aspect, 0.01f, 1000.0f);

    const float radius = std::max(activeSceneInfo.normalizedRadius, 0.001f);

    spdlog::info("Capture mode: {} preset(s) -> '{}' ({} warmup frames)",
                 selected.size(), options.outputDir, options.frameCount);

    for (const CapturePreset* preset : selected) {
        if (renderer.setAntiAliasingSettings(preset->antiAliasing) != RendererResult::Success) {
            spdlog::error("Failed to apply AA settings for preset '{}': {}",
                          preset->id, renderer.getLastError());
            return 1;
        }

        // Point lights live in the sticky scene submission, so re-submit the scene
        // once per preset to honor presets that isolate clustered lighting.
        if (renderer.submitScene(buildRenderScenePacket(activeModel, activeSceneInfo,
                                                        preset->pointLightsEnabled))
            != RendererResult::Success) {
            spdlog::error("Failed to submit scene for preset '{}': {}",
                          preset->id, renderer.getLastError());
            return 1;
        }

        ViewerState state;
        state.light       = preset->light;
        state.shadow      = preset->shadow;
        state.tonemap     = preset->tonemap;
        state.sky         = preset->sky;
        state.debugView   = preset->debug;
        state.antiAliasing = preset->antiAliasing;

        camera.setView(preset->camera.positionOffset * radius,
                       preset->camera.yawDegrees,
                       preset->camera.pitchDegrees);

        const RenderFramePacket packet = buildRenderFramePacket(camera, state);

        for (uint32_t frame = 0; frame < options.frameCount; ++frame) {
            if (renderer.renderFrame(packet) == RendererResult::Error) {
                spdlog::error("Renderer frame failed for preset '{}': {}",
                              preset->id, renderer.getLastError());
                return 1;
            }
        }

        const std::string outputPath = options.outputDir + "/" + preset->id + ".png";
        renderer.requestScreenshot(outputPath);
        if (renderer.renderFrame(packet) == RendererResult::Error) {
            spdlog::error("Renderer frame failed while capturing preset '{}': {}",
                          preset->id, renderer.getLastError());
            return 1;
        }

        spdlog::info("Captured preset '{}' -> {}", preset->id, outputPath);
    }

    renderer.shutdown();
    spdlog::info("Capture mode complete: {} PNG(s) written to '{}'",
                 selected.size(), options.outputDir);
    return 0;
}

void drawSunDirectionOverlay(const Camera& camera,
                             const RendererInstance& renderer,
                             const ViewerState& state)
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

    uint32_t fbW = 0;
    uint32_t fbH = 0;
    renderer.getWindowExtent(fbW, fbH);
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

int main(int argc, char** argv)
{
    auto stdoutSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto imguiSink  = std::make_shared<ImGuiLogSink>(500);

    auto logger = std::make_shared<spdlog::logger>(
        "POR", spdlog::sinks_init_list{ stdoutSink, imguiSink });
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::debug);
    spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");

    const ParsedArgs args = parseArgs(argc, argv);
    if (!args.error.empty()) {
        spdlog::error("{}", args.error);
        printUsage();
        return 1;
    }
    if (args.helpRequested) {
        printUsage();
        return 0;
    }
    if (args.listPresets) {
        spdlog::info("Capture presets:\n{}", capturePresetList());
        return 0;
    }
    if (args.captureRequested) {
        spdlog::info("ProjectOptimizedRenderer starting (capture mode)");
        try {
            return runCaptureMode(args.capture);
        } catch (const std::exception& e) {
            spdlog::critical("Fatal: {}", e.what());
            return 1;
        }
    }

    spdlog::info("ProjectOptimizedRenderer starting");

    try {
        RendererInstance renderer;
        const RendererCreateInfo createInfo{
            .width = 1280,
            .height = 720,
            .title = "ProjectOptimizedRenderer",
        };
        if (RendererInstance::create(createInfo, renderer) != RendererResult::Success) {
            throw std::runtime_error("Renderer creation failed: " + renderer.getLastError());
        }

        Model activeModel;
        SceneInfo activeSceneInfo;
        const std::string defaultModelPath = std::string(ASSET_DIR) + "/source/Sponza.gltf";
        if (!loadViewerModel(renderer, defaultModelPath, defaultModelPath,
                             activeModel, activeSceneInfo)) {
            throw std::runtime_error("Could not load the default viewer model");
        }

        ImGuiManager imguiManager;
        if (renderer.attachOverlay(&imguiManager) != RendererResult::Success) {
            throw std::runtime_error("ImGui overlay init failed: " + renderer.getLastError());
        }

        ViewerState viewerState;
        ImGuiManager::RenderToggles renderToggles;
        renderToggles.wireframe   = &viewerState.debugView.wireframe;
        renderToggles.showNormals = &viewerState.debugView.showNormals;
        imguiManager.setRenderToggles(renderToggles);

        uint32_t w = 0;
        uint32_t h = 0;
        renderer.getWindowExtent(w, h);

        Camera camera;
        const float aspectRatio = h > 0 ? static_cast<float>(w) / static_cast<float>(h) : 16.0f / 9.0f;
        camera.setPerspective(45.0f, aspectRatio, 0.01f, 1000.0f);
        camera.fitToScene(activeSceneInfo.normalizedRadius);

        registerViewerPanels(imguiManager,
                             renderer,
                             activeModel,
                             activeSceneInfo,
                             camera,
                             *imguiSink,
                             viewerState);

        auto lastTime = std::chrono::high_resolution_clock::now();
        auto* sdlWindow = static_cast<SDL_Window*>(renderer.getWindowHandle());

        SDL_SetWindowRelativeMouseMode(sdlWindow, viewerState.mouseCaptured);

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
                            SDL_SetWindowRelativeMouseMode(sdlWindow, viewerState.mouseCaptured);
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
                                SDL_SetWindowRelativeMouseMode(sdlWindow, true);
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
                        SDL_GetWindowSizeInPixels(sdlWindow, &pw, &ph);
                        if (pw > 0 && ph > 0) {
                            const float newAspect = static_cast<float>(pw) / static_cast<float>(ph);
                            camera.setPerspective(45.0f, newAspect, 0.01f, 1000.0f);
                            if (renderer.resize(static_cast<uint32_t>(pw), static_cast<uint32_t>(ph))
                                != RendererResult::Success) {
                                throw std::runtime_error("Renderer resize failed: " + renderer.getLastError());
                            }
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
                if (loadViewerModel(renderer,
                                    viewerState.pendingModelPath,
                                    defaultModelPath,
                                    activeModel,
                                    activeSceneInfo)) {
                    camera.fitToScene(activeSceneInfo.normalizedRadius);
                    viewerState.selectedMeshIndex = -1;
                }
                viewerState.pendingModelPath.clear();
            }

            if (!viewerState.pendingPanoramaPath.empty()) {
                if (renderer.loadHdrPanorama(viewerState.pendingPanoramaPath)
                    != RendererResult::Success) {
                    spdlog::error("Failed to load HDR panorama '{}': {}",
                                  viewerState.pendingPanoramaPath,
                                  renderer.getLastError());
                }
                viewerState.pendingPanoramaPath.clear();
            }

            if (viewerState.mouseCaptured) {
                camera.processKeyboard(keys);
            }
            camera.update(viewerState.deltaTime);

            imguiManager.beginFrame();
            drawSunDirectionOverlay(camera, renderer, viewerState);
            imguiManager.endFrame();

            const RendererResult frameResult =
                renderer.renderFrame(buildRenderFramePacket(camera, viewerState));
            if (frameResult == RendererResult::Error) {
                throw std::runtime_error("Renderer frame failed: " + renderer.getLastError());
            }
        }

        spdlog::info("Render loop ended, shutting down...");
        renderer.shutdown();

    } catch (const std::exception& e) {
        spdlog::critical("Fatal: {}", e.what());
        return 1;
    }

    return 0;
}
