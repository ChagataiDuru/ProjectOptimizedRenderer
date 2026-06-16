#include "viewer/ViewerPanels.h"

#include "core/Camera.h"
#include "core/Renderer.h"
#include "core/Window.h"
#include "debug/ImGuiManager.h"
#include "debug/LogSink.h"
#include "resource/Material.h"
#include "viewer/ViewerState.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_dialog.h>
#include <IconsFontAwesome6.h>
#include <algorithm>
#include <array>
#include <glm/gtc/quaternion.hpp>
#include <imgui.h>
#include <iterator>
#include <spdlog/spdlog.h>
#include <string>

namespace {

int sampleCountToComboIndex(MsaaSampleCount count)
{
    switch (count) {
        case MsaaSampleCount::X1: return 0;
        case MsaaSampleCount::X2: return 1;
        case MsaaSampleCount::X4: return 2;
        case MsaaSampleCount::X8: return 3;
    }
    return 0;
}

MsaaSampleCount comboIndexToSampleCount(int index)
{
    switch (index) {
        case 1: return MsaaSampleCount::X2;
        case 2: return MsaaSampleCount::X4;
        case 3: return MsaaSampleCount::X8;
        case 0:
        default: return MsaaSampleCount::X1;
    }
}

uint32_t sampleCountValue(MsaaSampleCount count)
{
    return static_cast<uint32_t>(count);
}

std::string supportedSampleCountsLabel(const std::array<bool, 4>& supported)
{
    static const char* labels[] = { "1x", "2x", "4x", "8x" };
    std::string result;
    for (size_t i = 0; i < supported.size(); ++i) {
        if (!supported[i]) {
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

void registerViewerPanels(ImGuiManager& imguiManager,
                          Renderer& renderer,
                          Camera& camera,
                          Window& window,
                          ImGuiLogSink& imguiSink,
                          ViewerState& state)
{
    imguiManager.setOpenModelCallback([&]() {
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
            &state.pendingModelPath,
            static_cast<SDL_Window*>(window.getHandle()),
            filters,
            static_cast<int>(std::size(filters)),
            nullptr,
            false);
    });

    imguiManager.registerPanel(ICON_FA_CHART_LINE " Performance", [&]() {
        const float fps = (state.deltaTime > 0.0f) ? (1.0f / state.deltaTime) : 0.0f;
        state.fpsHistory[static_cast<size_t>(state.fpsHistoryOffset)] = fps;
        state.fpsHistoryOffset =
            (state.fpsHistoryOffset + 1) % static_cast<int>(state.fpsHistory.size());

        ImGui::Text("FPS:        %.1f", fps);
        ImGui::Text("Frame time: %.2f ms", state.deltaTime * 1000.0f);
        ImGui::PlotLines("##fps",
                         state.fpsHistory.data(),
                         static_cast<int>(state.fpsHistory.size()),
                         state.fpsHistoryOffset,
                         "FPS",
                         0.0f,
                         300.0f,
                         ImVec2(0.0f, 60.0f));
    }, DockLocation::Bottom);

    imguiManager.registerPanel(ICON_FA_VIDEO " Camera", [&]() {
        const glm::vec3 pos = camera.getPosition();
        ImGui::Text("Position:  (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);

        const glm::vec3 euler = glm::degrees(glm::eulerAngles(camera.getOrientation()));
        ImGui::Text("Pitch/Yaw: (%.1f deg, %.1f deg)", euler.x, euler.y);
        ImGui::Spacing();
        ImGui::TextDisabled("WASD: move  Mouse: look  F1: toggle cursor");
    }, DockLocation::Right);

    imguiManager.registerPanel(ICON_FA_SUN " Light", [&]() {
        ImGui::SliderFloat3("Direction", &state.light.direction.x, -1.0f, 1.0f);
        ImGui::ColorEdit3("Color", &state.light.color.x);
        ImGui::SliderFloat("Intensity", &state.light.intensity, 0.0f, 10.0f);
        ImGui::SliderFloat("Ambient", &state.light.ambient, 0.0f, 1.0f);

        ImGui::Separator();
        ImGui::Text(ICON_FA_LAYER_GROUP " Cascaded Shadows");
        ImGui::SliderFloat("Max Distance", &state.shadow.maxDistance, 10.0f, 300.0f, "%.0f");
        ImGui::SliderFloat("CSM Lambda", &state.shadow.csmLambda, 0.0f, 1.0f,
                           "%.2f", ImGuiSliderFlags_None);
        ImGui::Checkbox("Debug Cascades", &state.shadow.debugCascades);
        if (state.shadow.debugCascades) {
            ImGui::SameLine();
            ImGui::TextDisabled("(red/green/blue/yellow = cascades 0-3)");
        }

        ImGui::Separator();
        ImGui::Text(ICON_FA_DROPLET " Shadow Filter");
        const char* filterItems[] = { "None (hard)", "PCF (soft)", "VSM (variance)" };
        ImGui::Combo("Mode", &state.shadow.filterMode, filterItems, 3);
        const char* qualityItems[] = { "Low 1024", "Medium 1024", "High 2048", "Ultra 2048" };
        ImGui::Combo("Quality", &state.shadow.qualityPreset, qualityItems, 4);
        ImGui::SliderFloat("Bias Constant", &state.shadow.depthBiasConstant, 0.0f, 5.0f, "%.2f");
        ImGui::SliderFloat("Bias Slope", &state.shadow.depthBiasSlope, 0.0f, 6.0f, "%.2f");
        ImGui::Checkbox("Caster Culling", &state.shadow.enableCasterCulling);
        const char* cullItems[] = { "None", "Front", "Back" };
        ImGui::Combo("Shadow Cull", &state.shadow.cullMode, cullItems, 3);
        if (state.shadow.filterMode == 1) {
            ImGui::SliderFloat("PCF Spread (texels)",
                               &state.shadow.pcfSpreadRadius,
                               0.5f,
                               8.0f,
                               "%.1f");
        }
        if (state.shadow.filterMode == 2) {
            ImGui::SliderFloat("VSM Bleed Reduction",
                               &state.shadow.vsmBleedReduction,
                               0.0f,
                               0.9f,
                               "%.2f");
            ImGui::TextDisabled("Higher = less bleeding, harder edges");
        }

        ImGui::Separator();
        ImGui::Text(ICON_FA_CLOUD " Sky");
        ImGui::Checkbox("Enable Sky", &state.sky.enabled);

        if (state.sky.enabled) {
            const char* skyModeNames[] = { "Procedural (Rayleigh+Mie)", "HDR Panorama" };
            ImGui::Combo("Sky Mode", &state.sky.mode, skyModeNames, 2);

            if (state.sky.mode == 1) {
                if (ImGui::Button("Load HDR Panorama...")) {
                    static const SDL_DialogFileFilter hdrFilters[] = {
                        { "HDR images", "hdr" },
                        { "All files",  "*" },
                    };

                    SDL_ShowOpenFileDialog(
                        [](void* userdata, const char* const* filelist, int /*filter*/) {
                            if (filelist && filelist[0]) {
                                *static_cast<std::string*>(userdata) = filelist[0];
                            }
                        },
                        &state.pendingPanoramaPath,
                        static_cast<SDL_Window*>(window.getHandle()),
                        hdrFilters,
                        static_cast<int>(std::size(hdrFilters)),
                        nullptr,
                        false);
                }
                ImGui::TextDisabled("Expects an equirectangular .hdr file");
            }
        }
    }, DockLocation::Right);

    imguiManager.registerPanel(ICON_FA_WAND_MAGIC_SPARKLES " Post Processing", [&]() {
        static const char* tonemapNames[] = { "Reinhard", "AgX", "PBR Neutral" };

        ImGui::Combo("Operator", &state.tonemap.mode, tonemapNames, 3);
        ImGui::SliderFloat("Exposure (EV)", &state.tonemap.exposure, -4.0f, 4.0f, "%.2f");
        ImGui::TextDisabled("EV +1 = 2x brighter,  EV -1 = 2x darker");

        ImGui::Separator();
        ImGui::Checkbox("Split-Screen Compare", &state.tonemap.splitScreen);
        if (state.tonemap.splitScreen) {
            ImGui::Indent();
            ImGui::Text("Left:  %s", tonemapNames[state.tonemap.mode]);
            ImGui::Combo("Right", &state.tonemap.splitRightMode, tonemapNames, 3);
            ImGui::Unindent();
            ImGui::TextDisabled("White line = split center");
        }
    }, DockLocation::Right);

    imguiManager.registerPanel(ICON_FA_WAND_MAGIC_SPARKLES " Anti-Aliasing", [&]() {
        AntiAliasingSettings pending = state.antiAliasing;
        const AntiAliasingStatus& status = renderer.getAntiAliasingStatus();
        bool changed = false;

        int mode = pending.mode == AntiAliasingMode::MSAA ? 1 : 0;
        static const char* modeNames[] = { "None", "MSAA" };
        if (ImGui::Combo("Mode", &mode, modeNames, 2)) {
            pending.mode = mode == 1 ? AntiAliasingMode::MSAA : AntiAliasingMode::None;
            changed = true;
        }

        int sampleIndex = sampleCountToComboIndex(pending.requestedSampleCount);
        static const char* sampleNames[] = { "1x", "2x", "4x", "8x" };
        ImGui::BeginDisabled(pending.mode == AntiAliasingMode::None);
        if (ImGui::Combo("Requested Samples", &sampleIndex, sampleNames, 4)) {
            pending.requestedSampleCount = comboIndexToSampleCount(sampleIndex);
            changed = true;
        }
        ImGui::EndDisabled();

        ImGui::Text("Supported: %s",
                    supportedSampleCountsLabel(status.supportedSampleCounts).c_str());
        ImGui::Text("Active:    %ux", sampleCountValue(status.activeSampleCount));
        if (status.activeSampleCount != status.requestedSampleCount &&
            pending.mode == AntiAliasingMode::MSAA) {
            ImGui::TextDisabled("Requested count fell back to the nearest supported count");
        }

        ImGui::Separator();
        ImGui::Text("Sample Shading");
        const bool sampleShadingSupported = status.sampleRateShadingSupported;
        if (!sampleShadingSupported) {
            ImGui::TextDisabled("Unavailable on this Vulkan device");
        }

        ImGui::BeginDisabled(!sampleShadingSupported);
        bool sampleShadingEnabled = pending.sampleShadingEnabled;
        if (ImGui::Checkbox("Enable Sample Shading", &sampleShadingEnabled)) {
            pending.sampleShadingEnabled = sampleShadingEnabled;
            changed = true;
        }

        float minSampleShading = pending.minSampleShading;
        if (ImGui::SliderFloat("Minimum Fraction", &minSampleShading, 0.0f, 1.0f, "%.2f")) {
            pending.minSampleShading = minSampleShading;
            changed = true;
        }

        const float presetValues[] = { 0.0f, 0.25f, 0.5f, 1.0f };
        const char* presetLabels[] = { "0.00", "0.25", "0.50", "1.00" };
        for (size_t i = 0; i < std::size(presetValues); ++i) {
            if (i > 0) {
                ImGui::SameLine();
            }
            if (ImGui::Button(presetLabels[i])) {
                pending.minSampleShading = presetValues[i];
                changed = true;
            }
        }
        ImGui::EndDisabled();

        ImGui::TextDisabled("Sample shading is separate from sample count; it does not render the scene at a higher resolution.");
        ImGui::Text("Effective: %s, %.2f",
                    status.sampleShadingEnabled ? "enabled" : "disabled",
                    status.minSampleShading);

        if (changed) {
            state.antiAliasing = pending;
            renderer.setAntiAliasingSettings(state.antiAliasing);
        }
    }, DockLocation::Right);

    imguiManager.registerPanel(ICON_FA_CUBES " Render Stats", [&]() {
        const auto& stats = renderer.getRenderStats();
        ImGui::Text("Draw calls:  %u", stats.drawCalls);
        ImGui::Text("Triangles:   %u", stats.triangles);
        ImGui::Text("Meshes:      %u", stats.meshCount);
        ImGui::Text("Materials:   %u", stats.materialCount);
        ImGui::Text("Textures:    %u", stats.textureCount);
        if (stats.textureMemoryBytes > 0) {
            ImGui::Text("Tex memory:  %.1f MB",
                        stats.textureMemoryBytes / (1024.0f * 1024.0f));
        }

        ImGui::Separator();
        if (ImGui::TreeNode("Shadow Culling")) {
            const auto& shadow = renderer.getShadowDebugInfo();
            const auto& culled = shadow.culledMeshes;
            const auto& total  = shadow.totalMeshes;
            const auto& splits = shadow.splitDepths;
            ImGui::Text("Splits: %.2f  %.2f  %.2f  %.2f",
                        splits.x, splits.y, splits.z, splits.w);
            ImGui::Text("Max distance: %.1f", shadow.maxDistance);
            ImGui::Text("Resolution: %u", shadow.cascadeResolution);
            ImGui::Text("VSM allocated: %s", shadow.vsmAllocated ? "yes" : "no");
            ImGui::Text("Shadow memory: %.1f MB", shadow.estimatedMemoryMB);
            if (camera.getFarZ() > state.shadow.maxDistance * 4.0f) {
                ImGui::TextDisabled("Camera far plane is much larger than shadow distance");
            }
            for (uint32_t c = 0; c < Renderer::CASCADE_COUNT; ++c) {
                const uint32_t drawn = shadow.drawnMeshes[c];
                ImGui::Text("Cascade %u: %u / %u drawn (-%u culled)",
                            c, drawn, total[c], culled[c]);
            }
            ImGui::TreePop();
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

        const float shadowMs  = timer.getElapsedMs("ShadowPass_Begin", "ShadowPass_End");
        const bool  hasBlur   = (state.shadow.filterMode == 2);
        const float blurMs    = hasBlur
            ? timer.getElapsedMs("BlurPass_Begin", "BlurPass_End") : 0.0f;
        const float clusterMs = timer.getElapsedMs("ClusterCull_Begin", "ClusterCull_End");
        const float sceneMs   = timer.getElapsedMs("ScenePass_Begin", "ScenePass_End");
        const float tonemapMs = timer.getElapsedMs("TonemapPass_Begin", "TonemapPass_End");
        const float imguiMs   = timer.getElapsedMs("TonemapPass_End", "ImGuiPass_End");
        const float totalMs   = shadowMs + blurMs + clusterMs + sceneMs + tonemapMs + imguiMs;
        const float budget    = 16.67f;

        ImGui::Text("Budget: %.2f / %.2f ms (60 Hz)", totalMs, budget);

        const float barWidth  = ImGui::GetContentRegionAvail().x;
        const float barHeight = 20.0f;
        ImVec2 barStart       = ImGui::GetCursorScreenPos();
        ImDrawList* draw      = ImGui::GetWindowDrawList();

        draw->AddRectFilled(barStart,
                            ImVec2(barStart.x + barWidth, barStart.y + barHeight),
                            IM_COL32(40, 40, 45, 255),
                            4.0f);

        const float shadowW = std::min((shadowMs / budget) * barWidth, barWidth);
        draw->AddRectFilled(barStart,
                            ImVec2(barStart.x + shadowW, barStart.y + barHeight),
                            IM_COL32(255, 165, 0, 200),
                            4.0f);

        const float blurW = hasBlur
            ? std::min((blurMs / budget) * barWidth, barWidth - shadowW) : 0.0f;
        if (blurW > 0.0f) {
            draw->AddRectFilled(ImVec2(barStart.x + shadowW, barStart.y),
                                ImVec2(barStart.x + shadowW + blurW,
                                       barStart.y + barHeight),
                                IM_COL32(180, 80, 220, 200),
                                4.0f);
        }

        const float usedW  = shadowW + blurW;
        const float clusterW = std::min((clusterMs / budget) * barWidth, barWidth - usedW);
        draw->AddRectFilled(ImVec2(barStart.x + usedW, barStart.y),
                            ImVec2(barStart.x + usedW + clusterW, barStart.y + barHeight),
                            IM_COL32(80, 210, 220, 200),
                            4.0f);

        const float usedWCluster = usedW + clusterW;
        const float sceneW = std::min((sceneMs / budget) * barWidth, barWidth - usedWCluster);
        draw->AddRectFilled(ImVec2(barStart.x + usedWCluster, barStart.y),
                            ImVec2(barStart.x + usedWCluster + sceneW, barStart.y + barHeight),
                            IM_COL32(66, 150, 250, 200),
                            4.0f);

        const float usedW2   = usedWCluster + sceneW;
        const float tonemapW = std::min((tonemapMs / budget) * barWidth,
                                        barWidth - usedW2);
        draw->AddRectFilled(ImVec2(barStart.x + usedW2, barStart.y),
                            ImVec2(barStart.x + usedW2 + tonemapW,
                                   barStart.y + barHeight),
                            IM_COL32(230, 210, 60, 200),
                            4.0f);

        const float usedW3 = usedW2 + tonemapW;
        const float imguiW = std::min((imguiMs / budget) * barWidth,
                                      barWidth - usedW3);
        draw->AddRectFilled(ImVec2(barStart.x + usedW3, barStart.y),
                            ImVec2(barStart.x + usedW3 + imguiW,
                                   barStart.y + barHeight),
                            IM_COL32(80, 200, 120, 200),
                            4.0f);

        draw->AddLine(ImVec2(barStart.x + barWidth, barStart.y),
                      ImVec2(barStart.x + barWidth, barStart.y + barHeight),
                      IM_COL32(255, 80, 80, 200),
                      2.0f);

        ImGui::Dummy(ImVec2(barWidth, barHeight + 4.0f));

        ImGui::ColorButton("##sh", ImVec4(1.0f, 0.65f, 0.0f, 1.0f),
                           ImGuiColorEditFlags_NoTooltip, ImVec2(10.0f, 10.0f));
        ImGui::SameLine();
        ImGui::Text("Shadow: %.3f ms", shadowMs);

        if (hasBlur) {
            ImGui::ColorButton("##bl", ImVec4(0.71f, 0.31f, 0.86f, 1.0f),
                               ImGuiColorEditFlags_NoTooltip, ImVec2(10.0f, 10.0f));
            ImGui::SameLine();
            ImGui::Text("Blur: %.3f ms", blurMs);
        }

        ImGui::ColorButton("##sc", ImVec4(0.26f, 0.59f, 0.98f, 1.0f),
                           ImGuiColorEditFlags_NoTooltip, ImVec2(10.0f, 10.0f));
        ImGui::SameLine();
        ImGui::Text("Scene: %.3f ms", sceneMs);

        ImGui::ColorButton("##cl", ImVec4(0.31f, 0.82f, 0.86f, 1.0f),
                           ImGuiColorEditFlags_NoTooltip, ImVec2(10.0f, 10.0f));
        ImGui::SameLine();
        ImGui::Text("Cluster cull: %.3f ms", clusterMs);

        ImGui::ColorButton("##tm", ImVec4(0.90f, 0.82f, 0.23f, 1.0f),
                           ImGuiColorEditFlags_NoTooltip, ImVec2(10.0f, 10.0f));
        ImGui::SameLine();
        ImGui::Text("Tonemap: %.3f ms", tonemapMs);

        ImGui::ColorButton("##im", ImVec4(0.31f, 0.78f, 0.47f, 1.0f),
                           ImGuiColorEditFlags_NoTooltip, ImVec2(10.0f, 10.0f));
        ImGui::SameLine();
        ImGui::Text("ImGui: %.3f ms", imguiMs);
    }, DockLocation::Bottom);

    imguiManager.registerPanel(ICON_FA_SITEMAP " Scene", [&]() {
        const auto& model = renderer.getImportedModel();
        const auto& stats = renderer.getRenderStats();

        ImGui::Text("%zu meshes, %u materials", model.meshes.size(), stats.materialCount);
        ImGui::Separator();

        if (ImGui::BeginChild("MeshList", ImVec2(0, 0), ImGuiChildFlags_None)) {
            for (int i = 0; i < static_cast<int>(model.meshes.size()); ++i) {
                const auto& mesh = model.meshes[static_cast<size_t>(i)];
                const bool isSelected = (state.selectedMeshIndex == i);

                if (mesh.materialIndex >= 0 &&
                    mesh.materialIndex < static_cast<int>(model.materials.size())) {
                    const auto& mat = model.materials[static_cast<size_t>(mesh.materialIndex)];
                    ImVec4 col(mat.baseColorFactor.r,
                               mat.baseColorFactor.g,
                               mat.baseColorFactor.b,
                               1.0f);
                    ImGui::ColorButton("##c", col,
                                       ImGuiColorEditFlags_NoTooltip |
                                       ImGuiColorEditFlags_NoDragDrop,
                                       ImVec2(12.0f, 12.0f));
                    ImGui::SameLine();
                }

                if (ImGui::Selectable(mesh.name.c_str(), isSelected)) {
                    state.selectedMeshIndex = i;
                }
            }
        }
        ImGui::EndChild();
    }, DockLocation::Right);

    imguiManager.registerPanel(ICON_FA_SLIDERS " Properties", [&]() {
        const auto& model     = renderer.getImportedModel();
        const auto& sceneInfo = renderer.getSceneInfo();

        ImGui::Text(ICON_FA_CUBE " Scene Bounds");
        ImGui::Separator();
        ImGui::Text("Original: (%.1f, %.1f, %.1f)",
                    model.boundsMin.x, model.boundsMin.y, model.boundsMin.z);
        ImGui::Text("       to (%.1f, %.1f, %.1f)",
                    model.boundsMax.x, model.boundsMax.y, model.boundsMax.z);
        ImGui::Text("Scale factor:      %.4f", sceneInfo.scaleFactor);
        ImGui::Text("Normalized radius: %.2f", sceneInfo.normalizedRadius);
        ImGui::Spacing();

        if (state.selectedMeshIndex < 0 ||
            state.selectedMeshIndex >= static_cast<int>(model.meshes.size())) {
            ImGui::TextDisabled("Select a mesh in the Scene panel");
            return;
        }

        const auto& mesh = model.meshes[static_cast<size_t>(state.selectedMeshIndex)];

        ImGui::Text(ICON_FA_CUBE " %s", mesh.name.c_str());
        ImGui::Separator();
        ImGui::Text("Vertices:  %zu", mesh.vertices.size());
        ImGui::Text("Triangles: %zu", mesh.indices.size() / 3);

        if (mesh.materialIndex >= 0 &&
            mesh.materialIndex < static_cast<int>(model.materials.size())) {
            const auto& mat = model.materials[static_cast<size_t>(mesh.materialIndex)];

            ImGui::Separator();
            ImGui::Text(ICON_FA_PALETTE " Material: %s", mat.name.c_str());

            ImVec4 baseColor(mat.baseColorFactor.r,
                             mat.baseColorFactor.g,
                             mat.baseColorFactor.b,
                             mat.baseColorFactor.a);
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

            auto showTex = [&](const char* label, int32_t idx) {
                if (idx >= 0 && idx < static_cast<int>(model.textures.size())) {
                    std::string path = model.textures[static_cast<size_t>(idx)].path;
                    const auto slash = path.find_last_of("/\\");
                    if (slash != std::string::npos) {
                        path = path.substr(slash + 1);
                    }
                    ImGui::Text("  %s: %s", label, path.c_str());
                } else {
                    ImGui::TextDisabled("  %s: (none)", label);
                }
            };

            showTex("Albedo", mat.albedoTextureIndex);
            showTex("Normal", mat.normalTextureIndex);
            showTex("Metal/Rgh", mat.metallicRoughnessTextureIndex);
        } else {
            ImGui::TextDisabled("No material assigned");
        }
    }, DockLocation::Right);

    imguiManager.registerPanel(ICON_FA_TERMINAL " Console", [&]() {
        static bool showDebug = false;
        static bool showInfo  = true;
        static bool showWarn  = true;
        static bool showError = true;

        ImGui::Checkbox("Debug", &showDebug); ImGui::SameLine();
        ImGui::Checkbox("Info",  &showInfo);  ImGui::SameLine();
        ImGui::Checkbox("Warn",  &showWarn);  ImGui::SameLine();
        ImGui::Checkbox("Error", &showError); ImGui::SameLine();
        if (ImGui::Button(ICON_FA_TRASH " Clear")) {
            imguiSink.clear();
        }

        ImGui::Separator();

        if (ImGui::BeginChild("LogScroll", ImVec2(0, 0), ImGuiChildFlags_None)) {
            for (const auto& entry : imguiSink.getEntries()) {
                bool show = false;
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
                        show = true;
                        break;
                }

                if (show) {
                    ImGui::PushStyleColor(ImGuiCol_Text, color);
                    ImGui::TextUnformatted(entry.message.c_str());
                    ImGui::PopStyleColor();
                }
            }

            static size_t lastCount = 0;
            const size_t currentCount = imguiSink.getEntryCount();
            if (currentCount != lastCount) {
                ImGui::SetScrollHereY(1.0f);
                lastCount = currentCount;
            }
        }
        ImGui::EndChild();
    }, DockLocation::Bottom);
}
