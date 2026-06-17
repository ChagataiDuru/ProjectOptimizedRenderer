#pragma once

class Camera;
class ImGuiLogSink;
class ImGuiManager;
class RendererInstance;
struct Model;
struct SceneInfo;
struct ViewerState;

void registerViewerPanels(ImGuiManager& imguiManager,
                          RendererInstance& renderer,
                          const Model& model,
                          const SceneInfo& sceneInfo,
                          Camera& camera,
                          ImGuiLogSink& imguiSink,
                          ViewerState& state);
