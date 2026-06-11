#pragma once

class Camera;
class ImGuiLogSink;
class ImGuiManager;
class Renderer;
class Window;
struct ViewerState;

void registerViewerPanels(ImGuiManager& imguiManager,
                          Renderer& renderer,
                          Camera& camera,
                          Window& window,
                          ImGuiLogSink& imguiSink,
                          ViewerState& state);
