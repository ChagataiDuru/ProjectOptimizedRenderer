#pragma once

#include "core/RenderSettings.h"

#include <array>
#include <string>

struct ViewerState {
    DirectionalLightData light;
    ShadowSettings shadow;
    TonemapSettings tonemap;
    SkySettings sky;
    DebugViewSettings debugView;
    AntiAliasingSettings antiAliasing;

    int selectedMeshIndex = -1;
    std::array<float, 90> fpsHistory{};
    int fpsHistoryOffset = 0;
    float deltaTime = 0.016f;

    std::string pendingModelPath;
    std::string pendingPanoramaPath;

    bool mouseCaptured = true;
};
