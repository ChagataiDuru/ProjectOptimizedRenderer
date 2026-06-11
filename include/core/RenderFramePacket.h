#pragma once

#include "core/RenderSettings.h"

struct RenderFramePacket {
    CameraData camera;
    DirectionalLightData light;
    ShadowSettings shadow;
    TonemapSettings tonemap;
    SkySettings sky;
    DebugViewSettings debug;
};
