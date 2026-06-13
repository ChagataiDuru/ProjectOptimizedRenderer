#pragma once

#include "core/RenderSettings.h"

// Per-frame submission contains only frame-global state. Scene submission lives
// beside this packet in RenderScenePacket.
struct RenderFramePacket {
    CameraData camera;
    DirectionalLightData light;
    ShadowSettings shadow;
    TonemapSettings tonemap;
    SkySettings sky;
    DebugViewSettings debug;
};
