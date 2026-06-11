#pragma once

#include "core/RenderSettings.h"
#include "core/ShaderInterface.h"

#include <vector>

struct RenderFramePacket {
    CameraData camera;
    DirectionalLightData light;
    std::vector<shader_interface::GpuPointLight> pointLights;
    ShadowSettings shadow;
    TonemapSettings tonemap;
    SkySettings sky;
    DebugViewSettings debug;
};
