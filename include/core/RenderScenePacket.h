#pragma once

#include "core/DrawCommand.h"
#include "core/ShaderInterface.h"

#include <vector>

// Sticky scene submission owned by the caller for the duration of submitScene().
// Renderer-owned mesh/material resources stay persistent; this packet only
// describes which draws and clustered point lights are active for the scene.
struct RenderScenePacket {
    std::vector<DrawCommand> draws;
    std::vector<shader_interface::GpuPointLight> pointLights;
};
