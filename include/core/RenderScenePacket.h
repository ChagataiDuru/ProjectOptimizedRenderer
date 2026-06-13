#pragma once

#include "core/DrawCommand.h"
#include "core/ShaderInterface.h"

#include <vector>

// Sticky scene submission passed into Renderer::submitScene(). The caller only
// owns the packet for the duration of the call; Renderer stores its own moved
// copy as the active submitted scene. Renderer-owned mesh/material resources
// stay persistent; this packet only describes which draws and clustered point
// lights are active for the scene.
struct RenderScenePacket {
    std::vector<DrawCommand> draws;
    std::vector<shader_interface::GpuPointLight> pointLights;
};
