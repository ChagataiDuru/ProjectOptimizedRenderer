#pragma once

#include "core/RenderSettings.h"

#include <glm/glm.hpp>
#include <string>
#include <vector>

// Scripted capture support for artifact reproduction and before/after comparison.
//
// Presets are plain data: a ViewerState-shaped settings bundle plus a deterministic
// camera placement. The capture runner in main.cpp applies one preset per PNG so
// rendering findings can be reproduced without driving the ImGui overlay by hand.

struct CaptureCamera {
    // Camera position in scene-normalized units: this offset is scaled by the
    // imported scene radius, then applied as the camera world position.
    glm::vec3 positionOffset{ 0.0f, 0.5f, 2.0f };
    float yawDegrees = 0.0f;
    float pitchDegrees = -10.0f;
};

struct CapturePreset {
    std::string id;
    std::string description;

    DirectionalLightData light{};
    ShadowSettings shadow{};
    TonemapSettings tonemap{};
    SkySettings sky{};
    DebugViewSettings debug{};
    AntiAliasingSettings antiAliasing{};
    CaptureCamera camera{};

    // When false the scene packet is submitted with no point lights, which isolates
    // clustered-forward lighting from the directional/shadow path.
    bool pointLightsEnabled = true;
};

// Stable, ordered preset table. New presets are appended so existing capture
// filenames keep their meaning across runs.
const std::vector<CapturePreset>& capturePresets();

// Returns nullptr when the id is unknown.
const CapturePreset* findCapturePreset(const std::string& id);

// One "id - description" line per preset, for --list-presets.
std::string capturePresetList();
