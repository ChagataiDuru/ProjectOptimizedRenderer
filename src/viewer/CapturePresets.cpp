#include "viewer/CapturePresets.h"

#include <algorithm>

namespace {

CapturePreset makeBase(const char* id, const char* description)
{
    CapturePreset preset{};
    preset.id = id;
    preset.description = description;
    return preset;
}

std::vector<CapturePreset> buildCapturePresets()
{
    std::vector<CapturePreset> presets;

    // ── Scene framing ────────────────────────────────────────────────────────
    presets.push_back(makeBase("scene-overview",
                               "Default framing, PCF shadows, no MSAA, Reinhard tonemap"));
    presets.push_back(makeBase("scene-floor-close",
                               "Low grazing view across the floor (shadow acne / bias)"));
    presets.back().camera.positionOffset = glm::vec3(0.0f, 0.12f, 0.55f);
    presets.back().camera.pitchDegrees = -32.0f;

    presets.push_back(makeBase("scene-wall-close",
                               "Close three-quarter view of a wall/column (normal maps, bias)"));
    presets.back().camera.positionOffset = glm::vec3(0.45f, 0.28f, 0.75f);
    presets.back().camera.yawDegrees = -25.0f;
    presets.back().camera.pitchDegrees = -6.0f;

    // ── Shadow filter modes and settings ─────────────────────────────────────
    presets.push_back(makeBase("shadow-hard", "Shadow filter mode: hard single-tap"));
    presets.back().shadow.filterMode = 0;

    presets.push_back(makeBase("shadow-pcf", "Shadow filter mode: 16-tap Poisson PCF"));
    presets.back().shadow.filterMode = 1;

    presets.push_back(makeBase("shadow-vsm", "Shadow filter mode: VSM moments + blur"));
    presets.back().shadow.filterMode = 2;

    presets.push_back(makeBase("shadow-bias-zero",
                               "PCF with zero depth bias (exposes acne / bias interaction)"));
    presets.back().shadow.filterMode = 1;
    presets.back().shadow.depthBiasConstant = 0.0f;
    presets.back().shadow.depthBiasSlope = 0.0f;

    presets.push_back(makeBase("shadow-distance-short",
                               "PCF with a 20-unit shadow distance (cascade splits / cutoff)"));
    presets.back().shadow.filterMode = 1;
    presets.back().shadow.maxDistance = 20.0f;

    presets.push_back(makeBase("shadow-cull-off",
                               "PCF with shadow caster culling disabled (culling correctness)"));
    presets.back().shadow.filterMode = 1;
    presets.back().shadow.enableCasterCulling = false;

    presets.push_back(makeBase("cascades-debug", "Cascade false-color debug overlay"));
    presets.back().shadow.debugCascades = true;

    // ── Shadow visibility validation ─────────────────────────────────────────
    // Exterior views of Sponza contain almost no shadowed receivers, so these presets
    // look down an interior corridor with a grazing sun to make shadows observable.
    presets.push_back(makeBase("shadow-interior",
                               "Interior corridor looking down at the floor"));
    presets.back().camera.positionOffset = glm::vec3(0.0f, 0.30f, 0.85f);
    presets.back().camera.pitchDegrees = -42.0f;

    presets.push_back(makeBase("shadow-interior-grazing-sun",
                               "Interior corridor with a low grazing sun (long shadows)"));
    presets.back().camera.positionOffset = glm::vec3(0.0f, 0.30f, 0.85f);
    presets.back().camera.pitchDegrees = -42.0f;
    presets.back().light.direction = glm::normalize(glm::vec3(1.0f, 0.22f, 0.35f));

    presets.push_back(makeBase("shadow-interior-grazing-sun-pcf",
                               "Grazing-sun interior with 16-tap PCF"));
    presets.back().camera.positionOffset = glm::vec3(0.0f, 0.30f, 0.85f);
    presets.back().camera.pitchDegrees = -42.0f;
    presets.back().light.direction = glm::normalize(glm::vec3(1.0f, 0.22f, 0.35f));
    presets.back().shadow.filterMode = 1;

    presets.push_back(makeBase("shadow-interior-grazing-sun-bias-zero",
                               "Grazing-sun interior with PCF and zero depth bias (acne check)"));
    presets.back().camera.positionOffset = glm::vec3(0.0f, 0.30f, 0.85f);
    presets.back().camera.pitchDegrees = -42.0f;
    presets.back().light.direction = glm::normalize(glm::vec3(1.0f, 0.22f, 0.35f));
    presets.back().shadow.filterMode = 1;
    presets.back().shadow.depthBiasConstant = 0.0f;
    presets.back().shadow.depthBiasSlope = 0.0f;

    // ── Anti-aliasing ────────────────────────────────────────────────────────
    presets.push_back(makeBase("msaa-4x", "MSAA 4x scene rendering"));
    presets.back().antiAliasing.mode = AntiAliasingMode::MSAA;
    presets.back().antiAliasing.requestedSampleCount = MsaaSampleCount::X4;

    presets.push_back(makeBase("msaa-4x-a2c", "MSAA 4x with alpha-to-coverage masked materials"));
    presets.back().antiAliasing.mode = AntiAliasingMode::MSAA;
    presets.back().antiAliasing.requestedSampleCount = MsaaSampleCount::X4;
    presets.back().antiAliasing.alphaToCoverageEnabled = true;

    presets.push_back(makeBase("msaa-sample-shading", "MSAA 4x with sample shading at 0.5"));
    presets.back().antiAliasing.mode = AntiAliasingMode::MSAA;
    presets.back().antiAliasing.requestedSampleCount = MsaaSampleCount::X4;
    presets.back().antiAliasing.sampleShadingEnabled = true;
    presets.back().antiAliasing.minSampleShading = 0.5f;

    // ── Tone mapping ─────────────────────────────────────────────────────────
    presets.push_back(makeBase("tonemap-reinhard", "Tone map: Reinhard"));
    presets.back().tonemap.mode = 0;

    presets.push_back(makeBase("tonemap-agx", "Tone map: AgX"));
    presets.back().tonemap.mode = 1;

    presets.push_back(makeBase("tonemap-pbr-neutral", "Tone map: Khronos PBR Neutral"));
    presets.back().tonemap.mode = 2;

    presets.push_back(makeBase("tonemap-split", "Split-screen tone map comparison (AgX right)"));
    presets.back().tonemap.splitScreen = true;
    presets.back().tonemap.splitRightMode = 1;

    // ── Sky / lighting isolation ─────────────────────────────────────────────
    presets.push_back(makeBase("sky-procedural", "Procedural Rayleigh+Mie sky"));
    presets.back().sky.enabled = true;
    presets.back().sky.mode = 0;

    presets.push_back(makeBase("sky-off", "Sky disabled (geometry plus clear color only)"));
    presets.back().sky.enabled = false;

    presets.push_back(makeBase("clusters-off",
                               "No point lights submitted (isolates clustered lighting)"));
    presets.back().pointLightsEnabled = false;

    presets.push_back(makeBase("normals-debug", "Normals debug view (normal map / TBN artifacts)"));
    presets.back().debug.showNormals = true;

    return presets;
}

}  // namespace

const std::vector<CapturePreset>& capturePresets()
{
    static const std::vector<CapturePreset> presets = buildCapturePresets();
    return presets;
}

const CapturePreset* findCapturePreset(const std::string& id)
{
    const auto& presets = capturePresets();
    const auto it = std::find_if(presets.begin(), presets.end(),
                                 [&id](const CapturePreset& preset) { return preset.id == id; });
    return it == presets.end() ? nullptr : &(*it);
}

std::string capturePresetList()
{
    std::string list;
    for (const CapturePreset& preset : capturePresets()) {
        list += preset.id;
        list += " - ";
        list += preset.description;
        list += '\n';
    }
    return list;
}
