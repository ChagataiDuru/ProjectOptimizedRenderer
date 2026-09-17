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

// Inside the atrium, eye height above the floor, looking along +X down the colonnade.
// Bounds-relative so the camera stays inside the building for any scene scale.
void setInteriorCamera(CapturePreset& preset)
{
    preset.camera.boundsRelative = true;
    preset.camera.positionOffset = glm::vec3(-0.55f, -0.75f, 0.0f);
    preset.camera.yawDegrees = -90.0f;
    preset.camera.pitchDegrees = -12.0f;
}

// Off-axis sun steep enough to reach the atrium floor through the open roof. Lower suns
// (e.g. 49 deg elevation) leave the whole floor in shadow, which a CPU ray cast against
// the scene confirms; this one lights ~17% of the floor and leaves visible shadow edges.
// The preset ids keep "grazing" for filename stability.
const glm::vec3 kGrazingSunDirection = glm::normalize(glm::vec3(0.55f, 0.82f, 0.10f));

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
    // stand inside the atrium, where the open roof lets the sun cast column shadows.
    presets.push_back(makeBase("shadow-interior",
                               "Interior atrium view down the colonnade"));
    setInteriorCamera(presets.back());

    presets.push_back(makeBase("shadow-interior-grazing-sun",
                               "Interior atrium with an off-axis sun (floor shadow edges)"));
    setInteriorCamera(presets.back());
    presets.back().light.direction = kGrazingSunDirection;

    presets.push_back(makeBase("shadow-interior-grazing-sun-pcf",
                               "Off-axis-sun interior with 16-tap PCF"));
    setInteriorCamera(presets.back());
    presets.back().light.direction = kGrazingSunDirection;
    presets.back().shadow.filterMode = 1;

    presets.push_back(makeBase("shadow-interior-grazing-sun-bias-zero",
                               "Off-axis-sun interior with PCF and zero depth bias (acne check)"));
    setInteriorCamera(presets.back());
    presets.back().light.direction = kGrazingSunDirection;
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

    // ── Interior comparisons (appended so existing ids keep their order) ─────
    presets.push_back(makeBase("shadow-interior-grazing-sun-hard",
                               "Off-axis-sun interior with hard single-tap shadows"));
    setInteriorCamera(presets.back());
    presets.back().light.direction = kGrazingSunDirection;
    presets.back().shadow.filterMode = 0;

    presets.push_back(makeBase("shadow-interior-grazing-sun-vsm",
                               "Off-axis-sun interior with VSM shadows"));
    setInteriorCamera(presets.back());
    presets.back().light.direction = kGrazingSunDirection;
    presets.back().shadow.filterMode = 2;

    presets.push_back(makeBase("msaa-4x-interior",
                               "Interior MSAA 4x (masked foliage in view)"));
    setInteriorCamera(presets.back());
    presets.back().antiAliasing.mode = AntiAliasingMode::MSAA;
    presets.back().antiAliasing.requestedSampleCount = MsaaSampleCount::X4;

    presets.push_back(makeBase("msaa-4x-a2c-interior",
                               "Interior MSAA 4x with alpha-to-coverage"));
    setInteriorCamera(presets.back());
    presets.back().antiAliasing.mode = AntiAliasingMode::MSAA;
    presets.back().antiAliasing.requestedSampleCount = MsaaSampleCount::X4;
    presets.back().antiAliasing.alphaToCoverageEnabled = true;

    presets.push_back(makeBase("msaa-sample-shading-interior",
                               "Interior MSAA 4x with sample shading at 0.5"));
    setInteriorCamera(presets.back());
    presets.back().antiAliasing.mode = AntiAliasingMode::MSAA;
    presets.back().antiAliasing.requestedSampleCount = MsaaSampleCount::X4;
    presets.back().antiAliasing.sampleShadingEnabled = true;
    presets.back().antiAliasing.minSampleShading = 0.5f;

    // Caster culling must be conservative: these two must render byte-identical. The
    // sun azimuth (zero X component) used to make the culling planes reject every mesh.
    for (const bool culling : { true, false }) {
        presets.push_back(makeBase(culling ? "shadow-cull-side-sun" : "shadow-cull-side-sun-off",
                                   culling ? "Atrium view, side sun, caster culling on"
                                           : "Atrium view, side sun, caster culling off"));
        presets.back().camera.boundsRelative = true;
        presets.back().camera.positionOffset = glm::vec3(0.354f, -0.52f, -0.0065f);
        presets.back().camera.yawDegrees = 88.7f;
        presets.back().camera.pitchDegrees = -13.0f;
        presets.back().light.direction = glm::normalize(glm::vec3(0.0f, 0.643f, 0.766f));
        presets.back().shadow.enableCasterCulling = culling;
    }

    // Image-based ambient lighting (ART-LGT-004).
    presets.push_back(makeBase("ibl-off", "Default framing with flat ambient instead of IBL"));
    presets.back().ibl.enabled = false;

    presets.push_back(makeBase("ibl-off-interior", "Interior framing with flat ambient instead of IBL"));
    setInteriorCamera(presets.back());
    presets.back().ibl.enabled = false;

    presets.push_back(makeBase("ibl-panorama-interior",
                               "Interior framing lit by a synthetic HDR panorama"));
    setInteriorCamera(presets.back());
    presets.back().sky.mode = 1;
    presets.back().syntheticPanorama = true;

    // Screen-space ambient occlusion (ART-LGT-005).
    presets.push_back(makeBase("ao-off-interior", "Interior framing with AO disabled"));
    setInteriorCamera(presets.back());
    presets.back().ao.enabled = false;

    presets.push_back(makeBase("ao-debug-interior", "Interior framing showing the AO buffer"));
    setInteriorCamera(presets.back());
    presets.back().ao.debugView = true;

    presets.push_back(makeBase("ao-on-exterior", "Default framing with AO enabled"));

    // The depth prepass must not change the image (compared with AO off, since AO
    // requires the prepass).
    presets.push_back(makeBase("perf-prepass-on-noao", "Default framing, prepass on, AO off"));
    presets.back().ao.enabled = false;

    presets.push_back(makeBase("perf-prepass-off-noao", "Default framing, prepass off, AO off"));
    presets.back().ao.enabled = false;
    presets.back().culling.enableDepthPrepass = false;

    presets.push_back(makeBase("perf-prepass-on-noao-interior", "Interior, prepass on, AO off"));
    setInteriorCamera(presets.back());
    presets.back().ao.enabled = false;

    presets.push_back(makeBase("perf-prepass-off-noao-interior", "Interior, prepass off, AO off"));
    setInteriorCamera(presets.back());
    presets.back().ao.enabled = false;
    presets.back().culling.enableDepthPrepass = false;

    // Main-pass visibility must not change the image (must-match against the defaults).
    presets.push_back(makeBase("perf-culling-off", "Default framing, main-pass frustum culling off"));
    presets.back().culling.enableFrustumCulling = false;

    presets.push_back(makeBase("perf-culling-off-interior", "Interior framing, main-pass frustum culling off"));
    setInteriorCamera(presets.back());
    presets.back().culling.enableFrustumCulling = false;

    presets.push_back(makeBase("perf-sort-off", "Default framing, draw sorting off"));
    presets.back().culling.sortDraws = false;

    presets.push_back(makeBase("perf-sort-off-interior", "Interior framing, draw sorting off"));
    setInteriorCamera(presets.back());
    presets.back().culling.sortDraws = false;

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
