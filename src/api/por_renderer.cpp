#include "por/por_renderer.h"

#include "core/RendererInstance.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <string>

struct por_renderer_t {
    RendererInstance instance;
    std::string lastError;
};

namespace {

bool validHeader(uint32_t structSize, uint32_t abiVersion, size_t expectedSize)
{
    return structSize >= expectedSize && abiVersion == POR_ABI_VERSION;
}

por_result setError(por_renderer_handle renderer, por_result result, const std::string& message)
{
    if (renderer) {
        renderer->lastError = message;
    }
    return result;
}

por_result fromRendererResult(RendererResult result)
{
    switch (result) {
        case RendererResult::Success: return POR_SUCCESS;
        case RendererResult::Skipped: return POR_SKIPPED;
        case RendererResult::Error: return POR_ERROR_UNKNOWN;
    }
    return POR_ERROR_UNKNOWN;
}

glm::mat4 toMat4(const por_mat4& matrix)
{
    glm::mat4 out(1.0f);
    std::memcpy(&out[0][0], matrix.value, sizeof(matrix.value));
    return out;
}

glm::vec3 toVec3(const por_vec3& value)
{
    return glm::vec3(value.x, value.y, value.z);
}

RenderFramePacket toFramePacket(const por_frame_packet& packet)
{
    RenderFramePacket out{};
    out.camera.view = toMat4(packet.view);
    out.camera.projection = toMat4(packet.projection);
    out.camera.position = toVec3(packet.camera_position);
    out.camera.nearZ = packet.camera_near_z;
    out.camera.farZ = packet.camera_far_z;

    out.light.direction = toVec3(packet.light_direction);
    out.light.color = toVec3(packet.light_color);
    out.light.intensity = packet.light_intensity;
    out.light.ambient = packet.ambient_intensity;

    out.shadow.csmLambda = packet.shadow_csm_lambda;
    out.shadow.maxDistance = packet.shadow_max_distance;
    out.shadow.debugCascades = packet.shadow_debug_cascades != 0u;
    out.shadow.filterMode = packet.shadow_filter_mode;
    out.shadow.pcfSpreadRadius = packet.shadow_pcf_spread_radius;
    out.shadow.vsmBleedReduction = packet.shadow_vsm_bleed_reduction;
    out.shadow.depthBiasConstant = packet.shadow_depth_bias_constant;
    out.shadow.depthBiasSlope = packet.shadow_depth_bias_slope;
    out.shadow.enableCasterCulling = packet.shadow_enable_caster_culling != 0u;
    out.shadow.qualityPreset = packet.shadow_quality_preset;
    out.shadow.cullMode = packet.shadow_cull_mode;

    out.tonemap.mode = packet.tonemap_mode;
    out.tonemap.exposure = packet.tonemap_exposure;
    out.tonemap.splitScreen = packet.tonemap_split_screen != 0u;
    out.tonemap.splitRightMode = packet.tonemap_split_right_mode;

    out.sky.enabled = packet.sky_enabled != 0u;
    out.sky.mode = packet.sky_mode;
    out.debug.wireframe = packet.debug_wireframe != 0u;
    out.debug.showNormals = packet.debug_show_normals != 0u;
    return out;
}

void fillStats(const Renderer::RenderStats& in, por_renderer_stats& out)
{
    out.draw_calls = in.drawCalls;
    out.triangles = in.triangles;
    out.opaque_draw_calls = in.opaqueDrawCalls;
    out.masked_draw_calls = in.maskedDrawCalls;
    out.mesh_count = in.meshCount;
    out.material_count = in.materialCount;
    out.texture_count = in.textureCount;
    out.texture_memory_bytes = static_cast<uint64_t>(in.textureMemoryBytes);
    out.aa_mode = static_cast<uint32_t>(in.aaMode);
    out.active_sample_count = in.activeSampleCount;
    out.sample_shading_enabled = in.sampleShadingEnabled ? 1u : 0u;
    out.min_sample_shading = in.minSampleShading;
    out.alpha_to_coverage_enabled = in.alphaToCoverageEnabled ? 1u : 0u;
    out.estimated_msaa_attachment_memory_bytes =
        static_cast<uint64_t>(in.estimatedMsaaAttachmentMemoryBytes);
    out.scene_gpu_ms = in.sceneGpuMs;
    out.tonemap_gpu_ms = in.tonemapGpuMs;
    out.total_gpu_frame_ms = in.totalGpuFrameMs;
}

} // namespace

extern "C" {

por_result por_renderer_create(const por_renderer_create_info* info,
                               por_renderer_handle* out_renderer)
{
    if (!info || !out_renderer) {
        return POR_ERROR_INVALID_ARGUMENT;
    }
    *out_renderer = nullptr;
    if (!validHeader(info->struct_size, info->abi_version, sizeof(*info))) {
        return POR_ERROR_ABI_MISMATCH;
    }

    try {
        auto renderer = std::make_unique<por_renderer_t>();
        const RendererCreateInfo createInfo{
            .width = info->width,
            .height = info->height,
            .title = info->title_utf8 ? info->title_utf8 : "ProjectOptimizedRenderer",
        };
        const RendererResult result = renderer->instance.create(createInfo);
        if (result != RendererResult::Success) {
            renderer->lastError = renderer->instance.getLastError();
            return fromRendererResult(result);
        }
        *out_renderer = renderer.release();
        return POR_SUCCESS;
    } catch (const std::bad_alloc&) {
        return POR_ERROR_OUT_OF_MEMORY;
    } catch (...) {
        return POR_ERROR_UNKNOWN;
    }
}

void por_renderer_destroy(por_renderer_handle renderer)
{
    delete renderer;
}

por_result por_renderer_resize(por_renderer_handle renderer, uint32_t width, uint32_t height)
{
    if (!renderer || width == 0u || height == 0u) {
        return setError(renderer, POR_ERROR_INVALID_ARGUMENT, "invalid resize arguments");
    }
    const RendererResult result = renderer->instance.resize(width, height);
    if (result != RendererResult::Success) {
        renderer->lastError = renderer->instance.getLastError();
    }
    return fromRendererResult(result);
}

por_result por_renderer_render_frame(por_renderer_handle renderer, const por_frame_packet* packet)
{
    if (!renderer || !packet) {
        return setError(renderer, POR_ERROR_INVALID_ARGUMENT, "invalid frame packet");
    }
    if (!validHeader(packet->struct_size, packet->abi_version, sizeof(*packet))) {
        return setError(renderer, POR_ERROR_ABI_MISMATCH, "frame packet ABI mismatch");
    }

    const RendererResult result = renderer->instance.renderFrame(toFramePacket(*packet));
    if (result != RendererResult::Success && result != RendererResult::Skipped) {
        renderer->lastError = renderer->instance.getLastError();
    }
    return fromRendererResult(result);
}

por_result por_renderer_get_stats(por_renderer_handle renderer, por_renderer_stats* out_stats)
{
    if (!renderer || !out_stats) {
        return setError(renderer, POR_ERROR_INVALID_ARGUMENT, "invalid stats arguments");
    }
    if (!validHeader(out_stats->struct_size, out_stats->abi_version, sizeof(*out_stats))) {
        return setError(renderer, POR_ERROR_ABI_MISMATCH, "stats ABI mismatch");
    }
    const uint32_t structSize = out_stats->struct_size;
    const uint32_t abiVersion = out_stats->abi_version;
    *out_stats = {};
    out_stats->struct_size = structSize;
    out_stats->abi_version = abiVersion;
    fillStats(renderer->instance.getRenderStats(), *out_stats);
    return POR_SUCCESS;
}

const char* por_renderer_get_last_error(por_renderer_handle renderer)
{
    if (!renderer) {
        return "null renderer";
    }
    if (!renderer->lastError.empty()) {
        return renderer->lastError.c_str();
    }
    return renderer->instance.getLastError().c_str();
}

por_result por_renderer_create_mesh(por_renderer_handle renderer,
                                    const por_mesh_desc*,
                                    por_mesh_handle* out_mesh)
{
    if (out_mesh) {
        *out_mesh = POR_INVALID_RESOURCE_HANDLE;
    }
    return setError(renderer, POR_ERROR_UNSUPPORTED_FEATURE,
                    "individual mesh resources are not implemented in the real wrapper yet");
}

por_result por_renderer_destroy_mesh(por_renderer_handle renderer, por_mesh_handle)
{
    return setError(renderer, POR_ERROR_UNSUPPORTED_FEATURE,
                    "individual mesh resources are not implemented in the real wrapper yet");
}

por_result por_renderer_create_material(por_renderer_handle renderer,
                                        const por_material_desc*,
                                        por_material_handle* out_material)
{
    if (out_material) {
        *out_material = POR_INVALID_RESOURCE_HANDLE;
    }
    return setError(renderer, POR_ERROR_UNSUPPORTED_FEATURE,
                    "individual material resources are not implemented in the real wrapper yet");
}

por_result por_renderer_destroy_material(por_renderer_handle renderer, por_material_handle)
{
    return setError(renderer, POR_ERROR_UNSUPPORTED_FEATURE,
                    "individual material resources are not implemented in the real wrapper yet");
}

por_result por_renderer_create_texture(por_renderer_handle renderer,
                                       const por_texture_desc*,
                                       por_texture_handle* out_texture)
{
    if (out_texture) {
        *out_texture = POR_INVALID_RESOURCE_HANDLE;
    }
    return setError(renderer, POR_ERROR_UNSUPPORTED_FEATURE,
                    "individual texture resources are not implemented in the real wrapper yet");
}

por_result por_renderer_destroy_texture(por_renderer_handle renderer, por_texture_handle)
{
    return setError(renderer, POR_ERROR_UNSUPPORTED_FEATURE,
                    "individual texture resources are not implemented in the real wrapper yet");
}

} // extern "C"
