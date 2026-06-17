#include "por/por_renderer.h"

#include <stdlib.h>
#include <string.h>

struct por_renderer_t {
    uint32_t width;
    uint32_t height;
    uint32_t frame_count;
    uint32_t next_mesh;
    uint32_t next_material;
    uint32_t next_texture;
    char last_error[256];
    por_renderer_stats stats;
};

static int por_valid_header(uint32_t struct_size, uint32_t abi_version, size_t expected_size)
{
    return struct_size >= expected_size && abi_version == POR_ABI_VERSION;
}

static por_result por_set_error(struct por_renderer_t* renderer, por_result result, const char* message)
{
    if (renderer && message) {
        strncpy(renderer->last_error, message, sizeof(renderer->last_error) - 1u);
        renderer->last_error[sizeof(renderer->last_error) - 1u] = '\0';
    }
    return result;
}

por_result por_renderer_create(const por_renderer_create_info* info,
                               por_renderer_handle* out_renderer)
{
    if (!info || !out_renderer) {
        return POR_ERROR_INVALID_ARGUMENT;
    }
    if (!por_valid_header(info->struct_size, info->abi_version, sizeof(*info))) {
        return POR_ERROR_ABI_MISMATCH;
    }

    struct por_renderer_t* renderer =
        (struct por_renderer_t*)calloc(1u, sizeof(struct por_renderer_t));
    if (!renderer) {
        return POR_ERROR_OUT_OF_MEMORY;
    }

    renderer->width = info->width;
    renderer->height = info->height;
    renderer->next_mesh = 1u;
    renderer->next_material = 1u;
    renderer->next_texture = 1u;
    renderer->stats.struct_size = sizeof(renderer->stats);
    renderer->stats.abi_version = POR_ABI_VERSION;
    renderer->stats.active_sample_count = 1u;
    *out_renderer = renderer;
    return POR_SUCCESS;
}

void por_renderer_destroy(por_renderer_handle renderer)
{
    free(renderer);
}

por_result por_renderer_resize(por_renderer_handle renderer, uint32_t width, uint32_t height)
{
    if (!renderer || width == 0u || height == 0u) {
        return por_set_error(renderer, POR_ERROR_INVALID_ARGUMENT, "invalid resize arguments");
    }
    renderer->width = width;
    renderer->height = height;
    renderer->last_error[0] = '\0';
    return POR_SUCCESS;
}

por_result por_renderer_render_frame(por_renderer_handle renderer, const por_frame_packet* packet)
{
    if (!renderer || !packet) {
        return por_set_error(renderer, POR_ERROR_INVALID_ARGUMENT, "invalid frame packet");
    }
    if (!por_valid_header(packet->struct_size, packet->abi_version, sizeof(*packet))) {
        return por_set_error(renderer, POR_ERROR_ABI_MISMATCH, "frame packet ABI mismatch");
    }
    renderer->frame_count += 1u;
    renderer->stats.draw_calls = renderer->frame_count;
    renderer->stats.triangles = renderer->frame_count * 2u;
    renderer->last_error[0] = '\0';
    return POR_SUCCESS;
}

por_result por_renderer_get_stats(por_renderer_handle renderer, por_renderer_stats* out_stats)
{
    if (!renderer || !out_stats) {
        return por_set_error(renderer, POR_ERROR_INVALID_ARGUMENT, "invalid stats arguments");
    }
    if (out_stats->struct_size < sizeof(*out_stats) ||
        out_stats->abi_version != POR_ABI_VERSION) {
        return por_set_error(renderer, POR_ERROR_ABI_MISMATCH, "stats ABI mismatch");
    }
    *out_stats = renderer->stats;
    out_stats->struct_size = sizeof(*out_stats);
    out_stats->abi_version = POR_ABI_VERSION;
    renderer->last_error[0] = '\0';
    return POR_SUCCESS;
}

const char* por_renderer_get_last_error(por_renderer_handle renderer)
{
    return renderer ? renderer->last_error : "null renderer";
}

por_result por_renderer_create_mesh(por_renderer_handle renderer,
                                    const por_mesh_desc* desc,
                                    por_mesh_handle* out_mesh)
{
    if (!renderer || !desc || !out_mesh) {
        return por_set_error(renderer, POR_ERROR_INVALID_ARGUMENT, "invalid mesh arguments");
    }
    if (!por_valid_header(desc->struct_size, desc->abi_version, sizeof(*desc))) {
        return por_set_error(renderer, POR_ERROR_ABI_MISMATCH, "mesh descriptor ABI mismatch");
    }
    *out_mesh = renderer->next_mesh++;
    renderer->stats.mesh_count += 1u;
    renderer->last_error[0] = '\0';
    return POR_SUCCESS;
}

por_result por_renderer_destroy_mesh(por_renderer_handle renderer, por_mesh_handle mesh)
{
    if (!renderer || mesh == POR_INVALID_RESOURCE_HANDLE) {
        return por_set_error(renderer, POR_ERROR_INVALID_ARGUMENT, "invalid mesh handle");
    }
    renderer->last_error[0] = '\0';
    return POR_SUCCESS;
}

por_result por_renderer_create_material(por_renderer_handle renderer,
                                        const por_material_desc* desc,
                                        por_material_handle* out_material)
{
    if (!renderer || !desc || !out_material) {
        return por_set_error(renderer, POR_ERROR_INVALID_ARGUMENT, "invalid material arguments");
    }
    if (!por_valid_header(desc->struct_size, desc->abi_version, sizeof(*desc))) {
        return por_set_error(renderer, POR_ERROR_ABI_MISMATCH, "material descriptor ABI mismatch");
    }
    *out_material = renderer->next_material++;
    renderer->stats.material_count += 1u;
    renderer->last_error[0] = '\0';
    return POR_SUCCESS;
}

por_result por_renderer_destroy_material(por_renderer_handle renderer, por_material_handle material)
{
    if (!renderer || material == POR_INVALID_RESOURCE_HANDLE) {
        return por_set_error(renderer, POR_ERROR_INVALID_ARGUMENT, "invalid material handle");
    }
    renderer->last_error[0] = '\0';
    return POR_SUCCESS;
}

por_result por_renderer_create_texture(por_renderer_handle renderer,
                                       const por_texture_desc* desc,
                                       por_texture_handle* out_texture)
{
    if (!renderer || !desc || !out_texture) {
        return por_set_error(renderer, POR_ERROR_INVALID_ARGUMENT, "invalid texture arguments");
    }
    if (!por_valid_header(desc->struct_size, desc->abi_version, sizeof(*desc))) {
        return por_set_error(renderer, POR_ERROR_ABI_MISMATCH, "texture descriptor ABI mismatch");
    }
    *out_texture = renderer->next_texture++;
    renderer->stats.texture_count += 1u;
    renderer->last_error[0] = '\0';
    return POR_SUCCESS;
}

por_result por_renderer_destroy_texture(por_renderer_handle renderer, por_texture_handle texture)
{
    if (!renderer || texture == POR_INVALID_RESOURCE_HANDLE) {
        return por_set_error(renderer, POR_ERROR_INVALID_ARGUMENT, "invalid texture handle");
    }
    renderer->last_error[0] = '\0';
    return POR_SUCCESS;
}
