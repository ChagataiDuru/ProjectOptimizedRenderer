#ifndef POR_RENDERER_H
#define POR_RENDERER_H

#include <stddef.h>
#include <stdint.h>

#define POR_ABI_VERSION 1u
#define POR_INVALID_RESOURCE_HANDLE UINT32_MAX

#if defined(_WIN32)
    #if defined(POR_BUILD_SHARED)
        #define POR_API __declspec(dllexport)
    #elif defined(POR_USE_SHARED)
        #define POR_API __declspec(dllimport)
    #else
        #define POR_API
    #endif
    #define POR_CALL __cdecl
#else
    #if defined(POR_BUILD_SHARED)
        #define POR_API __attribute__((visibility("default")))
    #else
        #define POR_API
    #endif
    #define POR_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct por_renderer_t* por_renderer_handle;
typedef uint32_t por_mesh_handle;
typedef uint32_t por_material_handle;
typedef uint32_t por_texture_handle;

typedef enum por_result {
    POR_SUCCESS = 0,
    POR_SKIPPED = 1,
    POR_ERROR_UNKNOWN = 2,
    POR_ERROR_INVALID_ARGUMENT = 3,
    POR_ERROR_UNSUPPORTED_FEATURE = 4,
    POR_ERROR_DEVICE_LOST = 5,
    POR_ERROR_RESOURCE_NOT_FOUND = 6,
    POR_ERROR_OUT_OF_MEMORY = 7,
    POR_ERROR_ABI_MISMATCH = 8,
} por_result;

typedef struct por_vec3 {
    float x;
    float y;
    float z;
} por_vec3;

typedef struct por_vec4 {
    float x;
    float y;
    float z;
    float w;
} por_vec4;

typedef struct por_mat4 {
    /* Column-major 4x4 matrix, matching the renderer's internal convention. */
    float value[16];
} por_mat4;

typedef struct por_renderer_create_info {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t width;
    uint32_t height;
    const char* title_utf8;
    uint32_t reserved[4];
} por_renderer_create_info;

typedef struct por_frame_packet {
    uint32_t struct_size;
    uint32_t abi_version;

    por_mat4 view;
    por_mat4 projection;
    por_vec3 camera_position;
    float camera_near_z;
    float camera_far_z;

    por_vec3 light_direction;
    float light_intensity;
    por_vec3 light_color;
    float ambient_intensity;

    float shadow_csm_lambda;
    float shadow_max_distance;
    uint32_t shadow_debug_cascades;
    int32_t shadow_filter_mode;
    float shadow_pcf_spread_radius;
    float shadow_vsm_bleed_reduction;
    float shadow_depth_bias_constant;
    float shadow_depth_bias_slope;
    uint32_t shadow_enable_caster_culling;
    int32_t shadow_quality_preset;
    int32_t shadow_cull_mode;

    int32_t tonemap_mode;
    float tonemap_exposure;
    uint32_t tonemap_split_screen;
    int32_t tonemap_split_right_mode;

    uint32_t sky_enabled;
    int32_t sky_mode;
    uint32_t debug_wireframe;
    uint32_t debug_show_normals;

    uint32_t reserved[8];
} por_frame_packet;

typedef struct por_renderer_stats {
    uint32_t struct_size;
    uint32_t abi_version;

    uint32_t draw_calls;
    uint32_t triangles;
    uint32_t opaque_draw_calls;
    uint32_t masked_draw_calls;
    uint32_t mesh_count;
    uint32_t material_count;
    uint32_t texture_count;
    uint64_t texture_memory_bytes;
    uint32_t aa_mode;
    uint32_t active_sample_count;
    uint32_t sample_shading_enabled;
    float min_sample_shading;
    uint32_t alpha_to_coverage_enabled;
    uint64_t estimated_msaa_attachment_memory_bytes;
    float scene_gpu_ms;
    float tonemap_gpu_ms;
    float total_gpu_frame_ms;

    uint32_t reserved[8];
} por_renderer_stats;

typedef struct por_vertex {
    por_vec3 position;
    por_vec3 normal;
    float uv[2];
    por_vec4 tangent;
} por_vertex;

typedef struct por_mesh_desc {
    uint32_t struct_size;
    uint32_t abi_version;
    const por_vertex* vertices;
    uint32_t vertex_count;
    const uint32_t* indices;
    uint32_t index_count;
    por_vec3 bounds_min;
    por_vec3 bounds_max;
    uint32_t reserved[4];
} por_mesh_desc;

typedef struct por_material_desc {
    uint32_t struct_size;
    uint32_t abi_version;
    por_vec4 base_color_factor;
    float metallic_factor;
    float roughness_factor;
    uint32_t alpha_mode;
    float alpha_cutoff;
    uint32_t double_sided;
    uint32_t reserved[4];
} por_material_desc;

typedef struct por_texture_desc {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t width;
    uint32_t height;
    uint32_t format;
    const void* rgba8_pixels;
    size_t rgba8_byte_count;
    uint32_t reserved[4];
} por_texture_desc;

POR_API por_result POR_CALL por_renderer_create(
    const por_renderer_create_info* info,
    por_renderer_handle* out_renderer);

POR_API void POR_CALL por_renderer_destroy(por_renderer_handle renderer);

POR_API por_result POR_CALL por_renderer_resize(
    por_renderer_handle renderer,
    uint32_t width,
    uint32_t height);

POR_API por_result POR_CALL por_renderer_render_frame(
    por_renderer_handle renderer,
    const por_frame_packet* packet);

POR_API por_result POR_CALL por_renderer_get_stats(
    por_renderer_handle renderer,
    por_renderer_stats* out_stats);

POR_API const char* POR_CALL por_renderer_get_last_error(
    por_renderer_handle renderer);

POR_API por_result POR_CALL por_renderer_create_mesh(
    por_renderer_handle renderer,
    const por_mesh_desc* desc,
    por_mesh_handle* out_mesh);

POR_API por_result POR_CALL por_renderer_destroy_mesh(
    por_renderer_handle renderer,
    por_mesh_handle mesh);

POR_API por_result POR_CALL por_renderer_create_material(
    por_renderer_handle renderer,
    const por_material_desc* desc,
    por_material_handle* out_material);

POR_API por_result POR_CALL por_renderer_destroy_material(
    por_renderer_handle renderer,
    por_material_handle material);

POR_API por_result POR_CALL por_renderer_create_texture(
    por_renderer_handle renderer,
    const por_texture_desc* desc,
    por_texture_handle* out_texture);

POR_API por_result POR_CALL por_renderer_destroy_texture(
    por_renderer_handle renderer,
    por_texture_handle texture);

#ifdef __cplusplus
}
#endif

#endif /* POR_RENDERER_H */
