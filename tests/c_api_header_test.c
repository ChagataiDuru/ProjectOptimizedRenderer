#include "por/por_renderer.h"

#include <stddef.h>

int main(void)
{
    por_renderer_handle renderer = 0;
    por_renderer_create_info create_info = {
        sizeof(por_renderer_create_info),
        POR_ABI_VERSION,
        640u,
        360u,
        "POR C header test",
        { 0u, 0u, 0u, 0u },
    };

    if (por_renderer_create(&create_info, &renderer) != POR_SUCCESS) {
        return 1;
    }

    por_frame_packet frame = { 0 };
    frame.struct_size = sizeof(frame);
    frame.abi_version = POR_ABI_VERSION;
    frame.camera_near_z = 0.01f;
    frame.camera_far_z = 1000.0f;
    frame.light_intensity = 1.0f;
    frame.ambient_intensity = 0.1f;
    frame.sky_enabled = 1u;

    if (por_renderer_render_frame(renderer, &frame) != POR_SUCCESS) {
        por_renderer_destroy(renderer);
        return 2;
    }

    por_renderer_stats stats = { 0 };
    stats.struct_size = sizeof(stats);
    stats.abi_version = POR_ABI_VERSION;
    if (por_renderer_get_stats(renderer, &stats) != POR_SUCCESS) {
        por_renderer_destroy(renderer);
        return 3;
    }

    por_renderer_destroy(renderer);
    return stats.draw_calls == 1u ? 0 : 4;
}
