#include "por/por_renderer.h"

#include <cstdint>
#include <type_traits>

static_assert(POR_ABI_VERSION == 1u);
static_assert(std::is_standard_layout_v<por_frame_packet>);
static_assert(std::is_standard_layout_v<por_renderer_stats>);
static_assert(sizeof(por_mesh_handle) == sizeof(std::uint32_t));

int main()
{
    por_renderer_handle renderer = nullptr;
    por_renderer_create_info createInfo{};
    createInfo.struct_size = sizeof(createInfo);
    createInfo.abi_version = POR_ABI_VERSION;
    createInfo.width = 320u;
    createInfo.height = 180u;
    createInfo.title_utf8 = "POR C++ header test";

    if (por_renderer_create(&createInfo, &renderer) != POR_SUCCESS) {
        return 1;
    }

    por_mesh_desc meshDesc{};
    meshDesc.struct_size = sizeof(meshDesc);
    meshDesc.abi_version = POR_ABI_VERSION;
    por_mesh_handle mesh = POR_INVALID_RESOURCE_HANDLE;
    if (por_renderer_create_mesh(renderer, &meshDesc, &mesh) != POR_SUCCESS) {
        por_renderer_destroy(renderer);
        return 2;
    }

    por_renderer_destroy(renderer);
    return mesh != POR_INVALID_RESOURCE_HANDLE ? 0 : 3;
}
