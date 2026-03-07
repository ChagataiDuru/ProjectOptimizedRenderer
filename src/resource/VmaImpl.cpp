// Single translation unit that defines the VMA implementation.
// VMA_STATIC_VULKAN_FUNCTIONS=0 and VMA_DYNAMIC_VULKAN_FUNCTIONS=1 are set
// globally via CMake target_compile_definitions, so all TUs see a consistent ABI.
#define VMA_IMPLEMENTATION
#include <volk.h>
#include <vk_mem_alloc.h>
