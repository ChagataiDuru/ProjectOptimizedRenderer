cmake_minimum_required(VERSION 3.25)
project(ProjectOptimizedRenderer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Platform detection ───────────────────────────────────────────────────────
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(APPLE_SILICON TRUE)
    set(HOMEBREW_PREFIX "/opt/homebrew")
elseif(APPLE)
    set(APPLE_SILICON FALSE)
    set(HOMEBREW_PREFIX "/usr/local")
else()
    set(APPLE_SILICON FALSE)
endif()

if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}")
endif()

if(WIN32)
    # Vulkan SDK path — auto-detect from env or allow override
    if(DEFINED ENV{VULKAN_SDK})
        set(VULKAN_SDK_PATH "$ENV{VULKAN_SDK}")
    else()
        # Common default install locations
        file(GLOB _vk_sdk_candidates "C:/VulkanSDK/*")
        if(_vk_sdk_candidates)
            list(SORT _vk_sdk_candidates ORDER DESCENDING)
            list(GET _vk_sdk_candidates 0 VULKAN_SDK_PATH)
        endif()
    endif()

    if(VULKAN_SDK_PATH)
        list(APPEND CMAKE_PREFIX_PATH "${VULKAN_SDK_PATH}")
        message(STATUS "Vulkan SDK: ${VULKAN_SDK_PATH}")
    endif()
endif()

# ── Vulkan 1.4 ────────────────────────────────────────────────────────────────
find_package(Vulkan 1.4 REQUIRED)
find_package(Threads REQUIRED)

# ── Conan dependencies ────────────────────────────────────────────────────────
find_package(glm CONFIG REQUIRED)
find_package(SDL3 CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(glslang CONFIG REQUIRED)
find_package(SPIRV-Tools CONFIG REQUIRED)

# ── Custom modules ────────────────────────────────────────────────────────────
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(cmake/Findvolk.cmake)
include(cmake/CompileShaders.cmake)

# MoltenVK check is Apple-only (the module itself early-returns on non-Apple)
if(APPLE)
    include(cmake/CheckMoltenVK.cmake)
endif()

find_package(volk REQUIRED)

# ── VMA (header-only) ─────────────────────────────────────────────────────────
set(VMA_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/VulkanMemoryAllocator/include")
if(NOT EXISTS "${VMA_INCLUDE_DIR}/vk_mem_alloc.h")
    message(FATAL_ERROR "VulkanMemoryAllocator not found at ${VMA_INCLUDE_DIR}.\n"
        "Run: git submodule update --init --recursive")
endif()

# ── Target ────────────────────────────────────────────────────────────────────
add_executable(ProjectOptimizedRenderer
    src/main.cpp
    src/core/VulkanContext.cpp
    src/core/Device.cpp
    src/core/Window.cpp
    src/core/Swapchain.cpp
    src/core/CommandBuffer.cpp
    src/core/FrameSync.cpp
    src/core/Renderer.cpp
    src/core/Camera.cpp
    src/resource/VmaImpl.cpp
    src/resource/Buffer.cpp
    src/resource/Image.cpp
    src/resource/Descriptor.cpp
    src/resource/PipelineLayout.cpp
    src/resource/Shader.cpp
    src/resource/ResourceSet.cpp
)

target_include_directories(ProjectOptimizedRenderer PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${VMA_INCLUDE_DIR}"
)

# ── Compiler flags (platform-specific) ────────────────────────────────────────
if(MSVC)
    target_compile_options(ProjectOptimizedRenderer PRIVATE
        /W4
        /permissive-
        /Zc:__cplusplus        # Report correct C++ standard version
        /Zc:preprocessor       # Use conformant preprocessor
        /utf-8                 # Source and execution character set
        /EHsc                  # Standard C++ exception handling
        # Suppress warnings that fire excessively in Vulkan/VMA headers
        /wd4324                # structure was padded due to alignment specifier
        /wd4127                # conditional expression is constant (VMA macros)
        /wd4100                # unreferenced formal parameter
        /wd4189                # local variable is initialized but not referenced
    )
else()
    target_compile_options(ProjectOptimizedRenderer PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        $<$<BOOL:${APPLE_SILICON}>:-march=native -mtune=apple-m2>
    )
endif()

# ── Compile definitions (platform-specific) ───────────────────────────────────
target_compile_definitions(ProjectOptimizedRenderer PRIVATE
    VK_ENABLE_BETA_EXTENSIONS
    $<$<CONFIG:Debug>:VK_DEBUG_MARKERS>
    SHADER_DIR="${CMAKE_BINARY_DIR}/shaders"
    # VMA: use dynamic Vulkan function loading via volk
    VMA_STATIC_VULKAN_FUNCTIONS=0
    VMA_DYNAMIC_VULKAN_FUNCTIONS=1
)

if(APPLE)
    target_compile_definitions(ProjectOptimizedRenderer PRIVATE
        VK_USE_PLATFORM_MACOS_MVK
    )
elseif(WIN32)
    target_compile_definitions(ProjectOptimizedRenderer PRIVATE
        VK_USE_PLATFORM_WIN32_KHR
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
endif()

# ── Shader compilation ────────────────────────────────────────────────────────
shader_compile_glsl(
    TARGET ProjectOptimizedRenderer
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/triangle.vert"
    OUTPUT "triangle.vert.spv"
)
shader_compile_glsl(
    TARGET ProjectOptimizedRenderer
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/triangle.frag"
    OUTPUT "triangle.frag.spv"
)
shader_compile_glsl(
    TARGET ProjectOptimizedRenderer
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/pbr.vert"
    OUTPUT "pbr.vert.spv"
)
shader_compile_glsl(
    TARGET ProjectOptimizedRenderer
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/pbr.frag"
    OUTPUT "pbr.frag.spv"
)

target_link_libraries(ProjectOptimizedRenderer PRIVATE
    volk::volk
    SDL3::SDL3
    spdlog::spdlog
    glm::glm
    Threads::Threads
    glslang::glslang
    glslang::SPIRV
    glslang::glslang-default-resource-limits
)
