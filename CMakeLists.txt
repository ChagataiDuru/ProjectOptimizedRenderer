cmake_minimum_required(VERSION 3.25)
project(ProjectOptimizedRenderer VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Platform detection ───────────────────────────────────────────────────────
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(APPLE_SILICON TRUE)
    set(HOMEBREW_PREFIX "/opt/homebrew")

    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
    set(CMAKE_BUILD_RPATH_USE_ORIGIN ON)
    # Embed all linked library directories into the binary's RPATH at build time
    set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)
    set(CMAKE_SKIP_BUILD_RPATH OFF)

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

# ── Targets ───────────────────────────────────────────────────────────────────
set(POR_RENDERER_CORE_SOURCES
    src/core/VulkanContext.cpp
    src/core/Device.cpp
    src/core/Window.cpp
    src/core/Swapchain.cpp
    src/core/CommandBuffer.cpp
    src/core/FrameSync.cpp
    src/core/Renderer.cpp
    src/core/RendererInstance.cpp
    src/core/VulkanUtil.cpp
    src/renderpasses/SkyPass.cpp
    src/renderpasses/TonemapPass.cpp
    src/renderpasses/ShadowPass.cpp
    src/renderpasses/ClusteredLightCullingPass.cpp
    src/resource/VmaImpl.cpp
    src/resource/Buffer.cpp
    src/resource/Image.cpp
    src/resource/Descriptor.cpp
    src/resource/PipelineLayout.cpp
    src/resource/Shader.cpp
    src/resource/ResourceSet.cpp
    src/resource/GLTFLoader.cpp
    src/resource/StbImageImpl.cpp
    src/resource/Texture.cpp
    src/resource/SamplerCache.cpp
    src/resource/SceneInfo.cpp
    src/resource/RendererResourceManager.cpp
    src/debug/GPUTimer.cpp
    src/debug/Screenshot.cpp
)

set(POR_VIEWER_SOURCES
    src/main.cpp
    src/core/Camera.cpp
    src/debug/ImGuiManager.cpp
    src/debug/LogSink.cpp
    src/viewer/ViewerPanels.cpp
    external/imgui/imgui.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/backends/imgui_impl_sdl3.cpp
    external/imgui/backends/imgui_impl_vulkan.cpp
)

add_library(por_renderer_core STATIC ${POR_RENDERER_CORE_SOURCES})
add_library(por_renderer_c_api STATIC src/api/por_renderer.cpp)
add_library(por_renderer_api_mock STATIC src/api/por_renderer_mock.c)
add_executable(por_viewer ${POR_VIEWER_SOURCES})

set_target_properties(por_viewer PROPERTIES OUTPUT_NAME "ProjectOptimizedRenderer")
set_target_properties(por_renderer_api_mock PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)

target_include_directories(por_renderer_core PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    "${VMA_INCLUDE_DIR}"
    "${CMAKE_SOURCE_DIR}/external/cgltf"
    "${CMAKE_SOURCE_DIR}/external/stb"
)

target_include_directories(por_viewer PRIVATE
    "${CMAKE_SOURCE_DIR}/external/imgui"
    "${CMAKE_SOURCE_DIR}/external/imgui/backends"
)

target_include_directories(por_renderer_api_mock PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
)

target_link_libraries(por_renderer_core PUBLIC
    volk::volk
    SDL3::SDL3
    spdlog::spdlog
    glm::glm
    Threads::Threads
    glslang::glslang
    glslang::SPIRV
    glslang::glslang-default-resource-limits
)

target_link_libraries(por_renderer_c_api PUBLIC por_renderer_core)
target_link_libraries(por_viewer PRIVATE por_renderer_core)

set(POR_PROJECT_TARGETS
    por_renderer_core
    por_renderer_c_api
    por_viewer
)

# ── Compiler flags (platform-specific) ────────────────────────────────────────
if(MSVC)
    foreach(target IN LISTS POR_PROJECT_TARGETS)
        target_compile_options(${target} PRIVATE
            /W4
            /permissive-
            /Zc:__cplusplus
            /Zc:preprocessor
            /utf-8
            /EHsc
            /wd4324
            /wd4127
            /wd4100
            /wd4189
        )
    endforeach()
else()
    foreach(target IN LISTS POR_PROJECT_TARGETS)
        target_compile_options(${target} PRIVATE
            -Wall
            -Wextra
            -Wpedantic
            $<$<BOOL:${APPLE_SILICON}>:-march=native -mtune=apple-m2>
        )
    endforeach()
endif()

# ── Compile definitions (platform-specific) ───────────────────────────────────
foreach(target IN LISTS POR_PROJECT_TARGETS)
    target_compile_definitions(${target} PRIVATE
        VK_ENABLE_BETA_EXTENSIONS
        $<$<CONFIG:Debug>:VK_DEBUG_MARKERS>
        SHADER_DIR="${CMAKE_BINARY_DIR}/shaders"
        ASSET_DIR="${CMAKE_SOURCE_DIR}/assets"
        VMA_STATIC_VULKAN_FUNCTIONS=0
        VMA_DYNAMIC_VULKAN_FUNCTIONS=1
    )
endforeach()

target_compile_definitions(por_viewer PRIVATE IMGUI_IMPL_VULKAN_USE_VOLK)

# ── Suppress warnings in third-party ImGui sources ────────────────────────────
set(IMGUI_SOURCES
    "${CMAKE_SOURCE_DIR}/external/imgui/imgui.cpp"
    "${CMAKE_SOURCE_DIR}/external/imgui/imgui_draw.cpp"
    "${CMAKE_SOURCE_DIR}/external/imgui/imgui_tables.cpp"
    "${CMAKE_SOURCE_DIR}/external/imgui/imgui_widgets.cpp"
    "${CMAKE_SOURCE_DIR}/external/imgui/backends/imgui_impl_sdl3.cpp"
    "${CMAKE_SOURCE_DIR}/external/imgui/backends/imgui_impl_vulkan.cpp"
)
if(MSVC)
    set_source_files_properties(${IMGUI_SOURCES} PROPERTIES COMPILE_FLAGS "/w")
else()
    set_source_files_properties(${IMGUI_SOURCES} PROPERTIES COMPILE_FLAGS "-w")
endif()

if(APPLE)
    find_library(POR_VULKAN_LOADER_LIBRARY
        NAMES vulkan.1 vulkan
        PATHS
            "${HOMEBREW_PREFIX}/lib"
            "$ENV{VULKAN_SDK}/lib"
        NO_DEFAULT_PATH
    )
    find_file(POR_MOLTENVK_ICD_JSON
        NAMES MoltenVK_icd.json
        PATHS
            "${HOMEBREW_PREFIX}/etc/vulkan/icd.d"
            "$ENV{VULKAN_SDK}/share/vulkan/icd.d"
        NO_DEFAULT_PATH
    )

    foreach(target IN LISTS POR_PROJECT_TARGETS)
        target_compile_definitions(${target} PRIVATE
            VK_USE_PLATFORM_MACOS_MVK
        )
    endforeach()
    if(POR_VULKAN_LOADER_LIBRARY)
        foreach(target IN LISTS POR_PROJECT_TARGETS)
            target_compile_definitions(${target} PRIVATE
                POR_VULKAN_LOADER_PATH="${POR_VULKAN_LOADER_LIBRARY}"
            )
        endforeach()
    endif()
    if(POR_MOLTENVK_ICD_JSON)
        foreach(target IN LISTS POR_PROJECT_TARGETS)
            target_compile_definitions(${target} PRIVATE
                POR_MOLTENVK_ICD_JSON_PATH="${POR_MOLTENVK_ICD_JSON}"
            )
        endforeach()
    endif()
    if(MOLTEN_VK_LIBRARY)
        foreach(target IN LISTS POR_PROJECT_TARGETS)
            target_compile_definitions(${target} PRIVATE
                POR_MOLTENVK_LIBRARY_PATH="${MOLTEN_VK_LIBRARY}"
            )
        endforeach()
    endif()
elseif(WIN32)
    foreach(target IN LISTS POR_PROJECT_TARGETS)
        target_compile_definitions(${target} PRIVATE
            VK_USE_PLATFORM_WIN32_KHR
            WIN32_LEAN_AND_MEAN
            NOMINMAX
        )
    endforeach()
endif()

# ── Shader compilation ────────────────────────────────────────────────────────
add_custom_target(por_shaders)

shader_compile_glsl(
    TARGET por_shaders
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/triangle.vert"
    OUTPUT "triangle.vert.spv"
)
shader_compile_glsl(
    TARGET por_shaders
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/triangle.frag"
    OUTPUT "triangle.frag.spv"
)
shader_compile_glsl(
    TARGET por_shaders
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/pbr.vert"
    OUTPUT "pbr.vert.spv"
)
shader_compile_glsl(
    TARGET por_shaders
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/pbr.frag"
    OUTPUT "pbr.frag.spv"
)
shader_compile_glsl(
    TARGET por_shaders
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/pbr_normals.frag"
    OUTPUT "pbr_normals.frag.spv"
)
shader_compile_glsl(
    TARGET por_shaders
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/shadow.vert"
    OUTPUT "shadow.vert.spv"
)
shader_compile_glsl(
    TARGET por_shaders
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/shadow_alpha.vert"
    OUTPUT "shadow_alpha.vert.spv"
)
shader_compile_glsl(
    TARGET por_shaders
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/shadow_alpha.frag"
    OUTPUT "shadow_alpha.frag.spv"
)
shader_compile_glsl(
    TARGET por_shaders
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/shadow_vsm.frag"
    OUTPUT "shadow_vsm.frag.spv"
)
shader_compile_glsl(
    TARGET por_shaders
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/shadow_vsm_alpha.frag"
    OUTPUT "shadow_vsm_alpha.frag.spv"
)
shader_compile_glsl(
    TARGET por_shaders
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/shadow_blur.comp"
    OUTPUT "shadow_blur.comp.spv"
)
shader_compile_glsl(
    TARGET por_shaders
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/cluster_cull.comp"
    OUTPUT "cluster_cull.comp.spv"
)
shader_compile_glsl(
    TARGET por_shaders
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/tonemap.vert"
    OUTPUT "tonemap.vert.spv"
)
shader_compile_glsl(
    TARGET por_shaders
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/tonemap.frag"
    OUTPUT "tonemap.frag.spv"
)
shader_compile_glsl(
    TARGET por_shaders
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/sky.vert"
    OUTPUT "sky.vert.spv"
)
shader_compile_glsl(
    TARGET por_shaders
    SOURCE "${CMAKE_SOURCE_DIR}/shaders/sky.frag"
    OUTPUT "sky.frag.spv"
)

add_dependencies(por_viewer por_shaders)

# ── ABI header compile tests ──────────────────────────────────────────────────
enable_testing()

add_executable(por_c_api_header_test tests/c_api_header_test.c)
target_link_libraries(por_c_api_header_test PRIVATE por_renderer_api_mock)
target_include_directories(por_c_api_header_test PRIVATE "${CMAKE_SOURCE_DIR}/include")
set_target_properties(por_c_api_header_test PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
add_test(NAME por_c_api_header_test COMMAND por_c_api_header_test)

add_executable(por_cpp_api_header_test tests/c_api_header_test.cpp)
target_link_libraries(por_cpp_api_header_test PRIVATE por_renderer_api_mock)
target_include_directories(por_cpp_api_header_test PRIVATE "${CMAKE_SOURCE_DIR}/include")
set_target_properties(por_cpp_api_header_test PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
add_test(NAME por_cpp_api_header_test COMMAND por_cpp_api_header_test)
