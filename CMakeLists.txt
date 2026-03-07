cmake_minimum_required(VERSION 3.25)
project(ProjectOptimizedRenderer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Apple Silicon detection ───────────────────────────────────────────────────
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(APPLE_SILICON TRUE)
    set(HOMEBREW_PREFIX "/opt/homebrew")
else()
    set(APPLE_SILICON FALSE)
    set(HOMEBREW_PREFIX "/usr/local")
endif()

list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_PREFIX}")

# ── Conan 2 toolchain ─────────────────────────────────────────────────────────
include(${CMAKE_BINARY_DIR}/generators/conan_toolchain.cmake)

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
include(cmake/CheckMoltenVK.cmake)

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
)

target_include_directories(ProjectOptimizedRenderer PRIVATE
    "${CMAKE_SOURCE_DIR}/include"
    "${VMA_INCLUDE_DIR}"
)

target_compile_options(ProjectOptimizedRenderer PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    $<$<BOOL:${APPLE_SILICON}>:-march=native -mtune=apple-m2>
)

target_compile_definitions(ProjectOptimizedRenderer PRIVATE
    VK_ENABLE_BETA_EXTENSIONS
    VK_USE_PLATFORM_MACOS_MVK
    $<$<CONFIG:Debug>:VK_DEBUG_MARKERS>
)

target_link_libraries(ProjectOptimizedRenderer PRIVATE
    Vulkan::Vulkan
    volk::volk
    SDL3::SDL3
    spdlog::spdlog
    glm::glm
    Threads::Threads
)
