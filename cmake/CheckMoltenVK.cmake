# CheckMoltenVK.cmake
# Verifies MoltenVK >= 1.4.0 is available on the system.
#
# Sets in parent scope:
#   MOLTEN_VK_FOUND        - TRUE if MoltenVK is found and version is sufficient
#   MOLTEN_VK_VERSION      - Full version string (e.g. "1.4.1")
#   MOLTEN_VK_INCLUDE_DIR  - Path to MoltenVK headers
#   MOLTEN_VK_LIBRARY      - Path to libMoltenVK.dylib

if(NOT APPLE)
    return()
endif()

set(_mvk_min_major 1)
set(_mvk_min_minor 4)
set(_mvk_min_patch 0)

# ── Locate headers ────────────────────────────────────────────────────────────
find_path(_mvk_include_dir
    NAMES MoltenVK/vk_mvk_moltenvk.h
    PATHS
        "${HOMEBREW_PREFIX}/opt/molten-vk/include"
        "$ENV{VULKAN_SDK}/include"
        /usr/local/include
    NO_DEFAULT_PATH
)

if(NOT _mvk_include_dir)
    message(WARNING
        "CheckMoltenVK: MoltenVK headers not found.\n"
        "  Install via: brew install molten-vk\n"
        "  Or set VULKAN_SDK to your SDK path.")
    set(MOLTEN_VK_FOUND FALSE PARENT_SCOPE)
    return()
endif()

# ── Extract version from header ───────────────────────────────────────────────
set(_mvk_header "${_mvk_include_dir}/MoltenVK/vk_mvk_moltenvk.h")

file(STRINGS "${_mvk_header}" _mvk_major_line
    REGEX "#define MVK_VERSION_MAJOR[ \t]+[0-9]+")
file(STRINGS "${_mvk_header}" _mvk_minor_line
    REGEX "#define MVK_VERSION_MINOR[ \t]+[0-9]+")
file(STRINGS "${_mvk_header}" _mvk_patch_line
    REGEX "#define MVK_VERSION_PATCH[ \t]+[0-9]+")

string(REGEX REPLACE ".*MVK_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" _mvk_major "${_mvk_major_line}")
string(REGEX REPLACE ".*MVK_VERSION_MINOR[ \t]+([0-9]+).*" "\\1" _mvk_minor "${_mvk_minor_line}")
string(REGEX REPLACE ".*MVK_VERSION_PATCH[ \t]+([0-9]+).*" "\\1" _mvk_patch "${_mvk_patch_line}")

set(_mvk_version "${_mvk_major}.${_mvk_minor}.${_mvk_patch}")

# ── Version comparison ────────────────────────────────────────────────────────
if(_mvk_major LESS _mvk_min_major
    OR (_mvk_major EQUAL _mvk_min_major AND _mvk_minor LESS _mvk_min_minor)
    OR (_mvk_major EQUAL _mvk_min_major AND _mvk_minor EQUAL _mvk_min_minor
        AND _mvk_patch LESS _mvk_min_patch))
    message(FATAL_ERROR
        "CheckMoltenVK: Found MoltenVK ${_mvk_version} but >= "
        "${_mvk_min_major}.${_mvk_min_minor}.${_mvk_min_patch} is required "
        "(for VK_KHR_dynamic_rendering_local_read / SPIR-V 1.6 support).\n"
        "  Upgrade via: brew upgrade molten-vk")
endif()

# ── Locate library ────────────────────────────────────────────────────────────
find_library(_mvk_library
    NAMES MoltenVK
    PATHS
        "${HOMEBREW_PREFIX}/opt/molten-vk/lib"
        "$ENV{VULKAN_SDK}/lib"
    NO_DEFAULT_PATH
)

if(NOT _mvk_library)
    message(WARNING "CheckMoltenVK: Headers found but libMoltenVK.dylib not located.")
endif()

# ── Publish results ───────────────────────────────────────────────────────────
set(MOLTEN_VK_FOUND        TRUE)
set(MOLTEN_VK_VERSION      "${_mvk_version}")
set(MOLTEN_VK_INCLUDE_DIR  "${_mvk_include_dir}")
set(MOLTEN_VK_LIBRARY      "${_mvk_library}")

message(STATUS "MoltenVK ${_mvk_version} found: ${_mvk_library}")
