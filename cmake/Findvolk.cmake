# Findvolk.cmake
# Finds the volk Vulkan meta-loader.
#
# Tries Conan/CMake config-mode package first, then falls back to manual search.
#
# Imported targets:
#   volk::volk
#
# Result variables:
#   volk_FOUND
#   volk_INCLUDE_DIR
#   volk_LIBRARY

# Prefer Conan/CMake config-mode package if available
find_package(volk CONFIG QUIET)
if(volk_FOUND)
    return()
endif()

# Build platform-specific search paths
set(_volk_search_paths)
if(APPLE)
    list(APPEND _volk_search_paths
        "${HOMEBREW_PREFIX}/include"
        "${HOMEBREW_PREFIX}/lib"
    )
elseif(WIN32)
    if(DEFINED ENV{VULKAN_SDK})
        list(APPEND _volk_search_paths
            "$ENV{VULKAN_SDK}/Include"
            "$ENV{VULKAN_SDK}/Lib"
        )
    endif()
endif()
list(APPEND _volk_search_paths
    /usr/local/include
    /usr/include
    /usr/local/lib
    /usr/lib
)

find_path(volk_INCLUDE_DIR
    NAMES volk.h
    PATHS ${_volk_search_paths}
    PATH_SUFFIXES volk
)

find_library(volk_LIBRARY
    NAMES volk
    PATHS ${_volk_search_paths}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(volk
    REQUIRED_VARS volk_LIBRARY volk_INCLUDE_DIR
)

if(volk_FOUND AND NOT TARGET volk::volk)
    add_library(volk::volk STATIC IMPORTED)
    set_target_properties(volk::volk PROPERTIES
        IMPORTED_LOCATION             "${volk_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${volk_INCLUDE_DIR}"
        INTERFACE_COMPILE_DEFINITIONS "VK_NO_PROTOTYPES"
    )

    if(APPLE)
        set_property(TARGET volk::volk APPEND PROPERTY
            INTERFACE_COMPILE_DEFINITIONS
                VK_USE_PLATFORM_MACOS_MVK
                VK_ENABLE_BETA_EXTENSIONS
        )
    elseif(WIN32)
        set_property(TARGET volk::volk APPEND PROPERTY
            INTERFACE_COMPILE_DEFINITIONS
                VK_USE_PLATFORM_WIN32_KHR
        )
    endif()
endif()

mark_as_advanced(volk_INCLUDE_DIR volk_LIBRARY)
