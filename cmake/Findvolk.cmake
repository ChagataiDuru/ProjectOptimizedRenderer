# Findvolk.cmake
# Finds the volk Vulkan meta-loader.
#
# Tries Conan-generated config first, then falls back to manual search.
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

find_path(volk_INCLUDE_DIR
    NAMES volk.h
    PATHS
        "${HOMEBREW_PREFIX}/include"
        /usr/local/include
        /usr/include
    PATH_SUFFIXES volk
)

find_library(volk_LIBRARY
    NAMES volk
    PATHS
        "${HOMEBREW_PREFIX}/lib"
        /usr/local/lib
        /usr/lib
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
    endif()
endif()

mark_as_advanced(volk_INCLUDE_DIR volk_LIBRARY)
