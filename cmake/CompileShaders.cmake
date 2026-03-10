# CompileShaders.cmake
# Provides shader_compile_glsl() for GLSL -> SPIR-V compilation.
# Targets Vulkan 1.4 / SPIR-V 1.6.

if(TARGET glslang::glslang)
    get_target_property(_glslang_dir glslang::glslang INTERFACE_INCLUDE_DIRECTORIES)
    get_filename_component(_glslang_root "${_glslang_dir}" DIRECTORY)
    set(_conan_hints "${_glslang_root}/bin")
endif()

# Build platform-specific search hints
set(_shader_compiler_hints ${_conan_hints})
if(APPLE)
    list(APPEND _shader_compiler_hints
        "${HOMEBREW_PREFIX}/bin"
        "$ENV{VULKAN_SDK}/bin"
    )
elseif(WIN32)
    if(DEFINED ENV{VULKAN_SDK})
        list(APPEND _shader_compiler_hints "$ENV{VULKAN_SDK}/Bin")
    endif()
    # Common Vulkan SDK default install location on Windows
    file(GLOB _vk_sdk_bin_dirs "C:/VulkanSDK/*/Bin")
    list(APPEND _shader_compiler_hints ${_vk_sdk_bin_dirs})
else()
    list(APPEND _shader_compiler_hints
        "$ENV{VULKAN_SDK}/bin"
        "/usr/local/bin"
    )
endif()

find_program(GLSLC_EXECUTABLE NAMES glslc
    HINTS ${_shader_compiler_hints}
)

find_program(GLSLANG_VALIDATOR_EXECUTABLE NAMES glslangValidator
    HINTS ${_shader_compiler_hints}
)

if(NOT GLSLC_EXECUTABLE AND NOT GLSLANG_VALIDATOR_EXECUTABLE)
    if(WIN32)
        message(WARNING "No GLSL compiler found (glslc or glslangValidator). "
            "Shader compilation will be unavailable. "
            "Install the LunarG Vulkan SDK from https://vulkan.lunarg.com "
            "or ensure VULKAN_SDK env var is set.")
    else()
        message(WARNING "No GLSL compiler found (glslc or glslangValidator). "
            "Shader compilation will be unavailable. "
            "Install via: brew install glslang  OR  set VULKAN_SDK env var.")
    endif()
endif()

set(SHADER_OUTPUT_DIR "${CMAKE_BINARY_DIR}/shaders")

# shader_compile_glsl(
#   TARGET      <cmake-target>       # target that depends on this shader
#   SOURCE      <path/to/shader.vert|.frag|.comp|...>
#   OUTPUT      <name.spv>           # filename only; placed in SHADER_OUTPUT_DIR
# )
function(shader_compile_glsl)
    cmake_parse_arguments(SHADER "" "TARGET;SOURCE;OUTPUT" "" ${ARGN})

    if(NOT SHADER_TARGET)
        message(FATAL_ERROR "shader_compile_glsl: TARGET is required")
    endif()
    if(NOT SHADER_SOURCE)
        message(FATAL_ERROR "shader_compile_glsl: SOURCE is required")
    endif()
    if(NOT SHADER_OUTPUT)
        message(FATAL_ERROR "shader_compile_glsl: OUTPUT is required")
    endif()

    set(output_path "${SHADER_OUTPUT_DIR}/${SHADER_OUTPUT}")

    file(MAKE_DIRECTORY "${SHADER_OUTPUT_DIR}")

    if(GLSLC_EXECUTABLE)
        set(compile_cmd
            "${GLSLC_EXECUTABLE}"
            --target-env=vulkan1.4
            -O
            -o "${output_path}"
            "${SHADER_SOURCE}"
        )
        set(compiler_name "glslc")
    elseif(GLSLANG_VALIDATOR_EXECUTABLE)
        set(compile_cmd
            "${GLSLANG_VALIDATOR_EXECUTABLE}"
            --target-env vulkan1.4
            --spirv-val
            -V
            -o "${output_path}"
            "${SHADER_SOURCE}"
        )
        set(compiler_name "glslangValidator")
    else()
        message(FATAL_ERROR "shader_compile_glsl: no GLSL compiler available")
    endif()

    add_custom_command(
        OUTPUT  "${output_path}"
        COMMAND ${compile_cmd}
        DEPENDS "${SHADER_SOURCE}"
        COMMENT "[${compiler_name}] Compiling ${SHADER_SOURCE} -> ${SHADER_OUTPUT}"
        VERBATIM
    )

    set(shader_target_name "shader_${SHADER_OUTPUT}")
    add_custom_target("${shader_target_name}" DEPENDS "${output_path}")
    add_dependencies("${SHADER_TARGET}" "${shader_target_name}")
endfunction()
