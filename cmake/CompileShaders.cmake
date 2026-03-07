# CompileShaders.cmake
# Provides shader_compile_glsl() for GLSL -> SPIR-V compilation.
# Targets Vulkan 1.4 / SPIR-V 1.6.

if(TARGET glslang::glslang)
    get_target_property(_glslang_dir glslang::glslang INTERFACE_INCLUDE_DIRECTORIES)
    get_filename_component(_glslang_root "${_glslang_dir}" DIRECTORY)
    set(_conan_hints "${_glslang_root}/bin")
endif()

find_program(GLSLC_EXECUTABLE NAMES glslc
    HINTS
        "${HOMEBREW_PREFIX}/bin"
        "$ENV{VULKAN_SDK}/bin"
)

find_program(GLSLANG_VALIDATOR_EXECUTABLE NAMES glslangValidator
    HINTS
        "${HOMEBREW_PREFIX}/bin"
        "$ENV{VULKAN_SDK}/bin"
)

if(NOT GLSLC_EXECUTABLE AND NOT GLSLANG_VALIDATOR_EXECUTABLE)
    message(WARNING "No GLSL compiler found (glslc or glslangValidator). "
        "Shader compilation will be unavailable. "
        "Install via: brew install glslang  OR  set VULKAN_SDK env var.")
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
