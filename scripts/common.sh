#!/usr/bin/env bash

por::die() {
    printf 'Error: %s\n' "$1" >&2
    exit 1
}

por::log_step() {
    printf '\n==> %s\n' "$1"
}

por::script_dir() {
    cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

por::repo_root() {
    cd "$(por::script_dir)/.." && pwd
}

por::python_bin() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        printf '%s\n' "${PYTHON_BIN}"
        return 0
    fi
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi
    por::die "Python 3 was not found. Install python3 or set PYTHON_BIN."
}

por::require_macos() {
    local uname_s
    uname_s="$(uname -s)"
    if [[ "${uname_s}" != "Darwin" ]]; then
        por::die "This script is for macOS hosts only. Detected '${uname_s}'."
    fi
}

por::run_helper() {
    local repo_root python_bin
    repo_root="$(por::repo_root)"
    python_bin="$(por::python_bin)"
    "${python_bin}" "${repo_root}/tools/common/project_tooling.py" "$@"
}

por::normalize_build_type() {
    por::run_helper build-type "${1:-Debug}"
}

por::resolve_macos_preset() {
    por::run_helper preset --platform macos --build-type "${1:-Debug}"
}

por::build_dir_for_preset() {
    por::run_helper build-dir --preset "$1"
}

por::resolve_executable() {
    por::run_helper executable --platform macos --preset "$1"
}

por::conan_toolchain_path() {
    local repo_root
    repo_root="$(por::repo_root)"
    printf '%s\n' "${repo_root}/build/conan/conan_toolchain.cmake"
}

por::cmake_user_presets_path() {
    local repo_root
    repo_root="$(por::repo_root)"
    printf '%s\n' "${repo_root}/CMakeUserPresets.json"
}

por::has_expected_conan_presets() {
    local presets_path
    presets_path="$(por::cmake_user_presets_path)"
    [[ -f "${presets_path}" ]] && grep -q 'build/conan/CMakePresets.json' "${presets_path}"
}
