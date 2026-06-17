#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ $# -gt 1 ]]; then
    por::die "Usage: ./scripts/build-macos.sh [Debug|RelWithDebInfo|Release]"
fi

por::require_macos

build_type="$(por::normalize_build_type "${1:-Debug}")"
repo_root="$(por::repo_root)"
preset="$(por::resolve_macos_preset "${build_type}")"
toolchain_path="$(por::conan_toolchain_path)"

if [[ ! -f "${toolchain_path}" ]] || ! por::has_expected_conan_presets; then
    por::log_step "Conan toolchain or presets are missing; running bootstrap"
    "${repo_root}/scripts/bootstrap-macos.sh" "${build_type}"
fi

cd "${repo_root}"

por::log_step "Configuring preset '${preset}'"
cmake --preset "${preset}" -DCMAKE_TOOLCHAIN_FILE="${toolchain_path}"

por::log_step "Building preset '${preset}'"
cmake --build --preset "${preset}" --config "${build_type}"

por::log_step "Build complete"
