#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ $# -gt 1 ]]; then
    por::die "Usage: ./scripts/run-macos.sh [Debug|RelWithDebInfo|Release]"
fi

por::require_macos

build_type="$(por::normalize_build_type "${1:-Debug}")"
repo_root="$(por::repo_root)"
preset="$(por::resolve_macos_preset "${build_type}")"
conan_run="${repo_root}/build/conan/conanrun.sh"

if ! executable_path="$(por::resolve_executable "${preset}")"; then
    por::log_step "Renderer executable is missing; running build"
    "${repo_root}/scripts/build-macos.sh" "${build_type}"
    executable_path="$(por::resolve_executable "${preset}")" || por::die "Could not locate the built renderer executable."
fi

if [[ -f "${conan_run}" ]]; then
    por::log_step "Activating Conan run environment"
    # shellcheck disable=SC1090
    set +u
    source "${conan_run}"
    set -u
fi

por::log_step "Launching ${executable_path}"
"${executable_path}"
