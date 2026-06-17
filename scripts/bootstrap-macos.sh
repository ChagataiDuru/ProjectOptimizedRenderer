#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

if [[ $# -gt 1 ]]; then
    por::die "Usage: ./scripts/bootstrap-macos.sh [Debug|RelWithDebInfo|Release]"
fi

por::require_macos

build_type="$(por::normalize_build_type "${1:-Debug}")"
repo_root="$(por::repo_root)"
preset="$(por::resolve_macos_preset "${build_type}")"

por::log_step "Bootstrapping ${build_type} environment for preset '${preset}'"
cd "${repo_root}"

por::log_step "Syncing git submodules"
git submodule update --init --recursive

por::log_step "Detecting Conan profile"
conan profile detect --force

por::log_step "Installing Conan dependencies"
conan install . --output-folder=build/conan --build=missing -s "build_type=${build_type}"

por::log_step "Bootstrap complete"
printf 'Preset: %s\n' "${preset}"
