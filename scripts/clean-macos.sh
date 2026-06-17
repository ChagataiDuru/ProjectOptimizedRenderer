#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

por::require_macos

remove_all=0
if [[ $# -gt 1 ]]; then
    por::die "Usage: ./scripts/clean-macos.sh [--all]"
fi
if [[ $# -eq 1 ]]; then
    case "$1" in
        --all)
            remove_all=1
            ;;
        *)
            por::die "Usage: ./scripts/clean-macos.sh [--all]"
            ;;
    esac
fi

repo_root="$(por::repo_root)"
targets=(
    "${repo_root}/build/debug"
    "${repo_root}/build/macos-debug"
    "${repo_root}/build/relwithdebinfo"
    "${repo_root}/build/release"
)

if [[ ${remove_all} -eq 1 ]]; then
    targets+=(
        "${repo_root}/build/conan"
        "${repo_root}/CMakeUserPresets.json"
    )
fi

por::log_step "Removing generated build outputs"
for target in "${targets[@]}"; do
    if [[ -e "${target}" ]]; then
        printf 'Removing %s\n' "${target}"
        rm -rf "${target}"
    fi
done

por::log_step "Clean complete"
