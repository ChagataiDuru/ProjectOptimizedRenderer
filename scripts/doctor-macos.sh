#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

por::require_macos

repo_root="$(por::repo_root)"
python_bin="$(por::python_bin)"

por::log_step "Running environment doctor"
cd "${repo_root}"
"${python_bin}" "${repo_root}/tools/doctor/doctor.py"
