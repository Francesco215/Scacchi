#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
SRC_DIR="${THIRD_PARTY_DIR}/benzene-vanilla-cmake"
BUILD_DIR="${SRC_DIR}/build"

REPO_URL="${MOHEX_REPO_URL:-https://github.com/cgao3/benzene-vanilla-cmake.git}"
REPO_REF="${MOHEX_REPO_REF:-master}"

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

cpu_count() {
  if have_cmd nproc; then
    nproc
  elif have_cmd sysctl; then
    sysctl -n hw.ncpu
  else
    getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1
  fi
}

require_cmd() {
  local cmd="$1"
  if ! have_cmd "${cmd}"; then
    printf 'Missing required command: %s\n' "${cmd}" >&2
    exit 1
  fi
}

mkdir -p "${THIRD_PARTY_DIR}"

require_cmd git
require_cmd cmake
require_cmd c++

if [[ ! -d "${SRC_DIR}" ]]; then
  git clone "${REPO_URL}" "${SRC_DIR}"
fi

git -C "${SRC_DIR}" fetch --all --tags
git -C "${SRC_DIR}" checkout "${REPO_REF}"

cmake -S "${SRC_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j"$(cpu_count)"

cat <<EOF

MoHex build complete.

Binary:
  ${BUILD_DIR}/src/mohex/mohex

If you want this shell to use it by default:
  export MOHEX_BINARY="${BUILD_DIR}/src/mohex/mohex"
EOF
