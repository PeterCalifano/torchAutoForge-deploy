#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: install_onnxruntime.sh <x86_64|arm64|aarch64> <install-prefix>

Downloads the CPU ONNX Runtime release archive matching the target runner and
prints the extracted root directory on stdout.
USAGE
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

arch="$1"
install_prefix="$2"
version="${ORT_VERSION:-1.23.0}"

case "$arch" in
  x86_64|amd64)
    asset_arch="x64"
    ;;
  arm64|aarch64)
    asset_arch="aarch64"
    ;;
  *)
    echo "Unsupported ONNX Runtime CI architecture: ${arch}" >&2
    usage
    exit 2
    ;;
esac

asset_name="onnxruntime-linux-${asset_arch}-${version}"
ort_root="${install_prefix}/${asset_name}"
archive_url="https://github.com/microsoft/onnxruntime/releases/download/v${version}/${asset_name}.tgz"

fix_release_include_layout() {
  local root="$1"
  local cmake_include_dir="${root}/include/onnxruntime"

  if [[ -f "${cmake_include_dir}/onnxruntime_cxx_api.h" ]]; then
    return 0
  fi

  # ORT release CMake targets point at include/onnxruntime, while some tarballs
  # place public headers directly under include. Mirror headers into the target
  # include path inside CI dependency cache instead of patching project CMake.
  mkdir -p "${cmake_include_dir}"
  find "${root}/include" -maxdepth 1 -type f -name '*.h' -exec cp -f {} "${cmake_include_dir}/" \;
  if [[ -d "${root}/include/core" && ! -e "${cmake_include_dir}/core" ]]; then
    cp -a "${root}/include/core" "${cmake_include_dir}/core"
  fi
}

fix_release_library_layout() {
  local root="$1"

  if [[ ! -e "${root}/lib64" && -d "${root}/lib" ]]; then
    ln -s lib "${root}/lib64"
  fi
}

if [[ ! -f "${ort_root}/lib/cmake/onnxruntime/onnxruntimeConfig.cmake" ]]; then
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "${tmp_dir}"' EXIT

  mkdir -p "${install_prefix}"
  echo "Downloading ${archive_url}" >&2
  curl -fsSL "${archive_url}" -o "${tmp_dir}/onnxruntime.tgz"
  tar -xzf "${tmp_dir}/onnxruntime.tgz" -C "${install_prefix}"
fi

fix_release_include_layout "${ort_root}"
fix_release_library_layout "${ort_root}"
printf '%s\n' "${ort_root}"
