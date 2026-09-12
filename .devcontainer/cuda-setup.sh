#!/usr/bin/env bash
set -euo pipefail

install_cuda="${INSTALL_CUDA:-off}"
cuda_version="${CUDA_VERSION:-12.9}"

case "${install_cuda,,}" in
  true|on|1|yes|enabled) ;;
  *) exit 0 ;;
esac

source /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "cuda-setup.sh: CUDA installation requires Ubuntu." >&2
  exit 1
fi

repo_tag="ubuntu${VERSION_ID//./}"
architecture="$(dpkg --print-architecture)"
case "$architecture" in
  amd64) repository_arch="x86_64" ;;
  arm64) repository_arch="sbsa" ;;
  *)
    echo "cuda-setup.sh: unsupported architecture '${architecture}'." >&2
    exit 1
    ;;
esac

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl gnupg

keyring_deb="/tmp/cuda-keyring.deb"
curl -fsSL \
  "https://developer.download.nvidia.com/compute/cuda/repos/${repo_tag}/${repository_arch}/cuda-keyring_1.1-1_all.deb" \
  -o "$keyring_deb"
dpkg -i "$keyring_deb"
rm -f "$keyring_deb"

cuda_package="cuda-toolkit-${cuda_version//./-}"
apt-get update
apt-get install -y --no-install-recommends "$cuda_package"
apt-get clean
rm -rf /var/lib/apt/lists/*
