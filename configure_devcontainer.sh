#!/usr/bin/env bash
# Configure the checked-in devcontainer for CPU/CUDA and optional ROS 2.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVCONTAINER_DIR="${ROOT_DIR}/.devcontainer"
JSON_PATH="${DEVCONTAINER_DIR}/devcontainer.json"
DOCKERFILE_PATH="${DEVCONTAINER_DIR}/Dockerfile"
JSON_UPDATER="${DEVCONTAINER_DIR}/update_devcontainer_json.py"

cuda="off"
cuda_version="12.9"
gpu_runtime="auto"
base_image="mcr.microsoft.com/devcontainers/cpp:1-ubuntu-24.04"
ros_mode="none"
ros_distro=""
ros_profile="ros-base"

usage() {
  cat <<'EOF'
Usage: ./configure_devcontainer.sh [options]

Options:
  --cuda | --no-cuda      Enable or disable CUDA (default: disabled).
  --cuda-version <v>      CUDA feature version (default: 12.9).
  --gpu-runtime <mode>    auto, docker, or podman.
  --base-image <image>    Complete Docker base image.
  --ros2 <distro>         Install a ROS 2 distro, such as jazzy.
  --ros-profile <profile> ros-base or desktop.
  --no-ros                Disable ROS installation.
  -h, --help              Show this help.
EOF
}

while (($# > 0)); do
  case "$1" in
    --cuda)
      cuda="on"
      shift
      ;;
    --no-cuda)
      cuda="off"
      shift
      ;;
    --cuda-version)
      [[ $# -ge 2 ]] || { echo "--cuda-version requires a value" >&2; exit 2; }
      cuda_version="$2"
      shift 2
      ;;
    --gpu-runtime)
      [[ $# -ge 2 ]] || { echo "--gpu-runtime requires a value" >&2; exit 2; }
      gpu_runtime="$2"
      shift 2
      ;;
    --base-image)
      [[ $# -ge 2 ]] || { echo "--base-image requires a value" >&2; exit 2; }
      base_image="$2"
      shift 2
      ;;
    --ros2)
      [[ $# -ge 2 ]] || { echo "--ros2 requires a distro" >&2; exit 2; }
      ros_mode="ros2"
      ros_distro="$2"
      shift 2
      ;;
    --ros-profile)
      [[ $# -ge 2 ]] || { echo "--ros-profile requires a value" >&2; exit 2; }
      ros_profile="$2"
      shift 2
      ;;
    --no-ros)
      ros_mode="none"
      ros_distro=""
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$gpu_runtime" == "auto" ]]; then
  if command -v docker >/dev/null 2>&1; then
    gpu_runtime="docker"
  elif command -v podman >/dev/null 2>&1; then
    gpu_runtime="podman"
  else
    gpu_runtime="docker"
  fi
fi
[[ "$gpu_runtime" == "docker" || "$gpu_runtime" == "podman" ]] \
  || { echo "GPU runtime must be docker or podman" >&2; exit 2; }
[[ "$ros_profile" == "ros-base" || "$ros_profile" == "desktop" ]] \
  || { echo "ROS 2 profile must be ros-base or desktop" >&2; exit 2; }
if [[ "$ros_mode" == "ros2" && -z "$ros_distro" ]]; then
  echo "A ROS 2 distro is required" >&2
  exit 2
fi

temporary_json="$(mktemp)"
temporary_dockerfile="$(mktemp)"
cleanup() {
  rm -f "$temporary_json" "$temporary_dockerfile"
}
trap cleanup EXIT

CUDA="$cuda" \
CUDA_VERSION="$cuda_version" \
DEVCONTAINER_GPU_RUNTIME="$gpu_runtime" \
ROS_MODE="$ros_mode" \
ROS_DISTRO="$ros_distro" \
ROS_PROFILE="$ros_profile" \
DEVCONTAINER_JSON_PATH="$JSON_PATH" \
python3 "$JSON_UPDATER" > "$temporary_json"

awk -v replacement="FROM ${base_image}" '
  BEGIN { replaced = 0 }
  /^FROM[[:space:]]+/ && replaced == 0 {
    print replacement
    replaced = 1
    next
  }
  { print }
  END { if (replaced == 0) exit 1 }
' "$DOCKERFILE_PATH" > "$temporary_dockerfile"

mv "$temporary_json" "$JSON_PATH"
mv "$temporary_dockerfile" "$DOCKERFILE_PATH"
trap - EXIT

printf 'Configured %s (%s, CUDA %s, ROS mode %s).\n' \
  "$JSON_PATH" "$gpu_runtime" "$cuda" "$ros_mode"
