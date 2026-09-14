#!/usr/bin/env bash
# Build the optional ROS 2 overlay independently of the standalone C++ build.

set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}/ros2"
ROS_DISTRO_NAME="${ROS_DISTRO:-jazzy}"

build_type="RelWithDebInfo"
clean=false
skip_tests=false
enable_cuda=false
metadata_sync=true
packages_select=()
cmake_args=()
colcon_args=()

info() { printf '\033[34m[INFO]\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[WARN]\033[0m %s\n' "$*" >&2; }
die() { printf '\033[31m[ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Usage:
  ./build_ros2.sh [options]

Options:
  --clean                    Remove ros2/build, ros2/install, and ros2/log.
  --skip-tests               Skip colcon test and test-result.
  --debug                    Use CMAKE_BUILD_TYPE=Debug.
  --release                  Use CMAKE_BUILD_TYPE=Release.
  --relwithdebinfo           Use RelWithDebInfo (default).
  --build-type <type>        Use an explicit CMake build type.
  --packages-select <pkg...> Build and test selected packages.
  --cuda                     Enable core CUDA and PTX compile/embed support.
  --cmake-arg <arg>          Append one CMake argument; repeatable.
  --colcon-arg <arg>         Append one colcon build argument; repeatable.
  --no-version-sync          Keep current ROS package metadata.
  -h, --help                 Show this help.

Examples:
  ./build_ros2.sh --clean
  ./build_ros2.sh --packages-select ptafdeploy_interfaces ptafdeploy_ros
  ./build_ros2.sh --cuda --cmake-arg -DCMAKE_CUDA_ARCHITECTURES=89

EOF
}

parse_args() {
    while (($# > 0)); do
        case "$1" in
            --clean)
                clean=true
                shift
                ;;
            --skip-tests)
                skip_tests=true
                shift
                ;;
            --debug)
                build_type="Debug"
                shift
                ;;
            --release)
                build_type="Release"
                shift
                ;;
            --relwithdebinfo)
                build_type="RelWithDebInfo"
                shift
                ;;
            --build-type)
                [[ $# -ge 2 ]] || die "--build-type requires a value"
                build_type="$2"
                shift 2
                ;;
            --packages-select)
                shift
                [[ $# -gt 0 && "$1" != --* ]] \
                    || die "--packages-select requires at least one package"
                while [[ $# -gt 0 && "$1" != --* ]]; do
                    packages_select+=("$1")
                    shift
                done
                ;;
            --cuda)
                enable_cuda=true
                shift
                ;;
            --cmake-arg)
                [[ $# -ge 2 ]] || die "--cmake-arg requires a value"
                cmake_args+=("$2")
                shift 2
                ;;
            --colcon-arg)
                [[ $# -ge 2 ]] || die "--colcon-arg requires a value"
                colcon_args+=("$2")
                shift 2
                ;;
            --no-version-sync)
                metadata_sync=false
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "Unknown argument: $1"
                ;;
        esac
    done
}

source_ros_environment() {
    local setup_file_="/opt/ros/${ROS_DISTRO_NAME}/setup.bash"
    [[ -f "$setup_file_" ]] \
        || die "ROS setup not found: ${setup_file_}"

    set +u
    # shellcheck disable=SC1090
    source "$setup_file_"
    set -u
    command -v colcon >/dev/null 2>&1 \
        || die "colcon was not found after sourcing ${setup_file_}"
}

sync_package_metadata() {
    if [[ "$metadata_sync" != true ]]; then
        return
    fi
    if ! "${SCRIPT_DIR}/generate_version.sh" --sync-ros2; then
        die "ROS 2 package metadata synchronization failed"
    fi
}

normalize_onnxruntime_cmake_args() {
    local arg_
    local root_hint_="${ONNXRUNTIME_ROOT:-}"
    local config_dir_
    local has_explicit_dir_=false

    for arg_ in "${cmake_args[@]}"; do
        case "$arg_" in
            -Donnxruntime_DIR=*)
                has_explicit_dir_=true
                ;;
            -DONNXRUNTIME_ROOT=*)
                root_hint_="${arg_#*=}"
                ;;
        esac
    done

    if [[ "$has_explicit_dir_" == true || -z "$root_hint_" ]]; then
        return
    fi

    config_dir_="${root_hint_%/}/lib/cmake/onnxruntime"
    if [[ -d "$config_dir_" ]]; then
        cmake_args+=("-Donnxruntime_DIR=${config_dir_}")
        info "ONNX Runtime CMake package: ${config_dir_}"
    else
        warn "ONNXRUNTIME_ROOT does not contain lib/cmake/onnxruntime: ${root_hint_}"
    fi
}

clean_workspace() {
    local path_
    for path_ in \
        "${WORKSPACE_DIR}/build" \
        "${WORKSPACE_DIR}/install" \
        "${WORKSPACE_DIR}/log"; do
        [[ "$path_" == "${WORKSPACE_DIR}/"* ]] \
            || die "Refusing unexpected clean target: ${path_}"
        if [[ -e "$path_" ]]; then
            rm -rf -- "$path_"
        fi
    done
}

run_build_and_tests() {
    local cuda_flag_="OFF"
    local build_command_=(colcon build --symlink-install)
    local test_command_=(colcon test --event-handlers console_direct+)
    local package_

    [[ "$enable_cuda" == true ]] && cuda_flag_="ON"

    if ((${#packages_select[@]} > 0)); then
        build_command_+=(--packages-select "${packages_select[@]}")
        test_command_+=(--packages-select "${packages_select[@]}")
    fi
    if ((${#colcon_args[@]} > 0)); then
        build_command_+=("${colcon_args[@]}")
    fi
    build_command_+=(
        --cmake-args
        "-DCMAKE_BUILD_TYPE=${build_type}"
        "-DPTAFDEPLOY_ENABLE_CUDA=${cuda_flag_}"
        "${cmake_args[@]}"
    )

    info "Workspace : ${WORKSPACE_DIR}"
    info "ROS distro: ${ROS_DISTRO_NAME}"
    info "Build type: ${build_type}"
    info "CUDA/PTX  : ${cuda_flag_}"

    (
        cd "$WORKSPACE_DIR"
        "${build_command_[@]}"
    )

    if [[ "$skip_tests" == true ]]; then
        return
    fi

    (
        cd "$WORKSPACE_DIR"
        set +u
        # shellcheck disable=SC1091
        source install/setup.bash
        set -u
        "${test_command_[@]}"
        if ((${#packages_select[@]} > 0)); then
            for package_ in "${packages_select[@]}"; do
                colcon test-result \
                    --test-result-base "build/${package_}" --verbose
            done
        else
            colcon test-result --verbose
        fi
    )
}

main() {
    parse_args "$@"
    [[ -d "$WORKSPACE_DIR" ]] \
        || die "ROS 2 workspace not found: ${WORKSPACE_DIR}"
    source_ros_environment
    normalize_onnxruntime_cmake_args
    if [[ "$clean" == true ]]; then
        clean_workspace
    fi
    sync_package_metadata
    run_build_and_tests
}

main "$@"
