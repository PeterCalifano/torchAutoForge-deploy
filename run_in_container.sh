#!/usr/bin/env bash
# Launch commands in the repository's standalone development image or start a
# persistent container prepared for VS Code attachment. Both modes preserve
# host ownership for files written through the repository bind mount.
#
# Examples:
#   ./run_in_container.sh                          # interactive bash
#   ./run_in_container.sh ./build/my_app --flag    # run a binary
#   ./run_in_container.sh --build -- ctest --test-dir build
#   ./run_in_container.sh --vscode --engine podman # attach-ready container
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_NAME="$(basename "$ROOT_DIR")"
PROJECT_SLUG="$(
  printf '%s' "$REPO_NAME" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9_.-]+/-/g; s/^[._-]+//; s/[._-]+$//'
)"
if [[ -z "$PROJECT_SLUG" ]]; then
  echo "Could not derive a valid container name from '${REPO_NAME}'."
  exit 1
fi

IMAGE_TAG="${PROJECT_SLUG}-dev:latest"
ENGINE=""
FORCE_BUILD="no"
USE_GPU="yes"
CUDA_VERSION="12.9"
MATLAB_ROOT=""
VSCODE_MODE="no"
VSCODE_USER="vscode"
VSCODE_CONTAINER_NAME="${PROJECT_SLUG}-vscode"
VSCODE_WORKSPACE="/workspaces/${PROJECT_SLUG}"
MATLAB_LABEL_KEY="dev.${PROJECT_SLUG}.matlab-root"

usage() {
  cat <<EOF
Usage: ./run_in_container.sh [options] [--] [command [args...]]

Runs a command inside the repository's standalone container image. The image
is built from .devcontainer/Dockerfile with INSTALL_CUDA=on when missing or
when --build is given. --vscode instead starts a detached attachment container.

Options:
  --build              Force (re)build of the image.
  --no-gpu             Do not request GPU access.
  --vscode             Start an attach-ready VS Code container and exit.
  --container-name <n> Container name for --vscode
                       (default: ${VSCODE_CONTAINER_NAME}).
  --matlab-root <path> Read-only MATLAB installation exposed at the same path.
  --image <name>       Image tag (default: ${IMAGE_TAG}).
  --engine <e>         Container engine: docker or podman (default: autodetect).
  --cuda-version <v>   CUDA toolkit version build argument
                       (default: ${CUDA_VERSION}).
  -h, --help           Show this help.

GPU notes:
  - Docker requires the NVIDIA Container Toolkit and uses --gpus all.
  - Podman requires an NVIDIA CDI specification and uses
    --device nvidia.com/gpu=all.
EOF
}

print_vscode_instructions() {
  cat <<EOF
Container: ${VSCODE_CONTAINER_NAME}
Workspace: ${VSCODE_WORKSPACE}
MATLAB root: ${MATLAB_ROOT:-not mounted}

In VS Code:
  1. Run "Dev Containers: Attach to Running Container...".
  2. Select "${VSCODE_CONTAINER_NAME}".
  3. On first attachment, run "Dev Containers: Open Named Configuration File"
     and set:
       {
         "workspaceFolder": "${VSCODE_WORKSPACE}",
         "remoteUser": "${VSCODE_USER}"
       }

Stop the container with:
  ${ENGINE} stop ${VSCODE_CONTAINER_NAME}
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      FORCE_BUILD="yes"
      ;;
    --no-gpu)
      USE_GPU="no"
      ;;
    --vscode)
      VSCODE_MODE="yes"
      ;;
    --container-name)
      shift
      VSCODE_CONTAINER_NAME="${1:-}"
      [[ -n "$VSCODE_CONTAINER_NAME" ]] \
        || { echo "--container-name requires a value."; exit 1; }
      ;;
    --matlab-root)
      shift
      MATLAB_ROOT="${1:-}"
      [[ -n "$MATLAB_ROOT" ]] \
        || { echo "--matlab-root requires a value."; exit 1; }
      ;;
    --image)
      shift
      IMAGE_TAG="${1:-}"
      [[ -n "$IMAGE_TAG" ]] \
        || { echo "--image requires a value."; exit 1; }
      ;;
    --engine)
      shift
      ENGINE="${1:-}"
      case "$ENGINE" in
        docker|podman) ;;
        *) echo "--engine must be 'docker' or 'podman'."; exit 1 ;;
      esac
      ;;
    --cuda-version)
      shift
      CUDA_VERSION="${1:-}"
      [[ -n "$CUDA_VERSION" ]] \
        || { echo "--cuda-version requires a value."; exit 1; }
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
  shift
done

if [[ "$VSCODE_MODE" == "yes" && $# -gt 0 ]]; then
  echo "--vscode starts an attachment container and does not accept a command."
  exit 1
fi

# Resolve optional host tools before constructing engine arguments so invalid
# paths fail without building or starting any container.
matlab_args_=()
if [[ -n "$MATLAB_ROOT" ]]; then
  MATLAB_ROOT="$(readlink -f -- "$MATLAB_ROOT" 2>/dev/null || true)"
  if [[ -z "$MATLAB_ROOT" || ! -d "$MATLAB_ROOT" ]]; then
    echo "--matlab-root must identify an existing directory."
    exit 1
  fi
  if [[ ! -f "${MATLAB_ROOT}/extern/include/mex.h" ]]; then
    echo "--matlab-root does not contain extern/include/mex.h: ${MATLAB_ROOT}"
    exit 1
  fi

  matlab_args_=(
    --mount "type=bind,source=${MATLAB_ROOT},target=${MATLAB_ROOT},readonly"
    --env "MATLAB_ROOT_DIR=${MATLAB_ROOT}"
  )
fi

# Select the engine explicitly before image inspection so all later commands
# use one stable runtime and ownership model.
if [[ -z "$ENGINE" ]]; then
  if command -v docker >/dev/null 2>&1; then
    ENGINE="docker"
  elif command -v podman >/dev/null 2>&1; then
    ENGINE="podman"
  else
    echo "Neither docker nor podman found in PATH."
    exit 1
  fi
fi

need_build_="$FORCE_BUILD"
if [[ "$need_build_" != "yes" ]] \
   && ! "$ENGINE" image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
  need_build_="yes"
fi
if [[ "$need_build_" == "yes" ]]; then
  echo "Building ${IMAGE_TAG} with ${ENGINE} (CUDA ${CUDA_VERSION})..."
  "$ENGINE" build \
    --build-arg INSTALL_CUDA=on \
    --build-arg CUDA_VERSION="$CUDA_VERSION" \
    --tag "$IMAGE_TAG" \
    "$ROOT_DIR/.devcontainer"
fi

# Keep Podman labeling and user-namespace behavior separate from GPU flags.
# This preserves bind-mount ownership even when GPU access is disabled.
podman_rootless_=""
engine_args_=()
if [[ "$ENGINE" == "podman" ]]; then
  podman_rootless_="$(
    "$ENGINE" info --format '{{.Host.Security.Rootless}}' 2>/dev/null || true
  )"
  if [[ "$podman_rootless_" != "true" \
        && "$podman_rootless_" != "false" ]]; then
    echo "Could not determine whether Podman is rootless."
    exit 1
  fi
  engine_args_+=(--security-opt=label=disable)
fi

gpu_args_=()
if [[ "$USE_GPU" == "yes" ]]; then
  if [[ "$ENGINE" == "docker" ]]; then
    gpu_args_=(--gpus all)
  else
    gpu_args_=(--device nvidia.com/gpu=all)
  fi
fi

if [[ "$VSCODE_MODE" == "yes" ]]; then
  # Reuse an existing attachment container only when its immutable MATLAB
  # mount matches the requested configuration.
  if "$ENGINE" container inspect "$VSCODE_CONTAINER_NAME" \
       >/dev/null 2>&1; then
    container_running_="$(
      "$ENGINE" container inspect \
        --format '{{.State.Running}}' "$VSCODE_CONTAINER_NAME"
    )"
    if [[ "$container_running_" == "true" ]]; then
      container_matlab_root_="$(
        "$ENGINE" container inspect \
          --format "{{index .Config.Labels \"${MATLAB_LABEL_KEY}\"}}" \
          "$VSCODE_CONTAINER_NAME" 2>/dev/null || true
      )"
      [[ "$container_matlab_root_" == "<no value>" ]] \
        && container_matlab_root_=""
      if [[ -n "$MATLAB_ROOT" \
            && "$MATLAB_ROOT" != "$container_matlab_root_" ]]; then
        echo "The running container uses a different MATLAB root: " \
             "${container_matlab_root_:-not mounted}"
        echo "Stop it before recreating it with ${MATLAB_ROOT}."
        exit 1
      fi
      [[ -n "$container_matlab_root_" ]] \
        && MATLAB_ROOT="$container_matlab_root_"
      echo "VS Code container is already running."
      print_vscode_instructions
      exit 0
    fi

    echo "Container '${VSCODE_CONTAINER_NAME}' exists but is stopped."
    echo "Remove it before recreating the attachment container:"
    echo "  ${ENGINE} rm ${VSCODE_CONTAINER_NAME}"
    exit 1
  fi

  # Query the image rather than assuming the devcontainer user's numeric IDs.
  # Docker requires an exact host match; rootless Podman maps the host user to
  # the image's declared development user through keep-id.
  if ! image_user_ids_="$(
    # The substitutions are intentionally evaluated by the image shell.
    # shellcheck disable=SC2016
    "$ENGINE" run --rm --entrypoint /bin/sh "$IMAGE_TAG" \
      -c 'printf "%s:%s\n" "$(id -u vscode)" "$(id -g vscode)"'
  )"; then
    echo "Image '${IMAGE_TAG}' does not provide the '${VSCODE_USER}' user."
    exit 1
  fi
  if [[ ! "$image_user_ids_" =~ ^[0-9]+:[0-9]+$ ]]; then
    echo "Could not resolve ${VSCODE_USER} numeric IDs in ${IMAGE_TAG}."
    exit 1
  fi
  image_user_uid_="${image_user_ids_%%:*}"
  image_user_gid_="${image_user_ids_##*:}"
  host_uid_="$(id -u)"
  host_gid_="$(id -g)"

  vscode_engine_args_=("${engine_args_[@]}")
  if [[ "$ENGINE" == "podman" ]]; then
    if [[ "$podman_rootless_" != "true" ]]; then
      echo "--vscode requires rootless Podman for ownership-preserving keep-id."
      exit 1
    fi
    vscode_engine_args_+=(
      --userns="keep-id:uid=${image_user_uid_},gid=${image_user_gid_}"
    )
    if [[ "$USE_GPU" == "yes" ]]; then
      vscode_engine_args_+=(--group-add keep-groups)
    fi
  elif [[ "$image_user_ids_" != "${host_uid_}:${host_gid_}" ]]; then
    echo "Docker cannot safely map ${VSCODE_USER} (${image_user_ids_}) to " \
         "host ${host_uid_}:${host_gid_}."
    echo "Use the normal Dev Containers workflow or tailor the image user IDs."
    exit 1
  fi

  # Forward only a live SSH-agent socket. The ephemeral container is removed
  # when stopped, so a later host session can supply a fresh socket path.
  vscode_ssh_args_=()
  if [[ -n "${SSH_AUTH_SOCK:-}" ]]; then
    ssh_agent_path_="$(readlink -f -- "$SSH_AUTH_SOCK" 2>/dev/null || true)"
    if [[ -S "$ssh_agent_path_" ]]; then
      vscode_ssh_args_=(
        --mount \
          "type=bind,source=${ssh_agent_path_},target=/tmp/host-ssh-agent.sock"
        --env SSH_AUTH_SOCK=/tmp/host-ssh-agent.sock
      )
    else
      echo "Warning: SSH_AUTH_SOCK is not a socket; forwarding is disabled."
    fi
  else
    echo "Warning: SSH_AUTH_SOCK is unset; forwarding is disabled."
  fi

  container_id_="$(
    "$ENGINE" run --detach --rm --init \
      --name "$VSCODE_CONTAINER_NAME" \
      "${vscode_engine_args_[@]}" \
      "${gpu_args_[@]}" \
      "${vscode_ssh_args_[@]}" \
      "${matlab_args_[@]}" \
      --user "$VSCODE_USER" \
      --label "${MATLAB_LABEL_KEY}=${MATLAB_ROOT}" \
      --mount "type=bind,source=${ROOT_DIR},target=${VSCODE_WORKSPACE}" \
      --workdir "$VSCODE_WORKSPACE" \
      --env DISPLAY="${DISPLAY:-}" \
      --entrypoint /usr/bin/sleep \
      "$IMAGE_TAG" infinity
  )"

  echo "Started VS Code container ${container_id_}."
  print_vscode_instructions
  exit 0
fi

# Ordinary command mode uses the host numeric IDs and a universally writable
# temporary HOME so files created in the repository remain host-owned without
# depending on the image's passwd database.
host_uid_="$(id -u)"
host_gid_="$(id -g)"
identity_args_=(
  --user "${host_uid_}:${host_gid_}"
  --env HOME=/tmp
)
if [[ "$ENGINE" == "podman" && "$podman_rootless_" == "true" ]]; then
  identity_args_+=(--userns=keep-id)
  if [[ "$USE_GPU" == "yes" ]]; then
    identity_args_+=(--group-add keep-groups)
  fi
fi

tty_args_=()
if [[ -t 0 && -t 1 ]]; then
  tty_args_=(-it)
fi
if [[ $# -eq 0 ]]; then
  set -- bash
fi

exec "$ENGINE" run --rm \
  "${tty_args_[@]}" \
  "${engine_args_[@]}" \
  "${identity_args_[@]}" \
  "${gpu_args_[@]}" \
  "${matlab_args_[@]}" \
  --mount "type=bind,source=${ROOT_DIR},target=/workspace" \
  --workdir /workspace \
  --env DISPLAY="${DISPLAY:-}" \
  "$IMAGE_TAG" "$@"
