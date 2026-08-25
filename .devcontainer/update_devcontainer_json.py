#!/usr/bin/env python3
"""Merge managed CPU/CUDA/ROS settings into devcontainer.json."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


DEFAULT_CUDA_VERSION = "12.9"
CUDA_FEATURE = "ghcr.io/devcontainers/features/nvidia-cuda:2"
CUDA_REMOTE_ENV = {
    "PATH": "/usr/local/cuda/bin:${containerEnv:PATH}",
    "LD_LIBRARY_PATH": (
        "/usr/local/cuda/lib64:${containerEnv:LD_LIBRARY_PATH}"
    ),
    "CUDA_HOME": "/usr/local/cuda",
}
ROS_CONTAINER_ENV = {
    "ROS_LOCALHOST_ONLY": "1",
    "ROS_DOMAIN_ID": "42",
}
DEFAULT_EXTENSIONS = [
    "ms-vscode.cpptools",
    "ms-vscode.cmake-tools",
    "ms-vscode.cpptools-extension-pack",
    "ms-vscode.cpp-devtools",
    "llvm-vs-code-extensions.vscode-clangd",
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.debugpy",
    "ms-python.autopep8",
    "njpwerner.autodocstring",
    "Gruntfuggly.todo-tree",
    "openai.chatgpt",
]
DOCKER_GPU_ARGS = ["--gpus", "all"]
PODMAN_GPU_ARGS = [
    "--device",
    "nvidia.com/gpu=all",
    "--security-opt=label=disable",
]


def _strip_jsonc_comments(text: str) -> str:
    """Remove JSONC comments while preserving quoted comment markers."""

    output: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        character = text[index]
        if in_string:
            output.append(character)
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            index += 1
            continue

        if character == '"':
            in_string = True
            output.append(character)
            index += 1
            continue
        if character == "/" and index + 1 < len(text):
            next_character = text[index + 1]
            if next_character == "/":
                index += 2
                while index < len(text) and text[index] not in "\r\n":
                    index += 1
                continue
            if next_character == "*":
                index += 2
                while index + 1 < len(text) and text[index:index + 2] != "*/":
                    if text[index] in "\r\n":
                        output.append(text[index])
                    index += 1
                index = min(index + 2, len(text))
                continue
        output.append(character)
        index += 1
    return "".join(output)


def _load_existing(path: Path) -> dict:
    """Load an existing JSON/JSONC object, or return an empty configuration."""

    if not path.is_file():
        return {}
    text = _strip_jsonc_comments(path.read_text(encoding="utf-8"))
    text = re.sub(r",(\s*[}\]])", r"\1", text).strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Cannot parse {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Top-level value in {path} must be an object")
    return value


def _strip_managed_gpu_args(arguments: list) -> list:
    """Remove GPU arguments owned by this updater from a run-argument list."""

    output = []
    index = 0
    while index < len(arguments):
        current = arguments[index]
        following = (
            arguments[index + 1] if index + 1 < len(arguments) else None
        )
        if (
            (current == "--gpus" and following == "all")
            or (
                current == "--device"
                and following == "nvidia.com/gpu=all"
            )
            or (
                current == "--security-opt"
                and following == "label=disable"
            )
        ):
            index += 2
            continue
        if current in {
            "--gpus=all",
            "--device=nvidia.com/gpu=all",
            "--security-opt=label=disable",
        }:
            index += 1
            continue
        output.append(current)
        index += 1
    return output


def main() -> int:
    """Merge environment-selected CUDA/ROS settings into devcontainer JSON."""

    cuda_enabled = os.environ.get("CUDA", "off") == "on"
    cuda_version = os.environ.get(
        "CUDA_VERSION", DEFAULT_CUDA_VERSION
    )
    gpu_runtime = os.environ.get(
        "DEVCONTAINER_GPU_RUNTIME", "docker"
    )
    if gpu_runtime not in {"docker", "podman"}:
        raise ValueError(
            "DEVCONTAINER_GPU_RUNTIME must be docker or podman"
        )

    ros_mode = os.environ.get("ROS_MODE", "none")
    ros_distro = os.environ.get("ROS_DISTRO", "")
    ros_profile = os.environ.get("ROS_PROFILE", "ros-base")
    path = Path(
        os.environ.get(
            "DEVCONTAINER_JSON_PATH",
            Path(__file__).with_name("devcontainer.json"),
        )
    )
    data = _load_existing(path)
    data.setdefault("name", "torchAutoForge-deploy")

    build = data.get("build", {})
    build = build if isinstance(build, dict) else {}
    build["dockerfile"] = "Dockerfile"
    if ros_mode == "none":
        build.pop("args", None)
    else:
        build["args"] = {
            "ROS_MODE": ros_mode,
            "ROS_DISTRO": ros_distro,
            "ROS_PROFILE": ros_profile,
        }
    data["build"] = build

    features = data.get("features", {})
    features = features if isinstance(features, dict) else {}
    features["ghcr.io/devcontainers/features/conda:1"] = {
        "addCondaForge": True,
        "version": "latest",
    }
    features["ghcr.io/devcontainers/features/python:1"] = {
        "installTools": True,
        "enableShared": True,
        "version": "3.12",
    }
    if cuda_enabled:
        features[CUDA_FEATURE] = {
            "installCudnn": True,
            "installCudnnDev": True,
            "installNvtx": True,
            "installToolkit": True,
            "cudaVersion": cuda_version,
            "cudnnVersion": "automatic",
        }
    else:
        features.pop(CUDA_FEATURE, None)
    data["features"] = dict(sorted(features.items()))

    run_arguments = _strip_managed_gpu_args(
        data.get("runArgs", [])
    )
    if cuda_enabled:
        managed_arguments = (
            DOCKER_GPU_ARGS
            if gpu_runtime == "docker"
            else PODMAN_GPU_ARGS
        )
        run_arguments = managed_arguments + run_arguments
    if run_arguments:
        data["runArgs"] = run_arguments
    else:
        data.pop("runArgs", None)

    remote_environment = data.get("remoteEnv", {})
    remote_environment = (
        remote_environment
        if isinstance(remote_environment, dict)
        else {}
    )
    if cuda_enabled:
        remote_environment.update(CUDA_REMOTE_ENV)
    else:
        for key in CUDA_REMOTE_ENV:
            remote_environment.pop(key, None)
    if remote_environment:
        data["remoteEnv"] = remote_environment
    else:
        data.pop("remoteEnv", None)

    container_environment = data.get("containerEnv", {})
    container_environment = (
        container_environment
        if isinstance(container_environment, dict)
        else {}
    )
    if ros_mode != "none":
        container_environment.update(ROS_CONTAINER_ENV)
    else:
        for key in ROS_CONTAINER_ENV:
            container_environment.pop(key, None)
    if container_environment:
        data["containerEnv"] = container_environment
    else:
        data.pop("containerEnv", None)

    customizations = data.get("customizations", {})
    customizations = (
        customizations if isinstance(customizations, dict) else {}
    )
    vscode = customizations.get("vscode", {})
    vscode = vscode if isinstance(vscode, dict) else {}
    existing_extensions = vscode.get("extensions", [])
    existing_extensions = (
        existing_extensions
        if isinstance(existing_extensions, list)
        else []
    )
    vscode["extensions"] = DEFAULT_EXTENSIONS + [
        extension
        for extension in existing_extensions
        if extension not in DEFAULT_EXTENSIONS
    ]
    customizations["vscode"] = vscode
    data["customizations"] = customizations

    json.dump(data, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(
            f"update_devcontainer_json.py: {error}",
            file=sys.stderr,
        )
        raise SystemExit(2) from error
