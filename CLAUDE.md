# CLAUDE.md

Guidance for Claude Code when working in `torchAutoForge-deploy`.

Read `AGENTS.md` first for engineering standards, package identity, repository
tailoring, and staged review. This file summarizes common commands and project
priorities; `AGENTS.md` remains authoritative.

## Project Overview

`torchAutoForge-deploy` is a C++20 deployment library for ONNX Runtime-backed
and optional TensorRT-backed ML inference:

- `ptafdeploy::inference::CInferenceManager` is the public backend-agnostic
  facade.
- `ptafdeploy::inference::CModelFacade` is the role-level facade used by
  C++/Python/MATLAB consumers.
- `ptafdeploy::inference::onnxruntime::CInferenceManager_ORT` is the ONNX
  Runtime backend.
- `ptafdeploy::inference::tensorrt::CInferenceManager_TensorRT_Engine` is the
  optional serialized-engine backend when `autoforge_deploy_ENABLE_TENSORRT=ON`.
- Python/MATLAB wrappers use the populated gtwrap interfaces under `src/`.
- OptiX is out of scope for this repo; keep this build lighter than renderer
  projects.

## Build Commands

```bash
./build_lib.sh
./build_lib.sh -t debug
./build_lib.sh -t release -i
./build_lib.sh -r
./build_lib.sh --clean -t debug
./build_lib.sh -N
./build_lib.sh --skip-tests
./build_lib.sh -D autoforge_deploy_ENABLE_CUDA=ON
./build_lib.sh -D autoforge_deploy_ENABLE_TENSORRT=ON -D TENSORRT_ROOT=/path/to/TensorRT
./build_lib.sh -p
./build_lib.sh -m
./build_lib.sh -p --wrap-update
./build_ros2.sh --clean
./run_in_container.sh --build -- ./build_lib.sh
```

Manual CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel $(nproc)
ctest --test-dir build --output-on-failure -j $(nproc)
```

## Tests

Tests use Catch2 v3. New `test*.cpp` files under `tests/<subdir>/` are
auto-discovered.

Current tests cover ORT metadata extraction, ORT inference, generic tensor
helpers, facade/model-role behavior, target-specific wrapper paths, optional
TensorRT behavior, and the ROS lifecycle bridge.

Do not add tests for generic template/logger/PTX/CMake mechanics while
tailoring template upgrades. Permanent tests here must exercise
`torchAutoForge-deploy` inference, wrappers, programs, or ROS-facing behavior.

Do not import `VerifyTemplateProject*` scripts or place recursive
configure/build/install/package/consumer checks in ordinary CTest. Run those as
fresh out-of-tree acceptance commands or CI jobs. Keep any exceptional
CMake-script test lightweight, target-owned, non-recursive, and limited to
behavior unavailable through Catch2, pytest, an existing target, or the
acceptance matrix. ROS ament/launch tests remain a deliberate project-specific
exception.

Ordinary configure/build operations must leave the gtwrap checkout and
superproject gitlink unchanged. Wrapper updates and submodule initialization are
explicit maintenance operations. Python wheels include only project-owned
runtime libraries alongside the extension; ONNX Runtime remains an external
runtime prerequisite.

## Development Priorities

1. Maintain the completed semantic alignment with
   `/home/peterc/devDir/dev-tools/cpp_cuda_template_project` signed tag
   `v2.0.1` at `1d87153b2d060bf03c2c9adcd1df6c6d4f40ea09`. Preserve ORT
   dependency wiring, independent TensorRT/CUDA policy, and the absence of OptiX.
   Track product work in `doc/development/autoforge_deploy_upgrade_plan.md`.
2. Make wrappers work for MATLAB/Python through a stable generic inference
   facade.
3. Review implementation for redundancy and readability before adding
   higher-level model support.
4. Add model-role facade for centroiding, object detection, feature matching,
   tracking, optical flow, and future learned modules.
5. Add YOLO and centroiding ONNX examples without hardcoding those models into
   the core ORT backend.
