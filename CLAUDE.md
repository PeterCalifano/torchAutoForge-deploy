# CLAUDE.md

Guidance for Claude Code when working in `torchAutoForge-deploy`.

This file mirrors the current repository guidance in `AGENTS.md`.

## Project Overview

`torchAutoForge-deploy` is a C++20 deployment library for ONNX Runtime-backed
ML inference. The current direction is:

- `deploy_infer::CInferenceManager` is the public backend-agnostic facade.
- `deploy_ort::CInferenceManager_ORT` is the ONNX Runtime backend.
- TensorRT standalone support is currently a not-implemented stub.
- Python/MATLAB wrappers are planned through gtwrap but interface files are not
  populated yet.
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
./build_lib.sh -D ENABLE_CUDA=ON
./build_lib.sh -p
./build_lib.sh -m
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
helpers, facade dispatch, and TensorRT stub failure behavior.

## Development Priorities

1. Keep CMake/build config synced with
   `/home/peterc/devDir/dev-tools/cpp_cuda_template_project`, while preserving
   ORT-specific dependency wiring and avoiding OptiX.
2. Make wrappers work for MATLAB/Python through a stable generic inference
   facade.
3. Review implementation for redundancy and readability before adding
   higher-level model support.
4. Add model-role facade for centroiding, object detection, feature matching,
   tracking, optical flow, and future learned modules.
5. Add YOLO and centroiding ONNX examples without hardcoding those models into
   the core ORT backend.
