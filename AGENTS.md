# AGENTS.md

Guidance for Codex when working in `torchAutoForge-deploy`.

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

Primary interface:

```bash
./build_lib.sh                        # RelWithDebInfo build, configure + build + test
./build_lib.sh -t debug               # Debug build
./build_lib.sh -t release -i          # Release build + install
./build_lib.sh -r                     # Rebuild only
./build_lib.sh --clean -t debug       # Clean reconfigure
./build_lib.sh -N                     # Use Ninja generator
./build_lib.sh --skip-tests           # Skip test execution
./build_lib.sh -D ENABLE_CUDA=ON      # Enable CUDA build path
./build_lib.sh -p                     # Enable Python wrapper generation
./build_lib.sh -m                     # Enable MATLAB wrapper generation
```

Manual CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel $(nproc)
ctest --test-dir build --output-on-failure -j $(nproc)
```

## Tests

```bash
ctest --test-dir build --output-on-failure -j $(nproc)
ctest --test-dir build -R CInferenceManager_ORT -V
ctest --test-dir build --show-only
```

Tests use Catch2 v3. New `test*.cpp` files under `tests/<subdir>/` are
auto-discovered.

Current tests cover:

- ORT model metadata extraction;
- ORT inference on `tests/matlab/testSamples/tracedSampleModel.onnx`;
- generic tensor helper behavior;
- facade artifact dispatch;
- TensorRT stub failure behavior.

## Key CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_CUDA` | OFF | CUDA support for future CUDA/ORT provider work |
| `ENABLE_TESTS` | ON | Build and run tests |
| `SANITIZE_BUILD` | OFF | Enable sanitizers |
| `WARNINGS_ARE_ERRORS` | OFF | Add `-Werror` |
| `autoforge_deploy_BUILD_PYTHON_WRAPPER` | OFF | gtwrap Python bindings |
| `autoforge_deploy_BUILD_MATLAB_WRAPPER` | OFF | gtwrap MATLAB bindings |

Use `onnxruntime_DIR` or `ONNXRUNTIME_ROOT` to point CMake at a local ONNX
Runtime install.

## Architecture

### Generic Inference API

- `src/inference/inference_common.h` defines backend/artifact enums, tensor
  descriptors, tensor views, owned buffers, model metadata, and runtime options.
- `src/inference/inference_manager.h/.cpp` provides facade dispatch by artifact
  type.

### ONNX Runtime Backend

- `src/inference/onnx_runtime/onnxruntime_inference_tools.hpp/.cpp` implements
  `CInferenceManager_ORT`.
- `LoadModel()` validates `.onnx`, creates an ORT session, applies provider
  options, and extracts model metadata.
- `Infer()` accepts host-memory `STensorView` inputs and returns owned
  `STensorBuffer` outputs.

### TensorRT Backend

- `src/inference/tensorrt/tensorrt_inference_engine.*` is an explicit
  not-implemented placeholder.
- Do not expose TensorRT as a working backend until real engine loading and
  inference are implemented.

### Wrappers

- `src/wrap_interface.i` and `src/inference/inference.i` are currently stubs.
- Prefer wrapping the generic facade and wrapper-friendly value types.
- Do not expose raw ORT handles, raw `void*`, or `std::byte` storage directly to
  MATLAB/Python.

## Development Priorities

1. Keep CMake/build config synced with
   `/home/peterc/devDir/dev-tools/cpp_cuda_template_project`, while preserving
   ORT-specific dependency wiring and avoiding OptiX.
2. Make wrappers work for MATLAB/Python through a stable generic inference
   facade.
3. Review current implementation for redundancy and readability before adding
   higher-level model support.
4. Add model-role facade for centroiding, object detection, feature matching,
   tracking, optical flow, and future learned modules.
5. Add YOLO and centroiding ONNX examples without hardcoding those models into
   the core ORT backend.
