# torchAutoForge-deploy

C++20 library for deploying machine-learning models through ONNX Runtime (ORT),
with optional CUDA execution-provider support and planned Python/MATLAB bindings
through gtwrap.

## Status

Prototype, but no longer an empty ORT stub. Current local implementation can:

- load an `.onnx` model with `deploy_ort::CInferenceManager_ORT`;
- query ORT model input/output names, element types, and shapes;
- run host-memory tensor inference through `deploy_infer::STensorView`;
- return owned output buffers through `deploy_infer::STensorBuffer`;
- dispatch through the generic `deploy_infer::CInferenceManager` facade for
  `.onnx` artifacts.

Still open:

- Python/MATLAB wrapper interfaces are not populated yet.
- TensorRT standalone backend is an explicit not-implemented stub.
- `run_ort_inference` is still a metadata probe with a hardcoded path.
- Higher-level model roles for centroiding, object detection, feature matching,
  tracking, and optical flow are planned but not implemented.

See `doc/development/autoforge_deploy_upgrade_plan.md` for the staged upgrade
plan.

## Dependencies

| Dependency | Version | Required | Notes |
|------------|---------|----------|-------|
| Eigen3 | >= 3.4 | Yes | Core C++ dependency |
| ONNX Runtime | local install | Yes | CMake package required |
| OpenCV | any | Optional | Image preprocessing/model adapters |
| CUDA Toolkit | >= 12.0 | Optional | ORT CUDA provider support |
| gtwrap | local or installed | Optional | Python/MATLAB wrappers |

OptiX is not part of this repo scope. Keep OptiX support in renderer/template
repos, not in `torchAutoForge-deploy`.

If ORT is installed in a non-standard location, set `onnxruntime_DIR` or
`ONNXRUNTIME_ROOT` before configuring.

## Build

Primary entrypoint:

```bash
./build_lib.sh                        # RelWithDebInfo: configure + build + test
./build_lib.sh -t debug               # Debug build
./build_lib.sh -t release -i          # Release build + install
./build_lib.sh -r                     # Rebuild only, skip configure
./build_lib.sh --clean -t debug       # Clean reconfigure
./build_lib.sh -N                     # Use Ninja generator
./build_lib.sh -j 8                   # Parallel jobs
./build_lib.sh --skip-tests           # Skip test execution
./build_lib.sh -D ENABLE_CUDA=ON      # Enable CUDA language/provider plumbing
./build_lib.sh -p                     # Enable Python wrapper generation
./build_lib.sh -m                     # Enable MATLAB wrapper generation
```

Manual CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel $(nproc)
ctest --test-dir build --output-on-failure -j $(nproc)
```

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_CUDA` | OFF | Enable CUDA language and CUDA compile interface |
| `ENABLE_TESTS` | ON | Build and run Catch2 tests |
| `SANITIZE_BUILD` | OFF | Enable configured sanitizers |
| `WARNINGS_ARE_ERRORS` | OFF | Add `-Werror` |
| `autoforge_deploy_BUILD_PYTHON_WRAPPER` | OFF | Build gtwrap Python bindings |
| `autoforge_deploy_BUILD_MATLAB_WRAPPER` | OFF | Build gtwrap MATLAB bindings |
| `autoforge_deploy_WRAPPER_INTERFACE_FILES` | `src/wrap_interface.i` | Ordered gtwrap interface files |

## Tests

```bash
ctest --test-dir build --output-on-failure -j $(nproc)
ctest --test-dir build -R CInferenceManager_ORT -V
ctest --test-dir build --show-only
```

Current configured tests cover:

- ORT model metadata extraction;
- ORT host-buffer inference on the checked-in ONNX fixture;
- tensor helper byte-count/dynamic-shape behavior;
- facade artifact dispatch;
- TensorRT backend stub failure mode.

## Architecture

```text
src/
  inference/
    inference_common.h      - generic tensor descriptors, views, buffers, options
    inference_manager.*     - backend-agnostic facade
    onnx_runtime/           - CInferenceManager_ORT backend
    tensorrt/               - explicit not-implemented TensorRT stub
  auxiliary/
    common_defs.h           - legacy tensor/image specs
    common_ops.h            - file checks and small utilities
    images_prepro.h         - image preprocessing helpers
  programs/
    get_available_providers - lists ORT providers
    run_ort_inference       - placeholder metadata probe
  wrap_interface.i          - gtwrap top-level interface, pending population
  inference/inference.i     - gtwrap inference interface, pending population
```

Public direction:

- `deploy_infer::CInferenceManager` should become the stable facade.
- `deploy_ort::CInferenceManager_ORT` should remain backend-specific.
- Higher-level model facades should sit above raw tensor inference, so YOLO,
  centroiding, feature matching, tracking, optical flow, and future models can
  share infrastructure without hardcoded model-specific branches in core ORT code.

## Current Gaps

- Wrapper interfaces need a stable MATLAB/Python-safe API surface.
- CLI needs model path and runtime options instead of hardcoded paths.
- Legacy `SInputOutputSpecs` / image-specific specs need review against
  `STensorDescriptor` and friends.
- Numeric inference tests should compare C++ ORT outputs against reference
  values.
- MATLAB wrapper smoke tests must prove model loading and inference inside
  MATLAB.
