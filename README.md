# torchAutoForge-deploy

C++/CUDA library for deploying machine learning models via ONNX Runtime (ORT). Provides a `CInferenceManager_ORT` class that wraps ORT sessions with optional CUDA/OptiX GPU acceleration and optional Python/MATLAB bindings via gtwrap.

## Status

**Early development / prototype.** The core inference manager compiles but is not yet fully functional. See [TODO](#todo) for open issues.

## Dependencies

| Dependency | Version | Required |
|------------|---------|----------|
| Eigen3 | ≥ 3.4 | Yes |
| ONNX Runtime | any | Yes |
| OpenCV | any | No (image preprocessing) |
| CUDA Toolkit | ≥ 12.0 | No (GPU acceleration) |
| OptiX | — | No (requires CUDA) |
| gtwrap | — | No (Python/MATLAB bindings) |

If ORT is installed in a non-standard location, set `OnnxRuntime_DIR` before configuring.

## Build

The primary interface is `build_lib.sh`:

```bash
./build_lib.sh                        # RelWithDebInfo: configure + build + test
./build_lib.sh -t debug               # Debug build
./build_lib.sh -t release -i          # Release build + install
./build_lib.sh -r                     # Rebuild only (skip CMake configure)
./build_lib.sh --clean -t debug       # Clean reconfigure
./build_lib.sh -N                     # Use Ninja generator
./build_lib.sh -j 8                   # Parallel jobs
./build_lib.sh --skip-tests           # Skip test execution
./build_lib.sh -D ENABLE_CUDA=ON      # Pass extra CMake definitions
./build_lib.sh -p                     # Enable Python wrapper (requires gtwrap)
./build_lib.sh -m                     # Enable MATLAB wrapper
```

Manual CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
ctest --test-dir build --output-on-failure -j $(nproc)
```

## CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_CUDA` | OFF | CUDA support (requires CUDA 12.0+) |
| `ENABLE_OPTIX` | OFF | OptiX acceleration (requires CUDA) |
| `ENABLE_TESTS` | ON | Build and run tests |
| `SANITIZE_BUILD` | OFF | Enable sanitizers |
| `WARNINGS_ARE_ERRORS` | OFF | `-Werror` |
| `autoforge_deploy_BUILD_PYTHON_WRAPPER` | OFF | gtwrap Python bindings |
| `autoforge_deploy_BUILD_MATLAB_WRAPPER` | OFF | gtwrap MATLAB bindings |

## Tests

```bash
ctest --test-dir build --output-on-failure -j $(nproc)
ctest --test-dir build -R CInferenceManager_ORT -V   # single test
ctest --test-dir build --show-only                   # list tests
```

Tests use Catch2 v3 (auto-fetched if not found). Any `test*.cpp` file placed in a subdirectory under `tests/` is auto-discovered.

## Architecture

```
src/
  inference/
    onnx_runtime/   — CInferenceManager_ORT (core, under development)
    tensorrt/       — CEngineLoader stub (not implemented)
  auxiliary/
    common_defs.h   — SInputOutputSpecs<> tensor I/O descriptor
    common_ops.h    — AccumProduct(), file-check utilities
    images_prepro.h — OpenCV image preprocessing (stub)
  programs/
    get_available_providers  — lists ORT providers at runtime
    run_ort_inference        — placeholder CLI runner
  wrap_interface.i           — gtwrap top-level interface (empty)
  inference/inference.i      — gtwrap inference interface (empty)
  config.h.in                — version header (PrintVersion / GetVersionString)
examples/
  object_detection_yoloV7/  — YOLOv7 object detection example
tests/
  inference/                — C++ tests for CInferenceManager_ORT
  matlab/                   — MATLAB import and inference tests
  simulink/                 — Simulink inference tests
```

---

## TODO

### Core: `CInferenceManager_ORT`

- [ ] Fix constructor argument-order mismatch: header declares `(bool inplace_init, const std::string& model_path, ...)` but `.cpp` implements them in opposite order; test file uses yet another order `(model_path, bool)` — pick one signature and make all sites consistent
- [ ] Implement `initialize()`: currently creates an empty `SInputOutputSpecs` — must query the ORT session for actual input/output names and shapes after session creation
- [ ] Remove hardcoded `initialize<float>()` call in constructor: infer type from model metadata or expose it as a parameter/template argument
- [ ] Fix tensor allocation in `initialize()` (marked `FIXME`): input/output shapes are empty at allocation time; allocation must happen after specs are populated from the session
- [ ] Implement YAML config constructor: the `(const std::string session_config_path)` overload has a `TODO` and parses nothing
- [ ] Implement `infer()` fully: the method runs the session but provides no way to supply input data or retrieve output data — add typed data-in/data-out parameters or buffer accessors

### `SInputOutputSpecs` / `SImagesInputOutputSpecs`

- [ ] Fix dangling `const char*` pointers: `input_names` / `output_names` store `.c_str()` of local `std::string` objects that go out of scope — store owned `std::vector<std::string>` and expose `const char*` views separately
- [ ] Fix `SImagesInputOutputSpecs` shape construction: `this->input_shapes.emplace_back(INPUT_T{batch_size, num_channels, height, width})` is invalid when `INPUT_T = int64_t`; input shape should be a flat `{batch, channels, height, width}` vector of scalars
- [ ] Extend `SInputOutputSpecs` to support multiple input tensors with independent shapes (currently only a flat single-tensor shape vector)

### Programs / CLI

- [ ] Remove hardcoded absolute path from `run_ort_inference.cpp` — use CLI argument or relative path
- [ ] Add CLI argument parsing to `run_ort_inference` (noted in source as `// TODO add tclap for argument parsing`)

### Tests

- [ ] Fix `testCInferenceManager_ORT.cpp` constructor call: `CInferenceManager_ORT(invalid_model_path, false)` does not match any declared constructor signature
- [ ] Implement dummy-value inference test (`CInferenceManager_ORT_infer` section is a stub)
- [ ] Add a cross-validation test: run inference in C++ and compare output numerically against Python/ORT reference values

### Stubs / Not Implemented

- [ ] Implement image preprocessing functions in `images_prepro.h` (namespace is declared but empty)
- [ ] Implement TensorRT backend: `CEngineLoader` in `src/inference/tensorrt/` is an empty class skeleton
- [ ] Populate `src/wrap_interface.i` and `src/inference/inference.i` to expose the inference manager to Python/MATLAB via gtwrap

### Bindings

- [ ] Implement and test Python wrapper via gtwrap or pybind11
- [ ] Implement and test MATLAB wrapper
- [ ] Test MATLAB multi-thread safety (ORT session calls from MATLAB environment)
- [ ] Test end-to-end ORT inference from within MATLAB

### MATLAB Module (see also `matlab/TODO.md`)

- [ ] Implement script to export `.pth`/`.pt` model to ONNX from within MATLAB workflow
- [ ] Investigate Simulink codegen from exported ONNX model

### Housekeeping

- [ ] Replace `DEBUG` preprocessor macro in `onnxruntime_inference_tools.hpp` with a proper logging mechanism; current `printd` logic with `!NDEBUG` is fragile
- [ ] Add `<filesystem>` include directly to `onnxruntime_inference_tools.hpp` (currently relies on transitive include from `common_ops.h`)
- [ ] Add `.gitignore` entry or clean up committed `build/` artifacts inside `examples/object_detection_yoloV7/build/`
- [ ] Complete YOLOv7 example: `object_detection_yoloV7.h` is nearly empty; end-to-end pipeline (preprocess → infer → NMS → draw) is unfinished
