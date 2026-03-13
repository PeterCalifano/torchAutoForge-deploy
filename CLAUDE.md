# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`torchAutoForge-deploy` is a C++20 library for deploying ML models via ONNX Runtime (ORT). It exposes `CInferenceManager_ORT` (namespace `deploy_ort`) as the central inference class, with optional Python/MATLAB bindings via gtwrap (submodule at `lib/wrap/`).

## Build Commands

The primary interface is `build_lib.sh`:

```bash
./build_lib.sh                        # RelWithDebInfo build, configure + build + test
./build_lib.sh -t debug               # Debug build
./build_lib.sh -t release -i          # Release build + install
./build_lib.sh -r                     # Rebuild only (skip CMake configure)
./build_lib.sh --clean -t debug       # Clean reconfigure
./build_lib.sh -N                     # Use Ninja generator
./build_lib.sh --skip-tests           # Skip test execution
./build_lib.sh -D ENABLE_CUDA=ON      # Pass extra CMake definitions
./build_lib.sh -p                     # Enable Python wrapper (gtwrap)
./build_lib.sh -m                     # Enable MATLAB wrapper (gtwrap)
```

Manual CMake:
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel $(nproc)
ctest --test-dir build --output-on-failure
```

## Running Tests

```bash
# All tests
ctest --test-dir build --output-on-failure -j $(nproc)

# Single test by name pattern
ctest --test-dir build -R CInferenceManager_ORT -V

# List available tests
ctest --test-dir build --show-only
```

Tests use Catch2 v3 (auto-fetched if not found). New test files named `test*.cpp` under `tests/<subdir>/` are auto-discovered. The ORT test requires a real `.onnx` file at a hardcoded path — see `tests/inference/testCInferenceManager_ORT.cpp:6`.

## Key CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `ENABLE_CUDA` | OFF | CUDA support (requires 12.0+) |
| `ENABLE_TESTS` | ON | Build and run tests |
| `SANITIZE_BUILD` | OFF | Enable sanitizers (`address,undefined,leak`) |
| `WARNINGS_ARE_ERRORS` | OFF | `-Werror` |
| `autoforge_deploy_BUILD_PYTHON_WRAPPER` | OFF | gtwrap Python bindings |
| `autoforge_deploy_BUILD_MATLAB_WRAPPER` | OFF | gtwrap MATLAB bindings |

ORT install path is hardcoded in `CMakeLists.txt:13-14` — adjust `onnxruntime_DIR` and `onnxruntime_INCLUDE_DIRS` to your local ORT build.

## Architecture

### Core: `src/inference/onnx_runtime/`

- **`onnxruntime_inference_tools.hpp`** — defines `CInferenceManager_ORT`. Constructors: default, path+bool+SessionOptions (primary), yml config path (stub). Template methods `initialize<T>()` and `infer<T>()` are defined inline in the header.
- **`onnxruntime_inference_tools.cpp`** — constructor implementations and `GetAvailableProviders()`.

**Current state of `CInferenceManager_ORT`:**
- `initialize<T>()`: creates `Ort::Session`, but **does not yet query model metadata** — input/output names and shapes are not populated from the ORT session. `SInputOutputSpecs` is default-constructed (empty).
- `infer<T>()`: calls `session_ptr_->Run(...)` but passes empty names/tensors — unusable until `initialize()` is complete.
- yml constructor: stub only (`// TODO`).
- **Constructor signature mismatch**: the test file calls `CInferenceManager_ORT(model_path, bool)` but the header declares `(bool, model_path)` — this must be resolved.

### Auxiliary: `src/auxiliary/`

- **`common_defs.h`** — `SInputOutputSpecs<INPUT_T, OUTPUT_T>` (names + shapes + element counts) and `SImagesInputOutputSpecs` (image-specialized variant, NCHW convention).
- **`common_ops.h`** — `AccumProduct<T>()`, `CheckFileExists()`, `CheckFileExistsWithExt()` (namespace `deploy_aux`).
- **`images_prepro.h`** — OpenCV-based image preprocessing helpers.

### Programs: `src/programs/`

- `get_available_providers` — lists ORT execution providers at runtime.
- `run_ort_inference` — placeholder runner (not functional yet).

### Wrappers

gtwrap interface files are currently **empty stubs**:
- `src/wrap_interface.i` — top-level, namespace `myspace` (unused)
- `src/inference/inference.i` — namespace `ptaf_deploy` (unused)

Python wrapping is also available directly via pybind11 (`handle_pybind11_wrapper()` in root CMakeLists). gtwrap is the submodule at `lib/wrap/` — see `lib/wrap/CLAUDE.md` for its internals.

### Config Header

`src/config.h.in` → `config.h` (CMake-configured): provides `PrintVersion()` / `GetVersionString()`.

## Compile Flags by Build Type

- **Debug**: `-Og -g -fmax-errors=3`
- **Release**: `-O3 -fmax-errors=1`
- **RelWithDebInfo** (default): `-O2 -g` + `-Wnull-dereference -Wfloat-equal -Wconversion -Wnon-virtual-dtor`

All builds also append `-Wall -Wextra -Wpedantic` from `build_lib.sh`.

## Required Dependencies

- **Eigen3** ≥ 3.4
- **ONNX Runtime** (locally built; set `onnxruntime_DIR` in `CMakeLists.txt`)
- **Optional**: OpenCV, CUDA Toolkit 12.0+, OptiX, gtwrap (`pyparsing` required)

---

## Development Roadmap

### ORT Inference API (`CInferenceManager_ORT`)

1. **Fix constructor signature mismatch** — align header declaration with test usage; decide canonical argument order `(model_path, inplace_init, session_options)`.

2. **Implement model metadata extraction in `initialize<T>()`**:
   - Query input/output count: `session_ptr_->GetInputCount()` / `GetOutputCount()`
   - Query names via `session_ptr_->GetInputNameAllocated(i, allocator_)`
   - Query shapes via `session_ptr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape()`
   - Populate `SInputOutputSpecs` from queried data (replace the default-constructed placeholder)

3. **Design data-bearing `infer()` API** — choose between:
   - `infer(const std::vector<T>& input_data) -> std::vector<T>` (simple, copy-based)
   - `infer(Ort::Value input_tensor) -> Ort::Value` (zero-copy, ORT-native)
   - Overload for `Eigen::MatrixXf` input for ergonomics

4. **Handle dynamic/symbolic dimensions** — ORT shapes may contain `-1` for dynamic axes; `SInputOutputSpecs` must tolerate these at query time and require concrete shapes at run time.

5. **Implement yml config constructor** — parse a YAML session config (model path, provider, thread count) and construct the manager. Consider using a `rapidyaml` or `ryml` header from `lib/header_only/`.

6. **Write complete Catch2 tests** in `tests/inference/testCInferenceManager_ORT.cpp`:
   - Valid model path + initialize → no throw
   - Inference with known dummy model (export a trivial ONNX from Python for testing)
   - Output shape/value correctness vs. Python reference

7. **CUDA EP support** — when `ENABLE_CUDA=ON`, append `OrtCUDAProviderOptions` to `session_options_` in `initialize()`.

### MATLAB/Python Wrapper

1. **Populate `src/inference/inference.i`** — expose `CInferenceManager_ORT` to gtwrap:
   ```cpp
   namespace ptaf_deploy {
     class CInferenceManager_ORT {
       CInferenceManager_ORT(string model_path, bool inplace_init);
       void initialize();
       // infer method once API is stable
       static std::vector<string> GetAvailableProviders();
     };
   }
   ```
   Follow gtwrap syntax rules in `lib/wrap/CLAUDE.md` (uppercase class names, one method per line, fully qualified types, angle-bracket includes).

2. **Wire the interface file into CMake** — in the root `CMakeLists.txt`, pass the interface file explicitly to `handle_gtwrappers()` via `WRAP_INTERFACE_FILES`, or ensure auto-discovery picks up `src/inference/inference.i`.

3. **Build and smoke-test Python wrapper**:
   ```bash
   ./build_lib.sh -p --skip-tests
   # then in python/:
   python3 -c "import autoforge_deploy; m = autoforge_deploy.ptaf_deploy.CInferenceManager_ORT('model.onnx', False)"
   ```

4. **Build and smoke-test MATLAB wrapper**:
   ```bash
   ./build_lib.sh -m --skip-tests
   ```
   Then in MATLAB: load the generated `.m` wrapper and call `CInferenceManager_ORT('model.onnx', false)`. Check `tests/matlab/` for existing fixture patterns.

5. **Validate MATLAB multithreading** — ORT creates internal thread pools; verify no deadlock when called from MATLAB's JVM/MEX thread. Consider setting `session_options_.SetIntraOpNumThreads(1)` as the safe default for MATLAB.
