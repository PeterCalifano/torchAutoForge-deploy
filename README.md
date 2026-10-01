# torchAutoForge-deploy

C++20 library for running exported ML models in production code. It loads ONNX
models through ONNX Runtime (ORT) or serialized TensorRT engines through one
backend-agnostic API, and exposes the same API to Python, MATLAB, and ROS 2.

A model is described by a `.ptafmodel` JSON manifest: artifact path, task role,
pre/postprocessing contract, execution targets, and fallback policy. The C++
library, the wrappers, the CLI tools, and the ROS 2 node all read the same
manifest, so switching a model from CPU to CUDA or TensorRT is a configuration
change.

## Features

- **One inference API, two backends.** `CInferenceManager` dispatches to ORT
  (`.onnx`) or, when built with TensorRT, to standalone `.engine` / `.plan`
  files.
- **Role-level models.** `CModelFacade` adds task contracts (raw tensor,
  centroiding, object detection) on top of raw tensor inference. Feature
  matching, tracking, and optical flow are reserved roles.
- **Validated manifests.** `.ptafmodel` files follow a versioned
  [JSON schema](schemas/ptafmodel.schema.json). `make-ptafmodel` creates and
  validates them.
- **Execution target selection.** Ordered target priority (CPU, CUDA, TensorRT
  EP) with optional fallback, device ID, and thread counts.
- **Python and MATLAB bindings** generated with gtwrap, without exposing raw
  backend handles.
- **ROS 2 Jazzy overlay** with a lifecycle node that serves inference as a
  service.
- **CLI tools** for one-shot inference, benchmarking, provider listing, and
  evaluation.
- **Shared tensor helpers** for HWC-to-NCHW conversion, centroid extraction, and
  YOLO-style raw detection decoding.
- **Tested on x86_64 and arm64** in CI. Jetson CUDA/TensorRT paths are run
  manually on target boards.

## Quick Start

Build and test with the default RelWithDebInfo configuration. ONNX Runtime must
be installed first (see [Dependencies](#dependencies)).

```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime
./build_lib.sh
```

Run a model once on a zero-filled input:

```bash
./build/src/programs/run_ort_inference model.onnx \
  --shape 1,3,640,640 --fill 0 --targets cpu --no-fallback
```

Load a manifest from C++:

```cpp
#include <inference/model_facade.h>

ptafdeploy::inference::CModelFacade model;
model.LoadModelConfig("model.ptafmodel");

const std::vector<int64_t> shape{1, 3, 640, 640};
std::vector<float> input(1 * 3 * 640 * 640, 0.0F);
const std::vector<float> output = model.InferSingleFloatInput(input, shape);
```

The same calls from Python, after building with `./build_lib.sh -p`:

```python
import autoforge_deploy as ptaf

model = ptaf.CModelFacade()
model.LoadModelConfig("model.ptafmodel")
output = model.InferSingleFloatInput(input_values, [1, 3, 640, 640])
```

Link the installed library from another CMake project:

```cmake
find_package(autoforge_deploy CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE autoforge_deploy::autoforge_deploy)
```

Complete examples live in [`examples/`](examples/): a
[YOLOv7 detector](examples/object_detection_yoloV7/) and
[centroiding models](examples/centroiding_models/), each with C++, Python, and
MATLAB entry points.

## Dependencies

| Dependency | Version | Required | Notes |
|------------|---------|----------|-------|
| Eigen3 | >= 3.4 | Yes | Core C++ dependency |
| ONNX Runtime | local install | Yes | CMake package required |
| OpenCV | local install | Examples only | YOLO and centroiding image IO/drawing |
| CUDA Toolkit | >= 12.0 | Optional | CUDA/PTX builds and TensorRT runtime prerequisites |
| TensorRT | local install | Optional | Standalone `.engine` / `.plan` runtime |
| gtwrap | local or installed | Optional | Python/MATLAB wrappers |

CMake finds ONNX Runtime in this order:

1. explicit `onnxruntime_DIR`;
2. CMake cache `ONNXRUNTIME_ROOT`;
3. environment `ONNXRUNTIME_ROOT`;
4. normal CMake config-package discovery.

`ONNXRUNTIME_ROOT` must contain
`lib/cmake/onnxruntime/onnxruntimeConfig.cmake`.

The CUDA/PTX build option is independent of TensorRT. The TensorRT backend may
link `CUDA::cudart` without enabling the project CUDA language. ORT GPU
execution requires an ONNX Runtime build with the CUDA or TensorRT execution
provider and its runtime dependencies. Enabling project CUDA compilation does
not supply those providers.

OptiX is not a dependency of this repository.

## Build

```bash
./build_lib.sh                        # RelWithDebInfo: configure + build + test
./build_lib.sh -t debug               # Debug build
./build_lib.sh -t release -i          # Release build + install
./build_lib.sh -r                     # Rebuild only, skip configure
./build_lib.sh --clean -t debug       # Clean reconfigure
./build_lib.sh -N                     # Use Ninja generator
./build_lib.sh -j 8                   # Parallel jobs
./build_lib.sh --skip-tests           # Skip test execution
./build_lib.sh -D autoforge_deploy_ENABLE_CUDA=ON
./build_lib.sh -D autoforge_deploy_ENABLE_TENSORRT=ON \
  -D TENSORRT_ROOT=/path/to/TensorRT
./build_lib.sh -p                     # Enable Python wrapper generation
./build_lib.sh -m                     # Enable MATLAB wrapper generation
./build_lib.sh -p --wrap-update       # Explicitly update the gtwrap checkout
```

Manual CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel $(nproc)
ctest --test-dir build --output-on-failure -j $(nproc)
```

See [`doc/build_script_doc.md`](doc/build_script_doc.md) for every
`build_lib.sh` option.

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `autoforge_deploy_ENABLE_CUDA` | OFF | Enable CUDA language and CUDA compile interface |
| `CUDA_PTX_USE_FAST_MATH` | ON | Use NVCC fast math for dedicated PTX sources |
| `CUDA_PTX_NVCC_FLAGS` | empty | Extra flags used only for PTX generation |
| `autoforge_deploy_ENABLE_TENSORRT` | OFF | Enable standalone TensorRT engine backend |
| `TensorRT_ROOT` / `TENSORRT_ROOT` | empty | Preferred and compatibility TensorRT installation roots |
| `ENABLE_TESTS` | ON | Build and run Catch2 tests |
| `SANITIZE_BUILD` | OFF | Enable configured sanitizers |
| `WARNINGS_ARE_ERRORS` | OFF | Add `-Werror` |
| `autoforge_deploy_METADATA_ONLY` | OFF | Resolve project identity/version without languages or dependencies |
| `WRITE_SOURCE_VERSION_FILE` | OFF | Opt in to updating source `VERSION` during configure |
| `LIB_TARGET_NAME_OVERRIDE` | empty | Override the physical target while retaining the canonical export |
| `autoforge_deploy_BUILD_PROGRAMS` | ON | Build command-line programs |
| `autoforge_deploy_BUILD_EXAMPLES` | ON | Build examples |
| `autoforge_deploy_BUILD_PYTHON_WRAPPER` | OFF | Build gtwrap Python bindings |
| `autoforge_deploy_BUILD_MATLAB_WRAPPER` | OFF | Build gtwrap MATLAB bindings |
| `autoforge_deploy_WRAPPER_INTERFACE_FILES` | `src/wrap_interface.i` | Ordered gtwrap interface files |
| `GTWRAP_MAINTENANCE_UPDATE` | OFF | Explicitly authorize wrapper-checkout updates |
| `GTWRAP_SYNC_TO_MASTER` | OFF | Request an update; requires maintenance authorization |

## Model Manifests

A minimal `.ptafmodel`:

```json
{
  "schema_version": 1,
  "artifact_path": "model.onnx",
  "task": "centroiding",
  "preprocessing": "opencv_grayscale_resize_linear_nchw_unit_scale",
  "postprocessing": "normalized_centroid_xy",
  "execution_target_priority": ["cpu"],
  "allow_fallback": false
}
```

Create one with
`make-ptafmodel init model.onnx --task centroiding --output model.ptafmodel`, or
run the installed `make_ptafmodel.py` wizard. Both use the same native writer
and validator. The [manifest schema reference](doc/development/ptafmodel_manifest_schema.md)
covers fields, defaults, templates, and validation commands.
[`examples/model_configs/`](examples/model_configs/) has working manifests.

## Command-Line Tools

| Program | Purpose |
|---------|---------|
| `run_ort_inference` | One inference on prepared float32 tensors, with metadata inspection and raw output files |
| `benchmark_model` | Warmup and timed iterations through the facade |
| `get_available_providers` | List ORT execution providers, or fail if required targets are missing |
| `make-ptafmodel` | Create and validate `.ptafmodel` manifests |
| `evaluate_inference` / `prepare_image_sequence` | Optional. Prepare image selections and summarize timing or point-error reports |

`run_ort_inference` accepts repeated `--input [name=]path.f32`,
`--shape [name=]d0,d1,...`, and `--fill [name=]value` options for multi-input
models. `--metadata-only` inspects the model without tensor data, and
`--output-dir` writes raw float32 outputs. See
[`doc/run_ort_inference.md`](doc/run_ort_inference.md) for the binary format,
naming rules, runtime options, and troubleshooting. See
[`doc/native_evaluation.md`](doc/native_evaluation.md) for the evaluation tools
and [`doc/image_and_inference_output.md`](doc/image_and_inference_output.md) for
the optional image annotation and JSON output utilities.

## Python And MATLAB Wrappers

`autoforge_deploy` is the Python distribution and import name. The wheel ships
the extension next to `libautoforge_deploy`. ONNX Runtime must be installed
separately.

Ordinary configuration does not modify the wrapper checkout. `./build_lib.sh -p`
or `-m` uses the checked-out `lib/wrap` and does not fetch, advance, or
initialize a submodule. Request checkout maintenance explicitly:

```bash
./build_lib.sh -p --wrap-update          # update an existing resolved checkout
./build_lib.sh -p --wrap-submodule-init  # initialize a declared fallback
```

Python wrapping assembles metadata, package sources, the extension, and
`_wrapper_build.py` under `<build>/python`. Build-only link metadata is excluded
from wheels.

For MATLAB, pass `Matlab_ROOT_DIR` to CMake or export `MATLAB_ROOT_DIR`. The
MATLAB smoke test covers the facade, model loading, inference, error recovery,
and MEX teardown.

`python/autoforgeDeployPy` is legacy code kept until the
[Python consolidation plan](doc/development/model_export_and_python_consolidation_plan.md)
is complete.

## ROS 2 Jazzy Overlay

The optional overlay under `ros2/` contains four packages:

- `ptafdeploy`: colcon shim for the core library;
- `ptafdeploy_interfaces`: float-tensor service and status messages;
- `ptafdeploy_ros`: lifecycle component/executable backed by `CModelFacade`;
- `ptafdeploy_spinup`: installed model fixture and launch files.

```bash
./build_ros2.sh --clean \
  --cmake-arg -Donnxruntime_DIR=/path/to/lib/cmake/onnxruntime
source ros2/install/setup.bash
ros2 launch ptafdeploy_spinup ptafdeploy.launch.py
```

The composed launch is `ptafdeploy_composition.launch.py`. Both launch files
accept `autostart`, `namespace`, `node_name`, and `model_config_path`. The
lifecycle node loads its `.ptafmodel` during configure, serves `~/infer` only
while active, and publishes `~/status`. See
[`doc/ros2_overlay.md`](doc/ros2_overlay.md) for the full contract.

## Benchmarks And Jetson

CPU smoke benchmark:

```bash
./build/src/programs/benchmark_model \
  tests/inference/model_configs/centroiding_fixture.ptafmodel \
  --iterations 20 --warmup 3
```

CUDA benchmark, when the installed ONNX Runtime has the CUDA execution
provider:

```bash
./build/src/programs/get_available_providers --require-targets cuda

scripts/run_jetson_runtime_smoke.py --build-dir build \
  --ort-model examples/model_configs/yolov7_640x640.ptafmodel \
  --device 0 --iterations 50 --warmup 5

# Equivalent direct benchmark_model call:
./build/src/programs/benchmark_model \
  examples/model_configs/yolov7_640x640.ptafmodel \
  --targets cuda --device 0 --no-fallback --iterations 50 --warmup 5
```

Standalone TensorRT engine benchmark:

```bash
cmake -S . -B build-trt \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -Dautoforge_deploy_ENABLE_TENSORRT=ON \
  -DTENSORRT_ROOT=/usr/local/tensorrt
cmake --build build-trt --parallel $(nproc)

trtexec --onnx=model.onnx --saveEngine=model.engine --skipInference

LD_LIBRARY_PATH=/usr/local/tensorrt/lib:$LD_LIBRARY_PATH \
scripts/run_jetson_runtime_smoke.py --build-dir build-trt \
  --trt-engine model.engine \
  --device 0 --iterations 50 --warmup 5 \
  -- --trt-profile 0
```

ORT TensorRT Execution Provider benchmark, when the installed ORT build exposes
`TensorrtExecutionProvider`:

```bash
scripts/run_jetson_runtime_smoke.py --build-dir build \
  --ort-model model.onnx \
  --with-tensorrt-ep \
  --device 0 --iterations 50 --warmup 5 \
  -- --task raw_tensor --input-shape 1,3,640,640
```

Jetson notes:

- Use the TensorRT and CUDA versions installed by JetPack. Set `TENSORRT_ROOT`
  only if TensorRT is outside the standard JetPack paths.
- Build engines on the target Jetson or a GPU with the same SM architecture.
  TensorRT engines are not portable across GPU architectures.
- Use `--trt-profile` for engines with multiple optimization profiles.
- Use `--device 0` on typical Jetson boards.
- DLA selection is not exposed or qualified.
- `scripts/run_jetson_runtime_smoke.py --dry-run ...` runs in CTest, so
  argument handling is checked without Jetson hardware.

## Tests

```bash
ctest --test-dir build --output-on-failure -j $(nproc)
ctest --test-dir build -R CInferenceManager_ORT -V
ctest --test-dir build --show-only
```

Tests use Catch2 v3 and cover:

- ORT model metadata extraction and host-buffer inference on the checked-in
  ONNX fixture;
- tensor helper byte counts and dynamic shapes;
- facade artifact dispatch and model-role behavior;
- TensorRT load errors when the backend is disabled or the engine path is
  invalid;
- TensorRT engine inference when `autoforge_deploy_ENABLE_TENSORRT=ON` and
  `PTAFDEPLOY_TENSORRT_TEST_ENGINE` points to a generated `.engine`;
- `run_ort_inference` metadata, input validation, and real inference;
- the Python wrapper smoke test and the ROS 2 lifecycle bridge.

## Architecture

```text
src/
  inference/
    inference_common.h      - generic tensor descriptors, views, buffers, options
    inference_manager.*     - backend-agnostic facade
    model_facade.*          - role-level model contract facade
    task_value_types.*      - generic wrapper-safe task values
    task_adapters.*         - schema-driven tensor conversion and row decoding
    onnx_runtime/           - CInferenceManager_ORT backend
    tensorrt/               - optional standalone TensorRT engine backend
  programs/                 - CLI tools listed above
  utils/
    filesystem.h            - file-path validation
    parsing/value_parsing.* - strict reusable textual-value parsing
    logging/CLogger.*       - dependency-free project logger
    images/                 - optional OpenCV image operations
    inference_output/       - optional native tensor JSON and report publication
  wrap_interface.i          - gtwrap top-level interface
  inference/inference.i     - gtwrap inference facade/value interface
schemas/
  ptafmodel.schema.json     - manifest JSON schema
examples/model_configs/
  *.ptafmodel               - versioned manifests with repo-relative artifact paths
ros2/
  ptafdeploy*               - optional lifecycle inference overlay
```

`CInferenceManager` is the backend-agnostic tensor API. `CModelFacade` adds
role-level contracts for C++, Python, and MATLAB. Model-specific schemas and
result interpretation belong in integrations above raw tensor inference, not in
the ORT backend.

The logger lives in `ptafdeploy::logging` and has no external dependencies. Set
its default threshold with `PTAFDEPLOY_LOG_LEVEL`.

### CUDA And Embedded PTX

With `autoforge_deploy_ENABLE_CUDA=ON`,
`src/template_src_kernels/placeholder.cu/.cuh` is an ordinary CUDA source
example. `placeholder_to_ptx.ptx.cu` is compiled separately to PTX, converted
to an embedded C byte array with CUDA `bin2c`, and linked into the library.
Dedicated `.ptx.cu` files never enter ordinary CUDA compilation. The library
embeds PTX but does not provide a CUDA Driver PTX loader.

## Install, Package, And Docs

```bash
cmake --preset docs
cmake --build --preset docs --target doc

cmake -S . -B build-package \
  -Donnxruntime_DIR=/path/to/lib/cmake/onnxruntime
cmake --build build-package --parallel
cpack --config build-package/CPackConfig.cmake
cpack --config build-package/CPackSourceConfig.cmake
```

Doxygen XML can feed gtwrap docstring generation. The build-tree `VERSION`,
CPack filenames, and Python package metadata all use the Git-derived
`FULL_VERSION`. The ignored source-tree `VERSION` is a fallback for builds
without Git and cannot overwrite the package version.

## Developer Containers

`.devcontainer` supports CPU or CUDA 12.9 profiles, optional ROS 2 build args,
and Docker or Podman GPU passthrough:

```bash
./configure_devcontainer.sh --no-cuda --no-ros
./configure_devcontainer.sh --cuda --gpu-runtime docker --ros2 jazzy
./run_in_container.sh --build -- \
  ./build_lib.sh -D autoforge_deploy_ENABLE_CUDA=ON
./run_in_container.sh --vscode --engine podman
./run_in_container.sh --matlab-root /path/to/MATLAB/R2023b -- ./build_lib.sh -m
```

The launcher runs with the host user and group IDs, so bind-mounted outputs stay
owned by the host user. `--vscode` starts a persistent container for VS Code to
attach to. `--matlab-root` mounts a MATLAB installation read-only at the same
absolute path and exports `MATLAB_ROOT_DIR`.

## CI

GitHub Actions runs separate native, documentation, CUDA/TensorRT, and ROS 2
workflows:

- **Native x86_64 and arm64** (`ubuntu-22.04`, `ubuntu-22.04-arm`): configure,
  build, run CTest, and run the CPU ORT provider check. The arm64 job catches
  ARM build and host-inference regressions without Jetson hardware.
- **Python wrapper** (`ubuntu-22.04`): builds the gtwrap Python module and runs
  `autoforge_deploy_python_facade_smoke`.
- **Docs**: builds Doxygen HTML/XML and uploads it as a Pages artifact. The
  deploy job is disabled (`if: ${{ false }}`), so the site is not published.
- **CUDA/TensorRT**: manual, self-hosted, requires `develop` and the repository
  variable `CI_USE_SELF_HOSTED=true`. Builds CUDA, embedded PTX, and the
  TensorRT backend together.
- **ROS 2 Jazzy** (Ubuntu 24.04): installs ONNX Runtime and runs the overlay
  lifecycle and launch tests.

Native and docs jobs cache ONNX Runtime downloads by OS, architecture, and ORT
version. They do not cache build trees. Native, docs, and ROS workflows run on
pushes to `main` and on pull requests targeting `main` or `develop`, and can be
dispatched manually (native manual runs require `develop`).

Hosted CI does not cover Jetson CUDA/TensorRT hardware or MATLAB. Run those
locally.

## Status And Limitations

Open items:

- Python/MATLAB APIs above the facade are minimal.
- The TensorRT backend supports host-staged dense tensor IO and runtime
  optimization-profile selection. Dynamic-output allocators and DLA are not
  qualified. Precision is set when the engine is built.
- NMS and alternate detector policies are pending.
- Jetson hardware validation is pending.

The [upgrade plan](doc/development/autoforge_deploy_upgrade_plan.md) tracks
staged work and records local validation of ORT CUDA selection, fallback when
the TensorRT EP is unavailable, and standalone TensorRT inference. Those results
apply only to the revisions and conditions recorded there.

## License

[MIT](LICENSE)
