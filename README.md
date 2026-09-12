# torchAutoForge-deploy

C++20 library for deploying machine-learning models through ONNX Runtime (ORT)
and optional TensorRT engine runtime, with Python/MATLAB bindings through gtwrap.

## Status

The prototype supports:

- ONNX loading, tensor metadata queries, and host-memory inference through
  `ptafdeploy::inference::CInferenceManager` and its ORT backend,
  `ptafdeploy::inference::onnxruntime::CInferenceManager_ORT`;
- backend-neutral inputs and owned outputs through
  `ptafdeploy::inference::STensorView` and `STensorBuffer`;
- role-level contracts through `ptafdeploy::inference::CModelFacade`, with
  enum-backed runtime settings and validated `.ptafmodel` manifests;
- Python/MATLAB bindings for model loading, metadata, runtime target selection,
  and single-input float inference, without exposing raw backend handles;
- shared HWC-to-NCHW conversion, centroid extraction, and YOLO-style raw detection
  decoding;
- one-shot inference on prepared float32 tensors with `run_ort_inference`, including
  dynamic shapes and raw output files, and timing through `benchmark_model`;
- serialized `.engine` / `.plan` inference through the same facade when built with
  `autoforge_deploy_ENABLE_TENSORRT=ON`.

Still open:

- Higher-level Python/MATLAB convenience APIs are still minimal.
- TensorRT standalone currently supports host-staged dense tensor IO for
  serialized engines plus runtime optimization-profile selection. Dynamic-output
  allocators and DLA execution are not qualified; precision remains an
  engine-build concern.
- Initial higher-level model roles exist for raw tensor, centroiding, and object
  detection; feature matching, tracking, and optical flow are reserved roles.

See `doc/development/autoforge_deploy_upgrade_plan.md` for the staged upgrade
plan. See `doc/development/ptafmodel_manifest_schema.md` for the versioned
`.ptafmodel` manifest schema.

## Dependencies

| Dependency | Version | Required | Notes |
|------------|---------|----------|-------|
| Eigen3 | >= 3.4 | Yes | Core C++ dependency |
| ONNX Runtime | local install | Yes | CMake package required |
| OpenCV | local install | Example-only | YOLO and centroiding image IO/drawing |
| CUDA Toolkit | >= 12.0 | Optional | CUDA/PTX builds and TensorRT runtime prerequisites |
| TensorRT | local install | Optional | Standalone `.engine` / `.plan` runtime |
| gtwrap | local or installed | Optional | Python/MATLAB wrappers |

OptiX support belongs to the renderer/template repositories and is excluded here.

ONNX Runtime discovery uses this precedence:

1. explicit `onnxruntime_DIR`;
2. CMake cache `ONNXRUNTIME_ROOT`;
3. environment `ONNXRUNTIME_ROOT`;
4. normal CMake config-package discovery.

`ONNXRUNTIME_ROOT` is expected to contain
`lib/cmake/onnxruntime/onnxruntimeConfig.cmake`.

The CUDA/PTX build option is independent of TensorRT. The TensorRT backend may
link `CUDA::cudart` without enabling the project CUDA language. ORT GPU execution
requires a compatible provider-enabled ONNX Runtime installation and its runtime
dependencies; enabling project CUDA compilation does not supply those providers.

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

## CMake Options

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

## Tests

```bash
ctest --test-dir build --output-on-failure -j $(nproc)
ctest --test-dir build -R CInferenceManager_ORT -V
ctest --test-dir build --show-only
```

The test suite covers:

- ORT model metadata extraction;
- ORT host-buffer inference on the checked-in ONNX fixture;
- tensor helper byte-count/dynamic-shape behavior;
- facade artifact dispatch;
- TensorRT load errors when the backend is disabled or the engine path is invalid;
- optional TensorRT engine inference when
  `autoforge_deploy_ENABLE_TENSORRT=ON` and
  `PTAFDEPLOY_TENSORRT_TEST_ENGINE` points to a generated `.engine`;
- one-shot ONNX CLI metadata, input-source validation, and real inference.

Permanent tests in this derived repository cover inference, wrappers, programs,
or ROS behavior. The template repository owns tests of shared CMake, logger,
and PTX mechanics.
Validate their integration here with explicit configure/build/package/consumer
checks rather than permanent copies of the donor tests.

## One-Shot ONNX Inference

`run_ort_inference` accepts already-prepared float32 tensors and runs exactly one
inference. For a model accepting shape `[1,3,640,640]`, run a zero-filled smoke test:

```bash
./build/src/programs/run_ort_inference model.onnx \
  --shape 1,3,640,640 --fill 0 --targets cpu --no-fallback
```

Use repeated `--input [name=]path.f32`, `--shape [name=]d0,d1,...`, and
`--fill [name=]value` options for multi-input models. `--metadata-only` inspects
the model without requiring tensor data, and `--output-dir` writes raw float32
outputs in addition to the bounded stdout preview. See
[`doc/run_ort_inference.md`](doc/run_ort_inference.md) for the complete binary
format, naming rules, runtime options, output contract, and troubleshooting.

## CUDA And Embedded PTX Examples

With `autoforge_deploy_ENABLE_CUDA=ON`,
`src/template_src_kernels/placeholder.cu/.cuh` remains an ordinary CUDA source
example. `placeholder_to_ptx.ptx.cu` is compiled separately to PTX, converted
to an embedded C byte array with CUDA `bin2c`, and linked into the library.
Dedicated `.ptx.cu` files never enter ordinary CUDA compilation.

This is compile-and-embed support only. The library does not provide a CUDA
Driver PTX loader, and OptiX is neither an option nor a dependency.

## Install, Package, Docs, And Logger

The installed consumer target is:

```cmake
find_package(autoforge_deploy CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE autoforge_deploy::autoforge_deploy)
```

Build documentation and packages with:

```bash
cmake --preset docs
cmake --build --preset docs --target doc

cmake -S . -B build-package \
  -Donnxruntime_DIR=/path/to/lib/cmake/onnxruntime
cmake --build build-package --parallel
cpack --config build-package/CPackConfig.cmake
cpack --config build-package/CPackSourceConfig.cmake
```

Doxygen XML can be connected to gtwrap docstring generation. The public
dependency-free logger lives in `ptafdeploy::logging`; set its default threshold
with `PTAFDEPLOY_LOG_LEVEL`.

The generated build-tree `VERSION` and CPack filenames use the Git-derived
`FULL_VERSION`; generated Python packaging metadata takes its version from
the same value. The ignored source-tree `VERSION` remains a no-Git fallback
and cannot overwrite the authoritative package value.

## Python And MATLAB Wrappers

`autoforge_deploy` is the sole active Python distribution and import identity.
`python/autoforgeDeployPy` is protected legacy material pending the approved
[Python consolidation plan](doc/development/model_export_and_python_consolidation_plan.md).

Wrapper resolution is read-only during ordinary configuration. The default
`./build_lib.sh -p` or `-m` command may use the checked-out `lib/wrap`, but it
does not fetch, advance, initialize, or add a submodule. Request checkout
maintenance explicitly:

```bash
./build_lib.sh -p --wrap-update          # update an existing resolved checkout
./build_lib.sh -p --wrap-submodule-init  # initialize a declared fallback
```

The Python wheel co-locates the extension with `libautoforge_deploy`; ONNX
Runtime must be installed separately.
Python wrapping assembles metadata, package sources, the extension, and
`_wrapper_build.py` under `<build>/python`; the checkout remains unchanged and
the build-only link metadata is excluded from wheels.

For MATLAB, pass `Matlab_ROOT_DIR` to CMake or export `MATLAB_ROOT_DIR`. The
wrapper smoke exercises this repository's generic facade, model loading,
inference, error recovery, and MEX teardown.

## ROS 2 Jazzy Overlay

The optional overlay under `ros2/` contains four packages:

- `ptafdeploy`: colcon shim for the core library;
- `ptafdeploy_interfaces`: float-tensor service and status messages;
- `ptafdeploy_ros`: lifecycle component/executable backed by `CModelFacade`;
- `ptafdeploy_spinup`: installed model fixture and launch files.

Build and run the standalone launch:

```bash
./build_ros2.sh --clean \
  --cmake-arg -Donnxruntime_DIR=/path/to/lib/cmake/onnxruntime
source ros2/install/setup.bash
ros2 launch ptafdeploy_spinup ptafdeploy.launch.py
```

The composed launch is `ptafdeploy_composition.launch.py`. Both accept
`autostart`, `namespace`, `node_name`, and `model_config_path`. The lifecycle
node loads its `.ptafmodel` during configure, serves `~/infer` only while
active, and publishes `~/status`. See
`doc/ros2_overlay.md` for the complete contract.

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

The standalone launcher runs with the host numeric identity so bind-mounted
outputs remain host-owned. Its VS Code mode starts a persistent attachment
container, and an optional MATLAB installation is mounted read-only at the same
absolute path while exporting `MATLAB_ROOT_DIR`.

## CI And API Docs

GitHub Actions separates native, documentation, CUDA/TensorRT, and ROS 2
responsibilities. Native jobs use Ubuntu 22.04 runners for x86_64 and arm64:

- `ubuntu-22.04`: configure, build, run default CTest, and run CPU ORT provider
  preflight.
- `ubuntu-22.04-arm`: same runnable CPU subset on arm64. This catches ARM build
  and host-inference regressions without requiring Jetson GPU hardware in CI.
- `ubuntu-22.04` Python wrapper job: builds the gtwrap Python module and runs
  the `autoforge_deploy_python_facade_smoke` CTest.
- Native and documentation jobs cache downloaded ONNX Runtime packages by OS,
  architecture, and ORT version. They do not cache CMake build trees.
- Doxygen HTML/XML is built separately and uploaded as a Pages artifact. The
  deployment job is currently disabled (`if: ${{ false }}`); an artifact upload
  does not publish the site.
- A manual guarded self-hosted workflow builds ordinary CUDA plus embedded PTX
  and the TensorRT backend together when repository variable
  `CI_USE_SELF_HOSTED=true`.
- Ubuntu 24.04/ROS 2 Jazzy CI installs ONNX Runtime and runs the real overlay
  lifecycle and launch tests.

Hosted CI does not validate Jetson CUDA/TensorRT hardware paths. Use the Jetson
benchmark commands below on target boards. MATLAB wrapper tests are not part of
hosted CI; run them locally where MATLAB is licensed and installed.

Native/docs/ROS workflows run on `main` pushes and pull requests targeting
`main` or `develop`. They can also be dispatched manually; native manual jobs
require `develop`.
CUDA/TensorRT is manual-only and additionally requires `develop` and the
self-hosted enable variable. Workflow definitions do not establish that a
particular revision has passed CI.

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
  programs/
    get_available_providers - lists ORT providers
    run_ort_inference       - one-shot prepared-tensor ONNX inference
    benchmark_model         - generic facade benchmark path
  utils/
    filesystem.h            - file-path validation
    parsing/value_parsing.* - strict reusable textual-value parsing
    logging/CLogger.*       - dependency-free project logger
    images/                 - optional OpenCV image operations
    inference_output/       - optional native tensor JSON and report publication
  wrap_interface.i          - gtwrap top-level interface
  inference/inference.i     - gtwrap inference facade/value interface
examples/model_configs/
  *.ptafmodel               - versioned manifests with repo-relative artifact paths
ros2/
  ptafdeploy*                - optional lifecycle inference overlay
```

`CInferenceManager` is the public backend-agnostic facade; `CModelFacade` adds
role-level contracts for C++, Python, and MATLAB. Keep model-specific schemas
and result interpretation in integrations above raw tensor inference. Reuse the
shared geometry records and tensor adapters without adding model-specific
branches to the ORT backend.

## Current Gaps

- NMS and alternate detector policies are pending.
- The development tracker records historical local validation of ORT CUDA
  selection, fallback when TensorRT EP is unavailable, and standalone TensorRT
  inference. These results apply to the revisions and conditions recorded there.
- Jetson hardware validation is still pending.

## Benchmark

Small CPU smoke benchmark:

```bash
./build/src/programs/benchmark_model \
  tests/inference/model_configs/centroiding_fixture.ptafmodel \
  --iterations 20 --warmup 3
```

Jetson/CUDA target smoke benchmark, when the installed ONNX Runtime package
has CUDA execution-target support:

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

TensorRT standalone engine benchmark:

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

ORT TensorRT Execution Provider smoke, when the installed ORT build exposes
`TensorrtExecutionProvider`:

```bash
scripts/run_jetson_runtime_smoke.py --build-dir build \
  --ort-model model.onnx \
  --with-tensorrt-ep \
  --device 0 --iterations 50 --warmup 5 \
  -- --role raw_tensor --input-shape 1,3,640,640
```

Jetson notes:

- Use the TensorRT and CUDA versions installed by JetPack.
- `scripts/run_jetson_runtime_smoke.py --dry-run ...` is covered by CTest so
  argument handling can be checked without Jetson hardware.
- Set `TENSORRT_ROOT` only if TensorRT is outside the standard JetPack paths.
- Use `--trt-profile` for engines with multiple optimization profiles.
- DLA selection is not currently exposed or qualified by this library.
- Build engines on the target Jetson or a compatible target GPU/SM; TensorRT
  engines are not portable across arbitrary GPU architectures.
- Use `--device 0` for typical Jetson boards unless multiple accelerators are exposed.

Optional [image and inference-output utilities](doc/image_and_inference_output.md)
provide native annotation and JSON serialization without changing the inference facades.
