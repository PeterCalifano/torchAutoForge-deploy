# cpp_cuda_template v1.11.3 to v1.12.2 Semantic Upgrade Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `torchAutoForge-deploy` to the applicable build, packaging,
CUDA/PTX, documentation, development-environment, CI, wrapper, and ROS 2
capabilities of signed `cpp_cuda_template_project` tag `v1.12.2` plus its
explicitly archived approved dirty patch, while preserving the target's
inference architecture and excluding OptiX.

**Architecture:** The original three-way semantic port from the target's
previous template baseline (`c84b950`) through donor `v1.11.3`
(`dbffe4d4babb88a8682bfd19cf2bea3bc93fe252`) is followed by the smaller
semantic delta to signed `v1.12.2`
(`490de59d9fc5ec09ab4d57a9dbf5d6e238c7209a`) and its recorded post-tag
pre-review snapshot. Later donor-owned corrections are recorded separately
under Task 19 and do not rewrite that archive. The current target tree remains authoritative for ORT, TensorRT,
wrappers, model facades, packaging identity, ROS, and tests. Donor changes are
adopted or adapted by intent rather than copied over target-owned behavior.

**Tech Stack:** CMake 3.21+, C++20, optional CUDA/PTX, ONNX Runtime, optional
TensorRT, gtwrap, Catch2 v3, Doxygen, CPack, ROS 2 Jazzy, ament/colcon, GitHub
Actions, Docker/Podman.

## Global Constraints

- Work in the current checkout so the pre-existing inference/wrapper changes
  remain part of the semantic merge.
- Preserve all pre-existing unstaged/untracked work and the dirty `lib/wrap`
  submodule; do not reset or commit either repository. Donor edits are limited
  to the explicitly approved generic Task 19 fixes, carry
  `TODO: review (imported from torchAutoForge-deploy)`, and are staged/audited
  independently from unrelated donor work.
- Keep `autoforge_deploy` as the CMake/package identity and `ptafdeploy` as the
  C++ namespace.
- Keep ONNX Runtime required and standalone TensorRT optional/off by default.
- Remove all OptiX build/runtime support. A legally required attribution that
  mentions the origin of the PTX helper is not product support and must remain.
- Keep `src/template_src_kernels/placeholder.cu`,
  `src/template_src_kernels/placeholder.cuh`, and
  `src/template_src_kernels/placeholder_to_ptx.ptx.cu` as CUDA/PTX examples.
- PTX scope is compile plus embed only; do not introduce a CUDA Driver runtime
  loader.
- Do not add permanent tests for imported generic template machinery. New
  permanent tests may cover only `torchAutoForge-deploy` inference, wrapper, or
  ROS-facing behavior.
- Do not import `VerifyTemplateProject*` or register recursive
  configure/build/install/package/consumer checks in ordinary CTest. Keep
  matrix and delivery validation in explicit fresh acceptance builds or CI.
- Add a permanent CMake-script test only when it is lightweight, target-owned,
  non-recursive, and cannot be expressed through Catch2, pytest, an existing
  target, or the acceptance matrix. ROS ament/launch tests are the deliberate
  project-specific exception.
- Update a checkbox only after its stated verification has completed.

---

## Task 1: Baseline, provenance, and tailoring ledger

**Files:**

- Modify: `doc/development/cpp_cuda_template_v1_11_3_upgrade_plan.md`
- Modify: `doc/development/autoforge_deploy_upgrade_plan.md`
- Modify: `AGENTS.md`

**Interfaces:**

- Consumes: current target checkout and donor range
  `c84b950..dbffe4d4babb88a8682bfd19cf2bea3bc93fe252`.
- Produces: recorded baseline and an Adopt/Adapt/Skip ledger for later tasks.

- [x] Capture branch, HEAD, `git describe`, complete status, submodule status,
  configured build options, and current test inventory in this document.
- [x] Verify donor tag `v1.11.3` resolves to
  `dbffe4d4babb88a8682bfd19cf2bea3bc93fe252` and its signature is valid.
- [x] Run the existing configured CPU build's CTest suite and record its exact
  pass/fail result as the pre-upgrade behavioral baseline.
- [x] Classify donor changes:
  - Adopt: project metadata/versioning, compiler flags, PTX helper, Doxygen,
    CPack, test discovery infrastructure, devcontainer tooling, CI templates,
    logger, ROS metadata synchronization.
  - Adapt: root/source CMake, wrapper handler, CUDA handler, workflows,
    documentation, package exports, ROS overlay.
  - Skip: OptiX, ZeroMQ, donor template conformance/static tests, donor rollout
    plans/reports, `add_ros2_support.sh`, donor MATLAB examples, generated
    wrappers, active template-validation workflows.
- [x] Update `AGENTS.md` and the earlier upgrade plan with the new donor
  provenance and the target-specific no-generic-tests rule.

**Gate:** The recorded status must still match the checkout except for this
plan/documentation work.

---

## Task 2: Root CMake metadata, versioning, and dependency resolution

**Files:**

- Modify: `CMakeLists.txt`
- Modify: `generate_version.sh`
- Add/modify: `cmake/HandleCompilerFlags.cmake`
- Modify: `cmake/HandleGitVersion.cmake`
- Modify: `cmake/HandleToolchain.cmake`

**Interfaces:**

- Produces these cache controls:

```cmake
option(PROJECT_METADATA_ONLY "Configure project metadata without build targets" OFF)
option(WRITE_SOURCE_VERSION_FILE "Also update the source-tree VERSION file" OFF)
set(LIB_TARGET_NAME_OVERRIDE "" CACHE STRING "Override the built library target name")
option(autoforge_deploy_BUILD_PROGRAMS "Build command-line programs" ON)
option(autoforge_deploy_BUILD_EXAMPLES "Build examples" ON)
```

- Preserves existing `ENABLE_CUDA`, `ENABLE_TENSORRT`, `TENSORRT_ROOT`,
  wrapper, sanitizer, warning, and test options.

- [x] Port project description, homepage, maintainer, license, and build-tree
  version handling from v1.11.3 without adopting the donor's product version.
- [x] Make source `VERSION` writes opt-in and install the generated build-tree
  `VERSION`.
- [x] Ensure `PROJECT_METADATA_ONLY=ON` resolves metadata without discovering
  ORT/TensorRT, enabling languages, or creating build targets.
- [x] Activate `handle_toolchain()` before compiler-dependent configuration and
  replace duplicated flag logic with the donor compiler-flags interface target.
- [x] Resolve ONNX Runtime in this order: explicit `onnxruntime_DIR`, CMake
  `ONNXRUNTIME_ROOT`, environment `ONNXRUNTIME_ROOT`, normal config-package
  discovery. Remove the `$HOME/devDir/...` default.
- [x] Keep target-owned `HandleTensorRT.cmake` and ensure TensorRT is not
  discovered when `ENABLE_TENSORRT=OFF`.
- [x] Print a final configuration summary containing actual project options
  only; do not mention OptiX.

**Gate:**

```bash
cmake -S . -B /tmp/ptafdeploy-metadata -DPROJECT_METADATA_ONLY=ON
git status --short
```

Metadata configuration must succeed without ORT and must not modify `VERSION`.

---

## Task 3: Source discovery, canonical target, and package exports

**Files:**

- Modify: `cmake/cmake_utils.cmake`
- Modify: `src/CMakeLists.txt`
- Modify: CMake files below `src/`
- Modify: package-config templates under `cmake/`

**Interfaces:**

- Produces canonical target `autoforge_deploy::autoforge_deploy` in both build
  and install trees.
- Ordinary CUDA source inventory excludes `*.ptx.cu`.

- [x] Replace malformed semicolon-containing glob patterns with one pattern per
  list element and filter dedicated PTX sources from ordinary CUDA sources.
- [x] Invoke the source inventory helper and retain target-specific source
  subdirectories, including ORT and optional TensorRT.
- [x] Add build alias/export-name handling so the canonical namespaced target is
  stable even when `LIB_TARGET_NAME_OVERRIDE` is used.
- [x] Use `GNUInstallDirs` consistently for headers, libraries, programs,
  exports, and package configuration.
- [x] Prevent build-only include directories and dependency implementation
  details from leaking into the installed target.
- [x] Keep public inference headers and wrapper-safe types installed under the
  existing include layout.

**Gate:** Configure/build/install to a temporary prefix, then build a temporary
consumer that includes `inference/model_facade.h` and links only
`autoforge_deploy::autoforge_deploy`. This is an integration check, not a new
repository test.

---

## Task 4: CUDA and PTX compile/embed support

**Files:**

- Add/modify: `cmake/cmake_cuda_ptx_tools.cmake`
- Modify: `cmake/HandleCUDA.cmake`
- Modify: `CMakeLists.txt`
- Modify: `src/CMakeLists.txt`
- Modify: `src/template_src_kernels/CMakeLists.txt`
- Modify: `src/template_src_kernels/placeholder_to_ptx.ptx.cu`

**Interfaces:**

```cmake
option(CUDA_PTX_USE_FAST_MATH
       "Enable NVCC fast math when compiling PTX kernels" ON)
set(CUDA_PTX_NVCC_FLAGS "" CACHE STRING
    "Additional NVCC flags used only for PTX compilation")
```

- `cuda_compile_and_embed(output_var cuda_arch include_dirs sources...)`
  produces object paths and `<output_var>_SYMBOLS`.

- [x] Port the v1.11.3 PTX macro with its single-numeric-architecture,
  `bin2c`, include-directory, and flag validation.
- [x] Enable C compilation for generated embedded sources without enabling
  CUDA when `ENABLE_CUDA=OFF`.
- [x] Activate PTX generation when CUDA is enabled and dedicated `.ptx.cu`
  sources are present; do not guard it with an OptiX option.
- [x] Keep `placeholder.cu/.cuh` as the ordinary CUDA example.
- [x] Keep `placeholder_to_ptx.ptx.cu` and rename its entry point to
  `ptafdeploy_ptx_example_kernel`; remove OptiX-specific comments/semantics.
- [x] Ensure CUDA build-only include directories do not leak into exports.
- [x] Do not add a PTX/logger/CMake unit test.

**Gate:**

```bash
cmake -S . -B /tmp/ptafdeploy-cuda \
  -GNinja -DENABLE_CUDA=ON -DENABLE_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=<one numeric architecture>
cmake --build /tmp/ptafdeploy-cuda --parallel
ctest --test-dir /tmp/ptafdeploy-cuda --output-on-failure
```

The build must compile the ordinary CUDA example and generate/embed PTX. CTest
must execute only target-owned tests.

---

## Task 5: Packaging, docs, wrappers, logger, and build helpers

**Files:**

- Modify: `cmake/HandleWrapper.cmake`
- Add/modify: `cmake/HandleDoxygenDocs.cmake`
- Modify: `doc/CMakeLists.txt`
- Modify: `doc/Doxyfile.in`
- Modify: `build_lib.sh`
- Add/modify: `CMakePresets.json`
- Add: `src/utils/logging/CLogger.h`
- Add: `src/utils/logging/CLogger.cpp`
- Modify: `src/utils/CMakeLists.txt`

**Interfaces:**

- Adds public `ptafdeploy::logging::ELogLevel`,
  `ptafdeploy::logging::ELogColorMode`, and
  `ptafdeploy::logging::CLogger`.
- `CLogger::setLevelFromEnvironment()` defaults to
  `PTAFDEPLOY_LOG_LEVEL`.

- [x] Merge donor wrapper docstrings, generated-path handling, Python
  executable/environment support, and CTest arguments with the target's newer
  RPATH and package-layout fixes.
- [x] Preserve the existing repo-specific Python facade smoke test instead of
  adding the donor generic import test.
- [x] Add dependency-free logger sources under the target namespace without
  refactoring existing program output in this upgrade.
- [x] Add namespaced Doxygen HTML/XML targets and connect wrapper docstring
  generation to XML when requested.
- [x] Tailor the donor docs preset with target option names and no OptiX.
- [x] Upgrade CPack metadata and source-ignore rules; ensure build trees,
  caches, generated wrappers, `.git`, and local artifacts are excluded.
- [x] Upgrade `build_lib.sh` with v1.11.3 Python/CTest/cross-build handling
  while preserving all current command-line options.

**Gate:** Build docs, Python wrapper, MATLAB wrapper generation, binary package,
source package, and an extracted source-package CPU build. Run existing
repo-specific wrapper tests only.

---

## Task 6: Developer environment and project-tailored workflows

**Files:**

- Modify: `.devcontainer/`
- Modify: `.github/workflows/c-cpp.yml`
- Add: `.github/workflows/build_linux_cuda.yml`
- Add: `.github/workflows/build_ros2_overlay.yml`
- Modify/add: `.github/ISSUE_TEMPLATE/`
- Modify/add: `.github/pull_request_template.md`
- Delete: stale `.github/workflows/*.templ0` and `*.templ1`

**Interfaces:**

- Native hosted CI remains CPU/ORT and wrapper focused.
- CUDA CI runs only when `vars.CI_USE_SELF_HOSTED == 'true'`.
- ROS CI uses Ubuntu 24.04/ROS 2 Jazzy.

- [x] Upgrade devcontainer JSONC-preserving updater, CPU/CUDA 12.9 profiles,
  Docker/Podman GPU arguments, extensions, and in-container runner.
- [x] Preserve the current x86_64/arm64, Python wrapper, docs, and Pages jobs;
  add full tag history and relevant tag triggers.
- [x] Add optional CUDA build workflow with no OptiX input. It builds CUDA/PTX
  integration and runs existing target tests.
- [x] Add project-tailored issue forms and PR checklist.
- [x] Remove dormant stale workflow copies and do not materialize donor
  template-conformance workflows.

**Gate:** Parse all JSON/YAML, shell-check applicable scripts, and configure the
container/workflow commands locally where dependencies permit. No generic
workflow test is added to CTest.

---

## Task 7: Target-specific ROS 2 interfaces

**Files:**

- Add: `ros2/ptafdeploy_interfaces/msg/FloatTensor.msg`
- Add: `ros2/ptafdeploy_interfaces/msg/ModelStatus.msg`
- Add: `ros2/ptafdeploy_interfaces/srv/InferFloatTensors.srv`
- Add: `ros2/ptafdeploy_interfaces/CMakeLists.txt`
- Add: `ros2/ptafdeploy_interfaces/package.xml`

**Interfaces:**

`FloatTensor.msg`:

```text
string name
int64[] shape
float32[] values
```

`ModelStatus.msg`:

```text
builtin_interfaces/Time stamp
string lifecycle_state
bool model_loaded
string model_role
string backend_detail
uint64 inference_count
float64 last_inference_ms
string last_error
```

`InferFloatTensors.srv`:

```text
FloatTensor[] inputs
---
bool success
FloatTensor[] outputs
string status
```

- [x] Add the three interfaces with `builtin_interfaces` and
  `rosidl_default_generators`.
- [x] Set package metadata from target project identity/version, not donor
  identity/version.
- [x] Add no scalar placeholder algorithm interfaces.

**Gate:** `colcon build --packages-select ptafdeploy_interfaces` succeeds in a
Jazzy environment.

---

## Task 8: Target-specific ROS 2 facade bridge

**Files:**

- Add: `ros2/ptafdeploy/CMakeLists.txt`
- Add: `ros2/ptafdeploy/package.xml`
- Add: `ros2/ptafdeploy_ros/include/ptafdeploy_ros/CInferenceLifecycleNode.h`
- Add: `ros2/ptafdeploy_ros/include/ptafdeploy_ros/conversions.h`
- Add: `ros2/ptafdeploy_ros/src/CInferenceLifecycleNode.cpp`
- Add: `ros2/ptafdeploy_ros/src/conversions.cpp`
- Add: `ros2/ptafdeploy_ros/src/ptafdeploy_node_main.cpp`
- Add: `ros2/ptafdeploy_ros/test/test_conversions.cpp`
- Add: `ros2/ptafdeploy_ros/test/test_inference_lifecycle_node.cpp`

**Interfaces:**

- Component: `ptafdeploy_ros::CInferenceLifecycleNode`.
- Node name: `ptafdeploy_inference`.
- Parameter: `model_config_path` (required at configure time).
- Service: `~/infer`.
- Lifecycle publisher: `~/status`.
- Core call: `CModelFacade::InferFloatTensors(const std::vector<SFloatTensor>&)`.

- [x] RED: add conversion tests proving valid tensor round-trip and rejection
  of negative/dynamic dimensions, cardinality mismatch, and overflow.
- [x] Run the conversion test and verify it fails because the bridge is absent.
- [x] GREEN: implement conversion helpers using `SFloatTensor`; do not expose
  raw ORT handles, `void*`, or byte storage.
- [x] Run the conversion test and verify it passes.
- [x] RED: add lifecycle tests for configure/load, inactive rejection, active
  fixture inference, exception translation, status accounting, and cleanup.
- [x] Run lifecycle tests and verify the expected missing-node failures.
- [x] GREEN: implement the lifecycle component with a guarded
  `std::unique_ptr<CModelFacade>`, serialized inference, and failure responses
  that do not terminate the node.
- [x] Increment `inference_count` only after successful calls; publish status on
  activation and after every request.
- [x] Run both ROS unit-test targets and verify they pass.

---

## Task 9: ROS runtime assets, launch, metadata, and integration

**Files:**

- Add: `ros2/ptafdeploy_spinup/`
- Add: `build_ros2.sh`
- Add: `ros2/tools/sync_package_metadata.py`
- Modify: `generate_version.sh`
- Modify: `.github/workflows/build_ros2_overlay.yml`

**Interfaces:**

- Launch arguments: `autostart`, `namespace`, `node_name`,
  `model_config_path`.
- Launches: standalone executable and composable component.
- Installed fixture input: values `1.0F` through `11.0F`, shape `[1, 11]`.
- Expected output: approximately `[1.686690331F, 4.697796822F]`.

- [x] Install `tracedSampleModel.onnx` and a ROS-relative
  `centroiding_traced_sample.ptafmodel` under the spinup package share tree.
- [x] Add standalone and composed launch files with automatic lifecycle
  configure/activate transitions when `autostart=true`.
- [x] RED: add launch tests for standalone/composed and root/namespaced modes,
  requiring active state and real fixture inference.
- [x] Run the launch test and verify it fails before runtime assets/launches are
  complete.
- [x] GREEN: complete launches, installed paths, and lifecycle orchestration.
- [x] Run all four launch modes and verify the expected numeric output and
  status metadata.
- [x] Add CUDA-only forwarding to `build_ros2.sh`; no OptiX compatibility flag.
- [x] Add `generate_version.sh --sync-ros2` and verify all four package
  manifests match the target project metadata.
- [x] Make ROS CI install target ONNX Runtime before rosdep/build/test.

**Gate:**

```bash
./build_ros2.sh --clean
colcon test --base-paths ros2 --event-handlers console_direct+
colcon test-result --test-result-base build --verbose
```

All target-owned ROS tests must report zero failures/errors.

---

## Task 10: Remove target-irrelevant template scaffolding

**Files:**

- Delete: `src/template_src/`
- Delete: `src/template_folder/`
- Delete: `examples/template_examples/`
- Delete: `tests/template_test/`
- Delete: `tests/template_fixtures/`
- Preserve: `src/template_src_kernels/`

- [x] Remove unrelated C++ placeholders, fixtures, examples, and the
  always-pass template test.
- [x] Remove CMake references to deleted scaffolding while retaining the
  CUDA/PTX example directory.
- [x] Scan for `template_project`, `cpp_playground`, stale workflow markers,
  spdlog, ZeroMQ, and OptiX. Resolve every hit except explicit provenance,
  historical upgrade documentation, mandatory PTX attribution, and the three
  user-retained placeholder paths.
- [x] Confirm no donor generic/static/conformance test was imported.

---

## Task 11: Final documentation and verification

**Files:**

- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `TODO`
- Modify: `doc/development/cpp_cuda_template_v1_11_3_upgrade_plan.md`
- Modify: relevant build/testing/ROS documentation under `doc/`

- [x] Document new build options, ORT discovery precedence, CUDA/PTX examples,
  packaging, logger, wrapper docs, devcontainers, CI, and ROS usage.
- [x] State precisely that PTX is compile/embed support and OptiX is absent.
- [x] Document the distinction between permanent repo-specific tests and
  one-time integration acceptance checks.
- [x] Run a fresh default CPU configure/build/CTest.
- [x] Run Debug and RelWithDebInfo builds with warnings enabled.
- [x] Run installed-consumer and binary/source-package checks.
- [x] Run Python wrapper generation plus the existing facade smoke test.
- [x] Run MATLAB wrapper generation; run MATLAB smoke only when a licensed
  runtime is available and otherwise record the limitation.
- [x] Run TensorRT-enabled configure/build/tests when the local SDK is
  available, retaining the existing environment-gated engine inference.
- [x] Run the explicit-architecture CUDA build and target-owned CTests.
- [x] Run the complete ROS Jazzy build/test matrix.
- [x] Verify configuration/build/package/docs commands do not introduce
  unexpected source-tree changes relative to the captured baseline.
- [x] Run `git diff --check` and review every changed path against this plan.
- [x] Record exact commands, pass counts, unavailable optional environments,
  and remaining external CI/Jetson observations below.

---

## Task 12: Moved-tag CMake-test policy refresh

**Files:**

- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`
- Modify: `tests/inference/CMakeLists.txt`
- Modify: `doc/development/cpp_cuda_template_v1_11_3_upgrade_plan.md`

- [x] Re-resolve and re-verify the moved signed `v1.11.3` donor tag.
- [x] Review the complete delta from the previously audited tag commit
  `f3fd6555c7c965eebc89d5aa40227b0f36ac5fde`.
- [x] Mirror the clarified derived-project CMake-test policy in local agent
  guidance.
- [x] Remove avoidable generated CMake-script wrappers from ordinary CTest
  while retaining direct tests of `benchmark_model` and
  `get_available_providers`.
- [x] Reconfigure the existing build tree and prove that its inventory contains
  neither stale template tests nor CMake-script test commands.
- [x] Run the focused program tests and the complete native CTest suite.
- [x] Run `git diff --check` and record the refresh evidence below.

**Gate:** Ordinary native CTest may contain Catch2, wrapper, direct
project-executable, and target-owned smoke tests, but no donor conformance test,
recursive project build, or avoidable CMake-script test. Build/install/package
and consumer matrices remain explicit acceptance or CI work.

---

## Task 13: Establish the v1.12.2 follow-up contract

**Files:**

- Modify: `doc/development/cpp_cuda_template_v1_11_3_upgrade_plan.md`
- Modify: `AGENTS.md`
- Modify: `CLAUDE.md`
- Update gitlink: `lib/wrap`

**Interfaces:**

- Committed donor: signed `cpp_cuda_template_project` tag `v1.12.2`, annotated
  tag object `1f60cebe98557bbba73a978f5a1243eb3c21f533`, commit
  `490de59d9fc5ec09ab4d57a9dbf5d6e238c7209a`.
- Approved post-tag snapshot:
  `doc/development/cpp_cuda_template_v1_12_2_dirty.patch`, SHA-256
  `1e251f34b353b393f9a3463565fb22ba553084cb49403513c133147b7a8db041`,
  stable patch ID `4645c110f7a48da523575f95faac5cb1500a87ce`. It records
  all 15 tracked donor deltas plus the untracked
  `cmake/StagePackageVersion.cmake.in`; only relevant production/documentation
  semantics are ported.
- Approved gtwrap revision: `bf9f78617830bfa88c70d81a1a791c0a0c90b548`
  from `lib/wrap` `origin/master`; stage only the superproject gitlink and do
  not modify or commit external gtwrap source.

- [x] Verify the signed tag, tag commit, donor status, and dirty patch identity.
- [x] Fast-forward the clean `lib/wrap` checkout to the approved remote commit.
- [x] Record the v1.12.2 Adopt/Adapt/Skip ledger in the execution record.
- [x] Update local guidance for the safe-clean, wrapper-runtime, Python
  `FULL_VERSION`, CI branch-policy, and target-owned-test contracts.

**Gate:** No donor identity, OptiX support, `VerifyTemplateProject*` test,
template workflow contract, or donor development report enters the target.

## Task 14: Port safe cleanup and package ownership

**Files:**

- Modify: `build_lib.sh`
- Modify: `CMakeLists.txt`
- Modify: `python/pyproject.toml.in`
- Modify: `python/setup.py.in`
- Modify: `python/setup.py`
- Modify: `doc/build_packaging_and_template_sync.md`

- [x] Capture failing acceptance evidence showing that the current clean helper
  lacks the v1.12.2 exact-checkout cache-ownership guard and that Python package
  metadata still uses `PROJECT_VERSION`.
- [x] Port the donor clean-path guard so `--clean` removes only a conventional
  in-repository build directory whose `CMakeCache.txt` resolves
  `CMAKE_HOME_DIRECTORY` to this checkout.
- [x] Port active/cache-proven source-tree exclusion and generated build-tree
  `VERSION` staging, including the applicable post-tag dirty patch.
- [x] Configure Python project metadata from `FULL_VERSION`; retain target name,
  Python 3.12 floor, ORT runtime contract, and prefix-relative installation.
- [x] Validate with disposable clean-safety, binary/source CPack, extracted
  source build, and external installed-consumer commands. Do not register any
  of these acceptance checks with CTest.

## Task 15: Port relocatable Python and MATLAB wrapper infrastructure

**Files:**

- Modify: `CMakeLists.txt`
- Modify: `cmake/HandleWrapper.cmake`
- Add: `cmake/HandlePythonWrapper.cmake`
- Add: `cmake/HandleMatlabWrapper.cmake`
- Add: `cmake/StagePythonRuntimeArtifacts.cmake`
- Modify: `tests/inference/CMakeLists.txt` only if the existing project-owned
  wrapper smoke needs lifecycle cleanup compatible with current gtwrap.
- Modify: wrapper documentation in `README.md` and
  `doc/build_packaging_and_template_sync.md`

- [x] Capture the current absence of
  `autoforge_deploy_GTWRAP_RUNTIME_DEPENDENCY_TARGETS` and split wrapper modules.
- [x] Adapt the v1.12.2 wrapper modules while preserving
  `src/wrap_interface.i`, `src/inference/inference.i`, namespace `ptafdeploy`,
  and target wrapper option names.
- [x] Declare `autoforge_deploy` as the direct project-owned shared runtime
  packaged beside the Python extension; reject unsupported target types and
  flat-package filename collisions.
- [x] Keep `_wrapper_build.py` checkout-only, use loader-relative runtime paths,
  and keep CMake install destinations prefix-relative.
- [x] Generate and run the existing target-owned Python facade smoke, build a
  wheel, inspect its `FULL_VERSION` and native runtime set, install it into an
  isolated target, and run real facade inference.
- [x] Generate the MATLAB wrapper at the approved gtwrap revision and run the
  existing target-owned model-load/inference smoke through construction,
  dispatch, error recovery, `clear classes`, and `clear mex` when MATLAB is
  available.

**Gate:** No gtwrap/template unit test is copied or registered. Validation
exercises only this repository's wrapped inference API and delivery artifacts.

## Task 16: Reconcile ROS 2 and container entry points

**Files:**

- Modify as required: `build_ros2.sh`
- Modify as required: `ros2/ptafdeploy/**`
- Modify as required: `ros2/ptafdeploy_interfaces/**`
- Modify as required: `ros2/ptafdeploy_ros/**`
- Modify as required: `ros2/ptafdeploy_spinup/**`
- Modify as required: `ros2/tools/sync_package_metadata.py`
- Modify: `run_in_container.sh`
- Modify: `doc/ros2_overlay.md`

- [x] Confirm the existing four-package overlay remains additive and that
  `ptafdeploy_ros` calls `CModelFacade`, never an ORT-specific backend.
- [x] Reconcile v1.12.2 metadata synchronization and build-helper behavior
  without replacing target messages, services, launch policy, or package names.
- [x] Tailor `run_in_container.sh` for the target's native, CUDA/TensorRT,
  wrapper, and ROS entry points without adding a generic container test.
- [x] Run shell syntax, metadata check, clean Jazzy build/test, and installed
  standalone/composed launch/service inference acceptance.

## Task 17: Initialize project-owned CI workflows

**Files:**

- Delete: `.github/workflows/c-cpp.yml`
- Add: `.github/workflows/build_linux.yml`
- Add: `.github/workflows/build_docs.yml`
- Modify: `.github/workflows/build_linux_cuda.yml`
- Modify: `.github/workflows/build_ros2_overlay.yml`
- Modify: `.github/scripts/install_onnxruntime.sh`
- Modify: `README.md`

- [x] Create separate native C++, documentation/Pages, CUDA/TensorRT, and ROS 2
  workflows using target commands and no donor template validation.
- [x] Run native/docs/ROS automatically for `main` pushes and pull requests
  targeting `main` or `develop`; permit `develop` validation through
  `workflow_dispatch` without an automatic `develop` push trigger.
- [x] Keep CUDA/TensorRT manual and disabled unless repository variable
  `CI_USE_SELF_HOSTED=true`; require a runner labelled
  `[self-hosted, Linux, X64, gpu, cuda]`, CUDA, and TensorRT.
- [x] Configure the dormant GPU job with `ENABLE_CUDA=ON`,
  `ENABLE_TENSORRT=ON`, explicit numeric CUDA architecture, and target-owned
  CTest only. Do not grep/assert generic placeholder or PTX helper behavior.
- [x] Keep Python facade validation in native CI; keep MATLAB local because its
  runtime/license contract is not appropriate for hosted runners.
- [x] Parse all YAML with PyYAML and inspect effective trigger/job/permission
  structure through one-time acceptance commands, not a committed Python test.

## Task 18: Full validation, review, and exact staging

- [x] Run fresh native CPU configure/build and all target-owned CTests.
- [x] Run Debug with warnings-as-errors, CUDA+TensorRT where locally available,
  documentation, install/consumer, source/binary CPack, Python, MATLAB, ROS 2,
  shell, structured-format, and metadata gates.
- [x] Prove CTest contains no template, recursive CMake, or donor conformance
  entry and that OptiX remains absent except mandatory PTX-helper attribution.
- [x] Run conflict-marker, stale donor identity, machine-local path,
  generated-artifact, executable-mode, and `git diff --check` scans.
- [x] Review every intended source file for file/API documentation and
  purpose-level readability under the staged-code quality gate.
- [x] After review-scope correction, clear the index and stage only the explicit
  build/configuration, CMake, wrapper-packaging, approved `lib/wrap`, Python
  runtime-loader, documentation-workflow, and directly imported CLogger/CUDA/PTX
  functionality allowlist. Leave the inference implementation, wrapper
  interfaces, native/CUDA/ROS workflows, ROS overlay, model adapters, auxiliary
  implementation changes, programs, tests, model fixtures, prose documentation,
  and unrelated deletions unstaged for coherent later batches.
- [x] Inspect `git diff --cached`, `git diff --cached --check`, and the staged
  submodule log; leave commit and push to the user.
- [x] Propose a scoped build/configuration commit title and description.

**Gate:** The index matches the explicit reviewed staging allowlist. A clean
index-only export must configure, build, and install the native project with
tests disabled, and must build/import the Python wrapper using the pre-feature
interfaces present at the staged parent revision. Project tests and ROS are not
represented as staged-tree evidence for this build-only consolidation.

## Task 19: Post-review ownership and portability corrections

**Files:**

- Modify: `cmake/HandleWrapper.cmake`
- Modify: `cmake/HandlePythonWrapper.cmake`
- Modify: `cmake/HandleTensorRT.cmake`
- Add: `cmake/FindTensorRT.cmake`
- Modify: `src/cmake/autoforge_deployConfig.cmake.in`
- Modify: `src/CMakeLists.txt`
- Modify: Python CI and wrapper/package documentation as required
- Modify: affected target public headers and machine-local example/test paths
- Modify generically applicable counterparts in
  `/home/peterc/devDir/dev-tools/cpp_cuda_template_project`, each marked
  `TODO: review (imported from torchAutoForge-deploy)`

- [x] Confirm Finding 1 is present in signed donor v1.12.2 and remains present
  in the live donor dirty patch: the target `HandleWrapper.cmake` is
  byte-identical to the donor, so this is not a failed local alignment.
- [x] Confirm the donor MATLAB regressions exercise generated wrapper behavior
  but do not inspect ignored files created inside the gtwrap checkout.
- [x] Remove the redundant configure-time MATLAB template write from target and
  donor, relying on the approved gtwrap fallback without modifying its source
  checkout. Verify a pristine temporary gtwrap tree remains byte-for-byte
  unchanged after ordinary MATLAB configure/generation.
- [x] Replace absolute TensorRT include/library export entries with imported
  targets discovered through an installed `FindTensorRT.cmake`. Validate a
  TensorRT-enabled install from a relocated prefix with an external consumer;
  keep this acceptance outside ordinary target CTest. Keep installed target
  RUNPATH loader-relative and prevent producer ONNX Runtime/CUDA/TensorRT paths
  from leaking through the exported package.
- [x] Promote the generic TensorRT finder/package-discovery support to the donor
  as optional infrastructure for ML-derived libraries. Do not enable or link
  TensorRT in the donor itself.
- [x] Move configured Python metadata, wrapper-link metadata, imports, wheel
  assembly, pip installation, and stubs to a build-owned staging tree. Keep
  source templates stable and prove wrapper configuration no longer rewrites
  tracked source files in target or donor.
- [x] Complete Doxygen contracts on the new target public inference APIs,
  remove the developer-local centroiding fixture fallback, and make the legacy
  YOLO example use normal package discovery instead of a personal ORT prefix.
- [x] Run the affected target native/CUDA/TensorRT/Python/MATLAB/package/docs
  gates and the donor's existing default, wrapper, packaging, CUDA/PTX, and
  ROS-compatible gates before staging either repository.
- [ ] Observe the donor OptiX profile on an SDK-equipped host. No OptiX SDK is
  available locally; Task 19 does not alter the donor OptiX handler, targets,
  sources, or tests, and the target intentionally remains OptiX-free.

### Explicitly deferred runtime findings

- [x] Record Finding 3 without changing runtime behavior in this
  consolidation. A later project-owned change must make model reload
  transactional: construct session/backend/metadata/contract candidate state,
  replace active state only on success, destroy each ORT session before its
  environment, and preserve the previously loaded model after a failed reload.
- [x] Record Finding 5 without adding a TensorRT fixture or DLA implementation
  now. Current consolidation evidence covers TensorRT compilation/linking,
  option validation, missing/invalid-engine diagnostics, and the optional
  no-fixture test path. Real serialized-engine execution and DLA behavior must
  be revalidated before they are treated as current supported evidence.

**Gate:** Neither checkout is modified by ordinary wrapper configuration;
installed TensorRT exports contain no producer SDK paths; Python wrapper builds
leave tracked source metadata unchanged; target and donor indexes are reviewed
independently. No donor conformance test is copied into this derived project.

The archived `cpp_cuda_template_v1_12_2_dirty.patch` remains the exact approved
pre-review donor snapshot. The live donor is expected to differ only by the
Task 19 imported corrections until those changes receive their own donor
review; do not recompute the archive hash as if it represented that later state.

## Execution Record

### Baseline

- Captured: 2026-07-23, Europe/Rome.
- Branch: `feature/ort_manager_pipeline_wrapped`.
- HEAD: `16f1e26fc8864de8624032fb55d799f5e9bb764d`.
- Describe: `v0.1.0-6-g16f1e26-dirty`.
- Index: no staged changes.
- Worktree: pre-existing modified and untracked inference, wrapper, build,
  documentation, CI, and test work; this plan is the only new path introduced
  before implementation.
- Submodule: `lib/wrap` is intentionally dirty at
  `1c27f1f8fff151bf58683a72fc9d15574c4a308c`
  (`cpp_cuda_templ_reference_tag-17-g1c27f1f`).
- Existing configured build: `RelWithDebInfo`, sanitizers off, warnings-as-errors
  off, ONNX Runtime from
  `/home/peterc/devDir/ML-repos/onnxruntime/install`.
- Existing CTest inventory: 22 tests. `test_template` is the sole
  target-irrelevant placeholder; the other 21 tests exercise inference,
  adapters, facade/config, benchmark/provider tooling, TensorRT error handling,
  and optional CPU/CUDA runtime behavior.
- Baseline command:
  `ctest --test-dir build --output-on-failure -j 6`.
- Baseline result: 22/22 passed, zero failures, 3.68 seconds.
- At initial capture, the clean donor `main` and signed annotated `v1.11.3`
  tag resolved to
  `f3fd6555c7c965eebc89d5aa40227b0f36ac5fde`; signature verification reported
  a good signature for `petercalifano.gs@gmail.com`, RSA key
  `SHA256:wekaTueFFQPCPB7/VR76jv4LljPlUuabwMEi0NBPSxc`.
- On the 2026-07-24 policy refresh, donor `main` remained clean and equal to
  `origin/main`, while the re-signed annotated `v1.11.3` tag object
  `e5421cad88f8fc2806a9a0c0f63c945f1e782726` resolved to commit
  `dbffe4d4babb88a8682bfd19cf2bea3bc93fe252`. Signature verification again
  reported the same good signer and RSA key. The delta from the initially
  audited commit changes only `AGENTS.md` and three tailoring/testing
  documentation files (83 insertions, 3 deletions).

### Tailoring ledger

| Donor area | Decision | Target treatment |
|---|---|---|
| Metadata/versioning | Adopt | Keep target product version and make source VERSION writes opt-in. |
| Compiler/toolchain flags | Adopt | Route through donor helpers; retain target options and C++20. |
| Root/source CMake | Adapt | Preserve ORT, TensorRT, wrappers, programs, and exported target. |
| CUDA/PTX | Adapt | Import compile/embed support; retain CUDA examples; remove OptiX coupling. |
| OptiX | Skip | No option, handler, SDK path, workflow input, ROS flag, or runtime claim. |
| Wrapper CMake | Adapt | Combine donor generation/docs/test handling with target RPATH/package fixes. |
| Doxygen/Pages | Adapt | Keep target API/docs and add XML-backed wrapper docstrings. |
| Catch2/Python discovery | Adopt infrastructure | Register only target-owned tests. |
| CPack/install consumers | Adapt | Preserve target identity/dependencies and add source-package hygiene. |
| Dependency-free logger | Adapt | Rename to `ptafdeploy::logging`; no generic logger test. |
| Devcontainers | Adapt | Target CPU/CUDA 12.9 profiles; no OptiX. |
| Native/CUDA CI | Adapt | Preserve target workflow; optional guarded CUDA build; no template validation. |
| ROS 2 overlay | Adapt | Four `ptafdeploy*` packages calling `CModelFacade` with real float tensors. |
| ROS metadata sync | Adopt | Synchronize target project metadata across four manifests. |
| ZeroMQ/spdlog | Skip | Neither is a target dependency; donor's logger replaces spdlog. |
| Template conformance/static tests | Skip | Template implementation is trusted; no generic tests in this target. |
| Tailoring/rollout scripts and reports | Skip | One-time donor tooling is not target runtime/build content. |
| Donor generated wrappers/MATLAB examples | Skip | Preserve target-owned wrapper sources and tests. |

### Verification results

- Baseline CPU CTest: 22/22 passed before implementation.
- CUDA/PTX scratch build: CUDA 12.9, architecture 89, ordinary CUDA compilation,
  PTX generation, `bin2c` embedding, and 22/22 then-current tests passed.
- Installed CPU consumer: configured, built, linked only
  `autoforge_deploy::autoforge_deploy`, and ran successfully.
- ROS interfaces scratch build: `ptafdeploy_interfaces` configured, generated,
  built, and installed successfully under ROS 2 Jazzy.
- ROS conversion RED: after correcting test-only CMake signature usage, build
  failed at the intentionally absent `ptafdeploy_ros/conversions.h`.
- ROS lifecycle RED: build failed at the intentionally absent
  `ptafdeploy_ros/CInferenceLifecycleNode.h`.
- ROS bridge GREEN: nested core install, interfaces, conversions, component,
  standalone node, and tests built; `test_conversions` and
  `test_inference_lifecycle_node` passed 2/2 in 0.30 seconds.
- ROS launch RED: installed spinup test failed because
  `ptafdeploy.launch.py` was intentionally absent.
- ROS launch GREEN: standalone/composed and root/namespaced cases loaded the
  installed ONNX fixture, reached active state, returned the expected centroid,
  and validated status metadata; launch test passed in 6.78 seconds.

### Final acceptance after the last implementation changes

- Default/final CPU:
  `cmake -S . -B /tmp/ptafdeploy-rel.FfeOVM`,
  `cmake --build /tmp/ptafdeploy-rel.FfeOVM --parallel 6`, and
  `ctest --test-dir /tmp/ptafdeploy-rel.FfeOVM --output-on-failure -j 6`
  passed 21/21 target-owned tests in 3.71 seconds.
- Debug with target warnings as errors:
  `/tmp/ptafdeploy-debug.RVu0X6` retained
  `CMAKE_BUILD_TYPE=Debug` and `WARNINGS_ARE_ERRORS=ON`; build and CTest passed
  21/21 in 3.68 seconds.
- CUDA/PTX:
  `/tmp/ptafdeploy-cuda-final.Q0C3OF` retained `ENABLE_CUDA=ON` and
  `CMAKE_CUDA_ARCHITECTURES=89`; build and CTest passed 21/21 in 3.18 seconds.
  `ninja -t commands` confirmed ordinary compilation of `placeholder.cu`,
  PTX-only compilation of `placeholder_to_ptx.ptx.cu`, `bin2c` embedding, and
  C compilation of `placeholder_to_ptx_embedded.c`. The PTX exports
  `ptafdeploy_ptx_example_kernel`.
- TensorRT:
  `/tmp/ptafdeploy-trt-final.FwntTS` used `ENABLE_TENSORRT=ON` and
  `TENSORRT_ROOT=/usr/local/tensorrt10.7-cuda12.6`; build and CTest passed
  22/22 in 3.95 seconds. `PTAFDEPLOY_TENSORRT_TEST_ENGINE` was not set, so the
  optional engine-inference case validated its documented no-fixture path
  rather than running a serialized engine.
- Python wrapper:
  `/tmp/ptafdeploy-python-fixed.8SPy27` generated Doxygen XML-backed gtwrap
  bindings and
  `ctest -R autoforge_deploy_python_facade_smoke --output-on-failure` passed
  1/1 in 1.54 seconds.
- MATLAB wrapper:
  `/tmp/ptafdeploy-matlab-final.ogGWWt` generated the R2023b MEX wrapper;
  MATLAB successfully acquired a local license and
  `ctest -R autoforge_deploy_matlab_model_load --output-on-failure` passed
  1/1 in 10.28 seconds. Wrapper compilation warnings were in bundled gtwrap,
  pybind11, Eigen, or generated sources, not the target's warning-clean core.
- Documentation:
  `cmake --build /tmp/ptafdeploy-rel.FfeOVM --target autoforge_deploy_doc`
  completed with Doxygen 1.9.8 and no warning/error. The Python wrapper build
  also generated the XML used for docstrings.
- Install/export:
  `cmake --install /tmp/ptafdeploy-rel.FfeOVM --prefix
  /tmp/ptafdeploy-install-release.ry7HtQ` followed by a fresh external consumer
  configure/build/run in `/tmp/ptafdeploy-consumer-release.IdjpWu` succeeded.
  The consumer linked only `autoforge_deploy::autoforge_deploy`.
- Packaging:
  CPack generated separate binary and source TGZ archives under
  `/tmp/ptafdeploy-packages-release.1bzSZw` (2,549,723 and 172,212,833 bytes).
  Neither archive contains Git metadata, build/install/log trees, Python
  caches, or removed template scaffolding. A fresh build extracted from the
  source TGZ passed 21/21 tests in 3.59 seconds.
- ROS 2 Jazzy:
  `./build_ros2.sh --clean --no-version-sync --cmake-arg
  -Donnxruntime_DIR=/home/peterc/devDir/ML-repos/onnxruntime/install/lib/cmake/onnxruntime`
  built all four packages in 34.1 seconds. Colcon reported 13 tests, zero
  errors, zero failures, and zero skipped. The spinup test passed all
  standalone/composed and root/namespaced real-inference modes in 6.79
  seconds.
- Metadata/tooling:
  `generate_version.sh --sync-ros2` was idempotent at
  `0.1.0+6.g16f1e26.dirty`; all four manifests pass
  `sync_package_metadata.py --check`. Shell syntax, Python compilation,
  devcontainer JSON/JSONC generation, and GitHub workflow YAML parsing passed.
- Defects discovered by acceptance checks and fixed before closing:
  CPack ignore regex serialization and binary/source filename collision;
  duplicate Python `_wrapper_build.py` producers; installed ROS ORT RPATH;
  and symlink-install-relative `.ptafmodel` artifact resolution.
- Final scope/hygiene:
  the CTest inventory contains 21 target-owned tests and no generic
  template/logger/PTX/CMake conformance test; the only implementation-tree
  OptiX text is the mandatory PTX helper attribution plus an explicit
  no-dependency statement. `git diff --check` passes. Generated Python caches
  were moved to a recoverable `/tmp` quarantine, and colcon output is ignored;
  the pre-existing root `log/` and dirty `lib/wrap` submodule were not altered.
- External observations still unavailable locally: GitHub-hosted x86_64,
  arm64, Pages, guarded self-hosted CUDA, and ROS workflow runs have not been
  observed after a push; no Jetson hardware run was performed. These are
  deployment observations, not local implementation failures.

### Moved-tag CMake-test policy refresh

- Donor provenance:
  clean `cpp_cuda_template_project` `main` equals `origin/main`; annotated tag
  object `e5421cad88f8fc2806a9a0c0f63c945f1e782726` resolves to
  `dbffe4d4babb88a8682bfd19cf2bea3bc93fe252`, and `git tag -v v1.11.3`
  reports the expected good signature.
- Donor delta:
  relative to the initially audited
  `f3fd6555c7c965eebc89d5aa40227b0f36ac5fde`, only `AGENTS.md`,
  `doc/developments/derived_project_upgrade_agent_guidelines.md`,
  `doc/template_usage.md`, and `doc/testing_and_ci.md` changed (83 insertions,
  3 deletions).
- Policy remediation:
  removed the generated `benchmark_model_bad_target.cmake` and
  `get_available_providers_capability.cmake` CTest wrappers. The same
  deterministic target-owned behavior is registered directly against
  `benchmark_model` and `get_available_providers`; environment-dependent
  CUDA/TensorRT observation remains in the explicit Jetson/acceptance path.
- Reconfiguration/build:
  `cmake -S . -B build` and `cmake --build build --parallel 6` succeeded. The
  regenerated inventory contains 23 target-owned tests, no `test_template`, and
  no test whose command is CMake. The two orphaned generated wrapper scripts
  were removed from the existing build tree.
- Tests:
  `ctest --test-dir build --output-on-failure -R
  'benchmark_model|get_available_providers'` passed 6/6 in 0.06 seconds;
  `ctest --test-dir build --output-on-failure -j 6` passed 23/23 in 2.93
  seconds.
- Static hygiene:
  no source `tests/cmake`, `VerifyTemplateProject*`, or generated `.cmake` test
  wrapper remains. The sole non-documentation `${CMAKE_COMMAND} -P` source hit
  is the project uninstall custom target in `cmake/HandleCustomTargets.cmake`;
  it is not registered with CTest and is outside the prohibition.
  `git diff --check` passes.

### v1.12.2 follow-up acceptance

- Provenance: annotated tag object
  `1f60cebe98557bbba73a978f5a1243eb3c21f533` resolves to
  `490de59d9fc5ec09ab4d57a9dbf5d6e238c7209a`; `git tag -v v1.12.2`
  reports the expected good RSA signature. The exact current donor worktree
  patch remains byte-identical to
  `cpp_cuda_template_v1_12_2_dirty.patch` at SHA-256
  `1e251f34b353b393f9a3463565fb22ba553084cb49403513c133147b7a8db041`
  and stable patch ID `4645c110f7a48da523575f95faac5cb1500a87ce`.
- gtwrap ownership: `lib/wrap` is clean at
  `bf9f78617830bfa88c70d81a1a791c0a0c90b548`, equal to
  `origin/master` (`cpp_cuda_templ_reference_tag-55-gbf9f786`). An
  unauthorized configure request with `GTWRAP_SYNC_TO_MASTER=ON` and no
  maintenance grant failed as required; ordinary Python/MATLAB configures left
  the checkout unchanged.
- Safe cleanup: a conventional build directory whose cache belongs to the
  donor checkout was refused and its cache hash remained unchanged. A
  cache-proven target-owned build directory was safely removed and
  reconfigured.
- Native builds: fresh RelWithDebInfo and Debug plus
  `WARNINGS_ARE_ERRORS=ON` configurations each built and passed 23/23
  target-owned tests. The JSON CTest inventory contains no CMake command,
  `VerifyTemplateProject*`, recursive build, or template test.
- Packaging: the final binary and source TGZs are 169,646 and 172,302,505
  bytes. Both
  contain `Full version: 0.1.0+6.g16f1e26.dirty`; neither contains Git data,
  generated build/install output, root/ROS logs, Python caches, egg metadata,
  or `_wrapper_build.py`. The first source-package audit exposed a root `log/`
  leak; the final stricter audit also exposed nested example/Python build trees
  and the generated wrapper bootstrap. The exclusions now follow the
  repository's reserved `build*` output contract, and the regenerated
  extracted no-Git source passed 23/23 tests.
- Installed consumer: a fresh prefix and the tailored standalone
  `examples/consumer_project` configured, linked only
  `autoforge_deploy::autoforge_deploy`, and ran with tensor cardinality 11.
- Python: the full wrapper tree passed 25/25 tests. The 5,931,428-byte wheel is
  versioned `0.1.0+6.g16f1e26.dirty` and contains only `__init__.py`, the
  extension, `libautoforge_deploy.so`, and wheel metadata; `_wrapper_build.py`
  is absent. An isolated venv installed the wheel and real ORT inference
  returned `[1.686690330505371, 4.697796821594238]`.
- MATLAB: R2023b was discovered through `MATLAB_ROOT_DIR`; generation/build and
  all 24 tests passed. The target-owned smoke covers construction, role
  dispatch, caught invalid-shape recovery followed by valid inference,
  `clear classes`, and `clear mex`.
- Documentation: Doxygen 1.9.8 generated separate HTML and XML artifacts. All
  new/substantially changed C++/CUDA files have Doxygen file ownership and
  callable contracts, and all changed Python modules/callables have docstrings.
- CUDA plus TensorRT: CUDA 12.9.41, numeric architecture 89, and TensorRT 10.7
  configured together. Ordinary `placeholder.cu` compilation, dedicated
  `placeholder_to_ptx.ptx.cu` generation, `bin2c` embedding, and all 24 tests
  passed. The optional engine-inference case used its documented no-fixture
  path because `PTAFDEPLOY_TENSORRT_TEST_ENGINE` was not provided.
- ROS 2 Jazzy: a clean environment-root-only ONNX Runtime build completed all
  four packages. Colcon reported 13 tests, zero errors, zero failures, and zero
  skipped; installed standalone/composed and root/namespaced cases performed
  real facade inference.
- Static acceptance: all shell files parse; all workflow YAML, JSON, and Python
  source parse; workflow triggers implement automatic `main`, manual
  `develop`, and PR-to-`main`/`develop` policy; CUDA CI requires both CUDA and
  TensorRT. ROS metadata is synchronized, PTX exports the retained example
  kernel, conflict/forbidden-test scans are clean, and `git diff --check`
  passes.
- External observations remain pending until publication: GitHub-hosted
  x86_64/arm64, Pages, ROS, and guarded self-hosted CUDA/TensorRT runs have not
  been observed on the remote, and no Jetson hardware benchmark was run.

### Task 19 post-review correction evidence

- Finding 1 ownership: the source-writing `HandleWrapper.cmake` path was
  byte-identical in the target and signed donor v1.12.2, so it was a donor
  defect on a weakly observed path rather than target drift. Temporary pristine
  gtwrap checkouts remained byte-for-byte unchanged after the corrected target
  and donor wrapper configurations.
- TensorRT relocation: a CUDA/TensorRT-enabled install relocated through
  `/tmp/ptafdeploy-tensorrt-relocation-clean-Vj7aqq` configured, built, and ran
  an external consumer. Installed exports contain imported TensorRT targets,
  no producer include/library paths, and loader-relative `$ORIGIN/../lib`
  RUNPATHs on both the library and executable.
- Python ownership: target and donor wrappers now assemble metadata, stable
  package sources, wrapper links, wheels, installs, and stubs below each build
  tree. Source fingerprints and the shared gtwrap status remained unchanged.
  Isolated wheel installs imported successfully; target real ORT inference
  returned `[1.156921625137329, 1.049822211265564]`.
- Target acceptance: native CTest passed 21 cases with two target-owned fixture
  cases reported skipped; CUDA/TensorRT passed 22 with two skipped; the full
  Python build passed 25/25; MATLAB passed 22 with two skipped; ROS 2 Jazzy
  reported 13 tests with zero errors/failures/skips. Doxygen 1.9.8 regenerated
  HTML and XML with no warnings after correcting the `LoadModel` overload
  contract.
- Donor non-regression: default CTest passed 28/28, CUDA/PTX passed 31/31, the
  focused Python packaging verifier and an actual isolated wrapper wheel/import
  passed, and a clean ROS 2 Jazzy overlay reported 10 tests with zero
  errors/failures/skips. OptiX execution remains an explicit external gate
  because no SDK is installed on this host.
- Scope: Findings 3 and 5 remain deliberately deferred. No transactional reload
  behavior, real TensorRT engine fixture, DLA behavior, donor OptiX behavior, or
  derived-project template-conformance test was added in this correction pass.
- Staging review correction: the over-broad 165-file index and the intermediate
  106-file mixed index were cleared. The replacement target index contains 60
  reviewed build/configuration paths: devcontainer and issue-template support,
  `build_docs` plus its ONNX Runtime installer, CMake/build/package/wrapper
  infrastructure, approved `lib/wrap`, the Python runtime loader, and directly
  imported CLogger/CUDA/PTX files. The
  inference implementation and wrapper interfaces, native/CUDA/ROS workflows,
  ROS overlay, programs, tests, fixtures, model adapters/configs, prose docs,
  plan/patch archive, auxiliary implementation edits, and unrelated deletions
  remain unstaged. An exact temporary-index clone configured, built, and
  installed natively with tests disabled; its Python wrapper generated, built,
  and imported successfully, and generated package metadata retained the full
  version `0.1.0+6.g16f1e26.dirty`.
- Donor staging review: eight hunk-scoped files are staged independently with
  the required `TODO: review (imported from torchAutoForge-deploy)` marker.
  Release/version and container-launcher hunks remain unstaged. An index-only
  donor checkout built the actual Python wrapper and passed the import,
  packaging, and build-tree-package-root gates 3/3 without changing the
  external wrapper checkout status.
- Proposed scoped target commit title:
  `build: align CMake, CUDA/PTX, wrappers, and ROS 2`. The review description
  should call out template-derived build/configuration alignment, optional
  TensorRT discovery and relocatable exports, build-owned wrapper packaging,
  the ROS 2 overlay/CI, the retained CUDA/PTX examples, and OptiX exclusion.
