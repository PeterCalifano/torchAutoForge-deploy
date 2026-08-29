# AGENTS.md

Guidance for Codex when working in `torchAutoForge-deploy`.

## Project Overview

`torchAutoForge-deploy` is a C++20 deployment library for ONNX Runtime-backed
and optional TensorRT-backed ML inference. The current direction is:

- `ptafdeploy::inference::CInferenceManager` is the public backend-agnostic facade.
- `ptafdeploy::inference::CModelFacade` is the role-level facade for prototype
  model usage from C++/Python/MATLAB.
- `ptafdeploy::inference::onnxruntime::CInferenceManager_ORT` is the ONNX Runtime backend.
- `ptafdeploy::inference::tensorrt::CInferenceManager_TensorRT_Engine` is the
  optional standalone TensorRT serialized-engine backend when
  `autoforge_deploy_ENABLE_TENSORRT=ON`.
- Default builds keep TensorRT as a clear not-built path for lighter installs.
- Python/MATLAB wrappers use gtwrap interface files under `src/`; expose the
  generic facade and wrapper-safe value types, not raw ORT handles.
- OptiX is out of scope for this repo; keep this build lighter than renderer
  projects.

## Python Package Identity

- `autoforge_deploy` is the sole active Python distribution and import identity.
- Treat `python/autoforgeDeployPy` as protected legacy material until its useful
  content is classified, replaced, or relocated through the approved consolidation
  plan; do not preserve or publish it as a second active package identity.

## Repository Tailoring Authority

This repository's product and build contracts are authoritative. Treat
`cpp_cuda_template_project` as a source of reusable mechanics, not as a file-copy
baseline or a requirement for structural parity.

- Preserve ONNX Runtime as the required backend, standalone TensorRT as an
  optional backend, the independent CUDA/PTX feature, the four-package ROS 2
  overlay, current workflows, and the list-driven installed dependency replay.
- TensorRT may use `CUDA::cudart` without enabling the project CUDA language or
  PTX examples. Do not make `autoforge_deploy_ENABLE_TENSORRT` imply
  `autoforge_deploy_ENABLE_CUDA` merely because another repository does so.
- Never restore a removed dependency, helper, option, workflow, test,
  integration, automation, placeholder, or template verifier solely because it
  exists in the donor template.
- Keep dormant target utilities dormant unless a target-owned requirement
  explicitly activates them. Do not add PEP 440 tests or retrofit package names
  merely to demonstrate shared wrapper metadata behavior.
- For every template-derived change, classify the donor behavior as adopt,
  adapt, already present, or skip, and document target-specific adaptation when
  it affects an externally visible contract.

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
./build_lib.sh -D autoforge_deploy_ENABLE_CUDA=ON
./build_lib.sh -D autoforge_deploy_ENABLE_TENSORRT=ON \
  -D TENSORRT_ROOT=/path/to/TensorRT
./build_lib.sh -p                     # Enable Python wrapper generation
./build_lib.sh -m                     # Enable MATLAB wrapper generation
./build_lib.sh -p --wrap-update       # Explicit wrapper-checkout maintenance
./build_ros2.sh --clean               # Build and test the ROS 2 Jazzy overlay
./run_in_container.sh --build -- ./build_lib.sh
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
- model-role contract facade behavior;
- TensorRT load-error behavior, plus optional engine inference when
  `PTAFDEPLOY_TENSORRT_TEST_ENGINE` is set in an
  `autoforge_deploy_ENABLE_TENSORRT=ON` build.

Do not copy template conformance/static tests into this derived project. The
template implementation is validated in its owning repository; permanent tests
here must exercise `torchAutoForge-deploy` inference, wrappers, programs, or
ROS-facing behavior.

Do not import donor `VerifyTemplateProject*` scripts or register recursive
configure/build/install/package/consumer builds in ordinary project CTest.
Keep those configuration-matrix and delivery checks in explicit fresh
out-of-tree acceptance commands or CI. A permanent CMake-script test is allowed
only when it is lightweight, target-owned, non-recursive, and cannot be covered
through Catch2, pytest, an existing target, or the acceptance matrix. Native
ROS package tests remain the project-specific `ament_cmake_gtest`/launch-testing
exception.

## Key CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `autoforge_deploy_METADATA_ONLY` | OFF | Configure project identity/version without dependencies |
| `autoforge_deploy_ENABLE_CUDA` | OFF | CUDA build plus ordinary/embedded-PTX examples |
| `CUDA_PTX_USE_FAST_MATH` | ON | Fast math for dedicated PTX compilation |
| `CUDA_PTX_NVCC_FLAGS` | empty | Extra flags used only for PTX compilation |
| `autoforge_deploy_ENABLE_TENSORRT` | OFF | Standalone TensorRT `.engine` / `.plan` backend |
| `TENSORRT_ROOT` | empty | TensorRT install root for non-standard installs |
| `ENABLE_TESTS` | ON | Build and run tests |
| `SANITIZE_BUILD` | OFF | Enable sanitizers |
| `WARNINGS_ARE_ERRORS` | OFF | Add `-Werror` |
| `autoforge_deploy_BUILD_PYTHON_WRAPPER` | OFF | gtwrap Python bindings |
| `autoforge_deploy_BUILD_MATLAB_WRAPPER` | OFF | gtwrap MATLAB bindings |
| `GTWRAP_MAINTENANCE_UPDATE` | OFF | Explicitly permit wrapper-checkout updates |
| `GTWRAP_SYNC_TO_MASTER` | OFF | Request an update; requires maintenance permission |

ONNX Runtime discovery precedence is explicit `onnxruntime_DIR`, CMake
`ONNXRUNTIME_ROOT`, environment `ONNXRUNTIME_ROOT`, then normal config-package
discovery. Top-level callers may still use `PROJECT_METADATA_ONLY`,
`ENABLE_CUDA`, and `ENABLE_TENSORRT` as compatibility aliases, but new build
automation and nested consumers must use the project-qualified options.

## Language And Programming Standards

### Language-agnostic engineering

- Follow the owning component's established conventions and keep each change in
  the smallest coherent scope that satisfies the requested behavior.
- Prefer small cohesive functions and classes with explicit contracts. Add an
  abstraction only when it clarifies ownership, reuse, or a stable interface;
  keep one authoritative source for each policy or piece of state.
- Validate external inputs at system boundaries and report actionable failures.
  Do not silently fall back to behavior that changes the advertised contract.
- Test observable behavior, invariants, and failure modes rather than internal
  implementation details or tunable defaults.
- During review and optimization, seek behavior-preserving reductions in code
  complexity. Simplify unnecessary nested loops, helper functions, conditionals,
  indirection, and abstractions while improving performance, maintainability,
  readability, and clarity. Keep performance work evidence-driven.
- Use 100 columns as a soft limit. Keep assignments and function calls on one
  line when readable; otherwise wrap at semantic boundaries and align
  continuation lines with the expression they continue.
- Treat newlines as logical separators: keep statements implementing the same
  small step together and put a blank line between distinct steps.
- Introduce each non-obvious logical block with a concise purpose-, rationale-,
  or invariant-oriented comment. Do not translate individual statements into
  prose or rewrite unrelated legacy code during focused cleanup.

### Python

- Use Python 3.12 or newer, PEP 8 naming/formatting, and PEP 257 Google-style
  docstrings for modules and public classes, methods, and functions.
- Add precise type annotations to every function and method signature, class
  attribute, and dataclass field. Keep code suitable for static checking and
  isolate or justify unavoidable `Any` boundaries.
- Prefer dataclasses for stable records and enums for choices with more than two
  values. Prefer functions for stateless transformations and classes for state,
  ownership, or durable behavioral interfaces.
- Use PyTorch for ML implementations and scikit-learn for supporting workflows
  where useful. Preserve ONNX export compatibility unless explicitly excluded.
- Provide runnable examples or demos with expected output for new library-scale
  functionality. Use Matplotlib for general plots, seaborn for statistical
  plots, and Pillow or OpenCV for image-specific work as appropriate.

### C++ And CUDA

- Use the repository-configured C++20 standard. Retain older compatibility only
  where an owning target explicitly requires it, and use the supported CUDA
  version selected by this repository's build matrix.
- Use Doxygen file headers and Doxygen documentation for public classes,
  functions, and methods, including parameters, return values, templates,
  exceptions, ownership, and invariants where applicable.
- Prefer concepts over SFINAE. Prefer classes when invariants, ownership, or
  behavior must be enforced; use aggregates only when aggregate semantics are
  the intended contract.
- Use Catch2 for C++ and CUDA unit tests and follow surrounding target naming.
- Keep an assignment and the beginning of its right-hand expression on the same
  line when readable. Apply the same rule to a function name and its first
  argument; wrap long expressions at semantic operators or argument groups.
- Follow C++ standards guidance and Jason Turner-style best practices, and
  justify meaningful language-feature or ownership choices against simpler
  considered alternatives.

### MATLAB

- Prefer functions for stateless or performance-sensitive algorithms and
  classes only when state or ownership justifies them. Use `self` rather than
  `obj` for the receiver.
- Follow the established PascalCase Hungarian datatype prefixes, use descriptive
  names, keep code-generation-targeted algorithms codegen-safe, and use explicit
  input/output `arguments` blocks.
- Keep function definitions unnested. Use a trailing underscore for file-local
  helper names, and retain the repository's `SIGNATURE`, `DESCRIPTION`, `INPUT`,
  `OUTPUT`, `CHANGELOG`, and `DEPENDENCIES` documentation sections.

## Architecture

### Generic Inference API

- `src/inference/inference_common.h` defines backend/artifact enums, tensor
  descriptors, tensor views, owned buffers, model metadata, and runtime options.
- `src/inference/inference_manager.h/.cpp` provides facade dispatch by artifact
  type.
- `src/inference/model_facade.h/.cpp` provides a role-level contract facade
  above raw tensor inference. Keep YOLO, centroiding, and future model-specific
  adapters here or above, not inside ORT backend code.
- `src/inference/task_value_types.h/.cpp` owns generic wrapper-safe task
  values. `src/inference/task_adapters.h/.cpp` owns schema-driven tensor
  preprocessing and row decoding. Keep model-specific schemas and result
  interpretation in integrations while reusing these generic conversions;
  prefer C++20 templates/constrained helpers when they remain simpler and safer.
- `examples/model_configs/*.ptafmodel` are simple key-value role configs with
  repo-relative `artifact_path`, model role, preprocessing/postprocessing names,
  and runtime keys parsed into enum-backed config values.
- `src/programs/benchmark_model.cpp` is the generic CPU/GPU/Jetson timing path;
  keep it routed through `CModelFacade`, not backend-specific shortcuts.

### ONNX Runtime Backend

- `src/inference/onnx_runtime/onnxruntime_inference_tools.hpp/.cpp` implements
  `CInferenceManager_ORT`.
- `LoadModel()` validates `.onnx`, creates an ORT session, maps backend-neutral
  execution targets to ORT providers, and extracts model metadata.
- `Infer()` accepts host-memory `STensorView` inputs and returns owned
  `STensorBuffer` outputs.

### TensorRT Backend

- `src/inference/tensorrt/tensorrt_inference_engine.*` loads serialized
  `.engine` / `.plan` artifacts when
  `autoforge_deploy_ENABLE_TENSORRT=ON`.
- Keep this backend focused on runtime execution. Engine building remains a
  `trtexec`/external build-step concern for now.
- Current implementation uses host-staged dense tensor IO. Do not claim support
  for DLA, dynamic-output allocators, or precision/profile build controls until
  implemented and tested.

### CUDA and PTX examples

- `src/template_src_kernels/placeholder.cu/.cuh` is the retained ordinary-CUDA
  build example.
- `src/template_src_kernels/placeholder_to_ptx.ptx.cu` is the retained
  compile-and-embed PTX example.
- Keep dedicated `.ptx.cu` sources out of the ordinary CUDA source list.
- OptiX is not a supported option or dependency. PTX support must remain
  usable without OptiX.

### Wrappers

- `src/wrap_interface.i` and `src/inference/inference.i` expose
  `CInferenceManager`, `CModelFacade`, enum-backed runtime/model config, and
  wrapper-safe tensor/contract values.
- Prefer wrapping the generic facade and wrapper-friendly value types.
- Do not expose raw ORT handles, raw `void*`, or `std::byte` storage directly to
  MATLAB/Python.
- Ordinary configure/build operations must not update, initialize, or add the
  gtwrap checkout. Use `--wrap-update` or `--wrap-submodule-init` only as an
  explicit maintenance operation.
- The Python extension and project runtime are co-located in wheels. Keep
  external ONNX Runtime libraries as installation prerequisites; do not replace
  the target's build RPATH with a wheel-only loader path.

### ROS 2

- `ros2/ptafdeploy_ros` is the target-specific lifecycle bridge and must call
  `CModelFacade`, not backend-specific code.
- Keep the `FloatTensor` message wrapper-safe and validate cardinality through
  the core tensor helpers.
- `model_config_path` is required at configure time; inference is accepted only
  while active and exceptions must become failure responses.
- `ros2/ptafdeploy_spinup` owns installed fixtures and standalone/composed
  launch tests.

## Commit And Staged-review Workflow

### Commit-message style

- Do not use Conventional Commits prefixes such as `feat:`, `fix:`, or `docs:`.
- Write an imperative, sentence-case subject with no trailing period, normally
  about 50-70 characters. Add a short parenthetical scope only when useful.
- Use `[MAJOR]` only for a significant capability or broad contract change,
  `[BUGFIX]` for a correctness defect or regression, and `[HOTFIX]` for an
  urgent narrow correction. Use no tag for routine work.
- When a body is useful, use imperative `-` bullets, one blank line between
  bullets, no terminal periods, and aligned continuation lines. Explain intent,
  design decisions, and behavioral consequences rather than listing files.
- Never add `Co-Authored-By` or other AI-attribution trailers.

### Authorization and review sequence

1. Never create or amend a commit unless the user explicitly instructs the
   agent to commit. Treat commit, tag, and push authorization independently;
   `next`, implement, finish, or stage does not grant commit permission.
2. Inspect the complete worktree and index before preparing a batch. Preserve
   and report unrelated user-owned staged and unstaged work; never reset,
   overwrite, or absorb it to simplify the batch.
3. Partition completed work into coherent functional batches with dependent
   tests and necessary documentation. Avoid micro-batches. A clearly labelled
   mixed batch is acceptable only for a few small miscellaneous changes that do
   not justify independent review units.
4. Before staging, review the full candidate diff for correctness, scope,
   formatting, comments, documentation, and unnecessary complexity. Run
   proportionate tests and static checks.
5. Stage only the reviewed batch through an explicit path or hunk allowlist.
   Inspect the complete index with `git diff --cached`, run
   `git diff --cached --check`, and repeat any validation affected by the
   staged representation.
6. Report staged paths, functional purpose, evidence, caveats, exclusions, and
   the exact proposed commit subject/body. Stop for user review without
   preparing another batch.
7. Advance only after the exact keyword `next`. Before advancing, confirm the
   previous batch is no longer staged; if it remains staged, stop for an
   explicit commit or index-clear instruction.
8. When explicitly asked to `commit and next`, commit the approved batch with
   the reviewed message, verify the index is clear, and only then prepare the
   next batch.

For every new or substantially modified source file in a staged batch, review
the staged result as a reader receives it. Ensure applicable file/module and
public-API documentation are present, related statements form clear logical
blocks, non-obvious blocks have concise contract-oriented comments, and useful
existing documentation is preserved. Limit cleanup to the batch's intended
scope.

## Development Priorities

1. Maintain the completed semantic alignment with
   `/home/peterc/devDir/dev-tools/cpp_cuda_template_project` signed tag
   `v2.0.1` at `1d87153b2d060bf03c2c9adcd1df6c6d4f40ea09`, while preserving
   ORT-specific dependency wiring, independent TensorRT/CUDA policy, and the
   intentional absence of OptiX. Keep the earlier v1.11.3/v1.12.2 upgrade plan
   only as historical provenance for the protected staged batch.
2. Make wrappers work for MATLAB/Python through a stable generic inference
   facade.
3. Review current implementation for redundancy and readability before adding
   higher-level model support.
4. Add model-role facade for centroiding, object detection, feature matching,
   tracking, optical flow, and future learned modules.
5. Add YOLO and centroiding ONNX examples without hardcoding those models into
   the core ORT backend.
