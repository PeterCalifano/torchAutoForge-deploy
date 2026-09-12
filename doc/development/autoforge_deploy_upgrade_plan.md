# torchAutoForge-deploy Upgrade Plan

Design rule for remaining release work: keep API, wrappers, configs, CLIs, docs,
and tests aligned around one coherent model/runtime contract. Do not add
release-only shortcuts that bypass validation or duplicate behavior just to make
a demo pass.

## Stage 1: Documentation Refresh

- [x] Update `README.md` status to describe current ORT metadata extraction, host-memory inference, generic facade, TensorRT stub, and wrapper gaps.
- [x] Update `AGENTS.md` to remove obsolete constructor-mismatch and `initialize<T>()` guidance.
- [x] Update `TODO` to separate closed local WIP from remaining work.
- [x] Create this staged plan in `doc/development/`.
- [x] State that implementation must remain general for future model classes, not tailored only to current YOLO or centroiding models.

## Stage 2: CMake Template Sync

- [x] Use `/home/peterc/devDir/dev-tools/cpp_cuda_template_project` at local commit `c84b950` as CMake/template source of truth.
- [x] Port prerelease/metadata version handling and `FULL_VERSION` packaging behavior.
- [x] Port `EXTRA_C_FLAGS` / `EXTRA_CXX_FLAGS` handling so build helper flags are appended without clobbering project-managed flags.
- [x] Port wrapper diagnostics and Python package workflow improvements from template.
- [x] Preserve ONNX Runtime discovery, `ort_target_interface`, and current inference-library wiring.
- [x] Keep optional CUDA support.
- [x] Prune OptiX user-facing support from this repo for lighter build configuration.
- [x] Skip template-only spdlog and ZeroMQ additions because current code does not use them.

### Stage 2 follow-up: v1.11.3 semantic sync

The earlier sync above remains the historical baseline. The active follow-up is
tracked in
`doc/development/cpp_cuda_template_v1_11_3_upgrade_plan.md` and upgrades from
`c84b950` to signed donor tag `v1.11.3`
(`f3fd6555c7c965eebc89d5aa40227b0f36ac5fde`).

- [x] Import applicable non-OptiX v1.11.3 build, packaging, PTX, documentation,
  wrapper, development-environment, CI, logger, and ROS 2 capabilities.
- [x] Preserve target identity, ORT/TensorRT behavior, wrappers, and all
  target-owned tests.
- [x] Keep the CUDA and PTX placeholder kernels as examples, but remove all
  unrelated template scaffolding and the generic always-pass test.
- [x] Add no permanent tests for generic template functionality; validate only
  target integration and target-owned behavior.

## Stage 3: Wrapper Configuration

- [x] Use `ptafdeploy` as the project namespace and `ptafdeploy::inference` as the nested namespace for inference APIs; do not keep the old `deploy_infer` namespace.
- [x] Make `src/wrap_interface.i` top-level namespace and includes match `ptafdeploy`.
- [x] Populate `src/inference/inference.i` around `ptafdeploy::inference::CInferenceManager` and wrapper-safe value types.
- [x] Expose model loading, metadata query, and float host-buffer inference to Python/MATLAB.
- [x] Avoid wrapping raw ORT handles, raw `void*`, and `std::byte` storage directly.
- [x] Add Python import smoke test.
- [x] Add MATLAB wrapper generation and model-load smoke test.
- [x] Set MATLAB-safe runtime defaults, especially ORT thread count, while allowing explicit overrides.

## Stage 4: Implementation Review And Cleanup

- [x] Review `ptafdeploy::inference` tensor types against legacy `SInputOutputSpecs` and `SImagesInputOutputSpecs`.
- [ ] Remove or deprecate redundant legacy specs where covered by generic descriptors/views/buffers.
  The headers remain installed at `2abdb68`; the Stage 18 pre-stage retirement
  candidate below corrects this earlier completion claim.
- [x] Audit ORT backend helper functions for readability, reuse, and error clarity.
- [x] Remove useless one-off local helpers that only hide simple operations.
- [x] Add tests for named input mapping, mixed named/unnamed rejection, dtype mismatch, shape mismatch, byte-count mismatch, null data, and unloaded-session error.
- [x] Add numeric reference test for ORT output values.

## Stage 5: General Model Facade

- [x] Add model-role layer above raw tensor inference, not inside ORT backend.
- [x] Define generic model contract for artifact path, role, input descriptors, output descriptors, preprocessing, and postprocessing.
- [x] Support initial roles: raw tensor model, centroiding model, object detection model.
- [x] Keep registry extensible for feature matching, tracking, optical flow, and future learned modules.
- [x] Avoid hardcoded model names and repo-specific paths in core library code.
- [x] Provide MATLAB-friendly facade that hides raw tensor plumbing but keeps metadata and errors visible.

## Stage 6: YOLO And Centroiding ONNX Support

- [x] Locate real YOLO and centroiding ONNX artifacts from related repos or repo-local model folders.
- [x] Add repo-local example configs instead of absolute model paths.
- [x] Implement YOLO model loading through the generic object-detection role.
- [x] Implement centroiding through the generic centroiding role.
- [x] Add C++ smoke tests for role config parsing and facade dispatch.
- [x] Add MATLAB smoke tests for loading both roles and running one fixture inference.
- [x] Add concrete YOLO-style HWC-to-NCHW preprocessing and raw detection postprocessing adapters.
- [x] Add concrete centroiding output extraction for x/y/confidence tensors.
- [x] Keep adapter conversion helpers shared and allocation-conscious; avoid duplicated local conversion logic.

## Stage 7: Runtime Configuration And Benchmarks

- [x] Replace free-form public runtime configuration strings with enum-backed model role, backend, artifact, and execution-target choices.
- [x] Parse `.ptafmodel` text runtime keys into enums at the file boundary and reject unsupported values before model load.
- [x] Share text-to-enum parsing between `.ptafmodel` loading and benchmark CLI.
- [x] Route execution-target priority, fallback policy, device id, and thread counts through `CInferenceManager` and `CModelFacade`.
- [x] Report requested targets plus applied/available ORT providers in backend metadata.
- [x] Add a generic `benchmark_model` CLI that loads `.ptafmodel` or raw artifacts through `CModelFacade`.
- [x] Add Jetson-specific benchmark documentation and expected command lines.
- [x] Add ORT provider preflight CLI support for required execution targets.
- [x] Add a manual Jetson runtime smoke runner for ORT CUDA, ORT TensorRT EP, and standalone TensorRT engine paths.
- [x] Add dry-run CTest coverage for the Jetson smoke runner so hosted CI validates CLI wiring without Jetson hardware.
- [x] Add TensorRT serialized-engine runtime optimization-profile selection.
- [x] Validate ORT CUDA target selection on local GPU.
- [x] Validate ORT TensorRT target fallback to CUDA when TensorRT EP is unavailable.
- [x] Validate standalone TensorRT serialized-engine load/inference on local GPU.
- [ ] Validate CUDA/TensorRT runtime path on Jetson hardware.
- [ ] Validate ONNX Runtime TensorRT Execution Provider when an ORT build exposing `TensorrtExecutionProvider` is available.

## Stage 7.5: Manifest Schema

- [x] Define `.ptafmodel` as a versioned model manifest, not an ONNX replacement.
- [x] Add `schema_version = 1` as a required key.
- [x] Document required keys, optional keys, enum domains, defaults, and path resolution.
- [x] Add a repo-local template manifest.
- [x] Keep manifest runtime fields backend-neutral; map targets to ORT providers only inside the ORT backend.

## Stage 8: Adapter Layer Generalization

- [x] Keep model adapters above `CModelFacade`/raw tensor inference, not inside ORT backend code.
- [x] Add wrapper-safe value types for centroid and object-detection outputs.
- [x] Add MATLAB vector/matrix bridge helpers for image preprocessing, centroid extraction, and YOLO raw detection decoding.
- [x] Cover adapter behavior with focused synthetic tests instead of model-file-specific hardcoded output values.
- [ ] Add optional NMS and box-format adapters once a target detector contract requires them.

## Stage 9: Real Centroiding ONNX Integration Example

Read-only discovery targets:

- `/media/peterc/SCRATCH/ml-outputs/ml-based-centroiding`
- `/home/peterc/devDir/ML-repos/ml-based-centroiding`

Current discovery result: exported ONNX exists at
`/media/peterc/SCRATCH/ml-outputs/ml-based-centroiding/onnx_checkpoints/best_model_film_20260305_155227/best_model_film_20260305_155227_0.onnx`.
The graph contract is `image` float32 `[-1, 1, 1536, 2048]`,
`prior_vector` float32 `[-1, 3]`, `prediction` float32 `[-1, 2]`, and
`centre_of_brightness` float32 `[-1, 2]`. The three prior channels follow the
export/evaluation code contract: phase angle in radians, sun direction angle
from +x in radians, and log-fixed-bounds-normalized reference size.

No dataset image plus per-sample labels were found in the requested output tree.
The real-artifact regression therefore uses a repo-local synthetic Lambertian
sphere fixture whose image phase/sun direction/reference-size prior are derived
from the same sample parameters. The asserted label is the image-derived
brightness centroid, not a random prior or unrelated aggregate metric. This
keeps auxiliary inputs coherent with image content while avoiding writes to the
external centroiding repo.

- [x] Stop implementation of this example until an exported centroiding `.onnx` exists in the requested locations or a concrete exported path is provided.
- [x] Select one exported centroiding ONNX artifact and record its expected input names, shapes, dtype, output names, output scaling, and preprocessing contract.
- [x] Select one sample image plus corresponding centroid labels from available dataset roots, not from aggregate eval JSON alone; use a coherent synthetic fixture because no per-sample dataset files were found in the requested locations.
- [x] Add a repo-local `.ptafmodel` example config for the selected centroiding ONNX without hardcoding machine-local paths into core library code.
- [x] Implement a C++ example/regression test that loads the selected `.ptafmodel` through `CModelFacade`, runs ORT inference, and checks output contract plus a tolerance-based `centre_of_brightness` sanity assertion against the synthetic sample label.
- [x] Keep the real-artifact test optional or fixture-gated if the ONNX/data cannot be stored in this repo due to size/licensing.
- [x] Run the real centroiding ONNX through the CPU ORT target.
- [x] Run the real centroiding ONNX through the CUDA ORT target when the provider is available locally.

## Stage 10: CI And Published API Documentation

- [x] Replace the placeholder C++ workflow with a real Ubuntu 22.04 build/test workflow.
- [x] Add native x86_64 CI coverage on `ubuntu-22.04`.
- [x] Add native arm64 CI coverage on `ubuntu-22.04-arm` for CI-runnable CPU tests.
- [x] Cache the CMake build folder and downloaded ONNX Runtime dependency per OS, architecture, build type, ORT version, and source/config hash.
- [x] Keep ARM CI CPU-only; do not require Jetson CUDA/TensorRT hardware in hosted CI.
- [x] Add x86_64 Python wrapper CI coverage with gtwrap generation and `autoforge_deploy_python_facade_smoke`.
- [x] Keep MATLAB wrapper tests out of hosted CI; they remain local validation because MATLAB licensing/runtime is machine-specific.
- [x] Add Doxygen HTML documentation build in CI.
- [x] Add GitHub Pages deployment for Doxygen HTML on `main`/`master` pushes.
- [ ] Confirm GitHub CI passes on x86_64, arm64, and Python wrapper hosted runners.
- [ ] Confirm GitHub Pages is enabled in repository settings after first successful docs artifact upload.

## Stage 11: Release Design And Documentation Review

- [x] Review API layering for coherent facade/backend/adapters/wrappers split.
- [x] Review runtime configuration for enum-backed choices and early validation.
- [x] Review TensorRT/Jetson runtime controls for serialized-engine prototype use.
- [x] Review Python/MATLAB wrapper generation and hosted-CI wrapper policy.
- [x] Review README, manifest schema, Doxygen output, and release status docs.
- [x] Record review evidence in `doc/development/release_design_review.md`.

## Stage 12: Post-v2.0.1 Product API Consolidation

This stage starts from committed build/config checkpoint `432f39d`. It
consolidates the existing product-owned inference and wrapper implementation
without folding in the later execution-session redesign proposed by the
portability audit.

- [x] Approve `CInferenceManager` as the backend-neutral facade and
  `CModelFacade` as the model-role facade for this consolidation.
- [x] Consolidate generic tensor, runtime, and model-contract value types.
- [x] Consolidate ONNX Runtime dispatch and the optional standalone TensorRT
  engine path required by `CInferenceManager`.
- [x] Consolidate model-role parsing and the shared preprocessing and
  postprocessing adapters used by `CModelFacade`.
- [x] Replace the deleted template placeholder wrapper with the product
  Python/MATLAB interface and wrapper-safe bridge functions.
- [x] Include only target-owned inference, facade, adapter, and wrapper tests;
  do not restore template conformance or recursive CMake tests.
- [x] Correct stale release-review claims about DLA qualification, Pages
  deployment, and historical verification evidence.
- [x] Verify native C++ tests, strict warnings, Python import/facade behavior,
  MATLAB wrapper generation/runtime smoke, TensorRT configuration, install,
  and an installed consumer from the exact candidate index.
- [x] Stage the reviewed product batch through an explicit allowlist and freeze
  it for consolidation review without committing.
- [ ] Design the private type-erased/transactional execution-session layer as
  a subsequent change on top of the consolidated facade API.

Review gate for this stage:

- [x] Review the complete candidate diff as the public API reader receives it.
- [x] Run the native, strict-warning, wrapper, TensorRT-configure, install, and
  installed-consumer checks from a fresh out-of-tree candidate build.
- [x] Stage only the approved product API, CLI, wrapper, target-owned tests,
  fixtures, and directly supporting documentation.
- [x] Inspect the complete cached diff and run `git diff --cached --check`.
- [x] Freeze the reviewed index for consolidation review without committing;
  later demo implementation may continue only in separate unstaged paths until
  this batch is cleared.

## Stage 13: YOLO Facade Demos

Implementation may begin in separate unstaged paths after the Stage 12 index is
frozen, but no demo batch may be staged until that index has been reviewed and
cleared. The demos exercise the existing object-detection facade and shared
adapters; they do not add YOLO policy to the ORT backend or redesign either
public facade.

Verified external fixture snapshot:

- model SHA-256
  `99b8030554ed03fa15be185acd58065df6c64c2a79af392a419c87af857aaf63`;
- image SHA-256
  `c8f0a677a1356569e2ce71d2fa88c1030c0ae57ecf5e14170e02d9a86a20dcb4`;
- contract `images` float32 `[1,3,640,640]` to `output` float32
  `[1,25200,85]`; and
- native, Python, and MATLAB CPU/CUDA runs all report score-sorted class-17
  detections, with small resize/provider-dependent numeric differences.

### Combined Demo Batch

- [x] Verify the selected YOLO ONNX artifact, manifest, preprocessing shape,
  raw output layout, and one reproducible inference path.
- [x] Keep the artifact external when size or licensing prevents committing it;
  provide a clear path/config override instead of a machine-local default.
- [x] Add a native C++ CLI demo that loads through `CModelFacade`, preprocesses
  through the shared HWC-to-NCHW adapter, runs inference, and decodes through
  the generic detection-row adapter.
- [x] Remove the unused oneTBB dependency from the standalone native consumer
  and prove that configuration succeeds with TBB discovery disabled.
- [x] Add only target-owned parsing, validation, and observable-output tests;
  do not test generic template, CMake, or gtwrap functionality.
- [x] Validate the standalone CLI under strict warnings and run target-owned
  help plus real CPU inference smokes; also run a real CUDA inference manually.
- [x] Add a runnable Python demo using the generated `CModelFacade` wrapper and
  wrapper-safe adapter values rather than raw ORT handles.
- [x] Document prerequisites, invocation, and expected summary output.
- [x] Run the demo against the verified artifact on CPU and CUDA.
- [x] Add a runnable MATLAB demo using the generated `CModelFacade` wrapper and
  MATLAB bridge helpers.
- [x] Follow repository MATLAB argument/documentation conventions and report
  actionable artifact/image/shape errors.
- [x] Run the demo against the verified artifact on CPU and CUDA and pass
  MATLAB `checkcode` without findings.
- [x] Stage the native, Python, MATLAB, manifest, target-owned tests, and demo
  documentation as one coherent allowlisted batch, then continue directly to
  ROS 2 while retaining the staged files as explicitly authorized.

### Deferred Design

- [x] Keep NMS, alternate YOLO output layouts, image rendering, and a general
  adapter registry out of the first demos until a concrete artifact requires
  those contracts.

### Stage 13.1: Native CLI Parsing Consolidation

Use only the local TCLAP checkout at
`/home/peterc/devDir/dev-tools/tclap` commit
`dcabc731db60b46ef72cdc6f9a4eec5311f6393e`. Its header set is byte-for-byte
identical to the copies already used by local Spectra-RT and
`future-onboard-sw`; include its `COPYING` file rather than reproducing the
incomplete license bundle present in those two consumers.

- [x] Import the exact local TCLAP headers and license without downloading or
  editing third-party implementation files.
- [x] Define one source-scoped CMake `INTERFACE` target with third-party
  headers treated as system includes; keep it out of the core library,
  wrappers, installed package contract, and ROS 2 overlay.
- [x] Replace duplicated native CLI token/value parsing in `benchmark_model`,
  `get_available_providers`, and the standalone YOLO demo while preserving
  target-owned enum, range, shape, and runtime validation.
- [x] Keep normal help output on stdout, route actionable application failures
  through `CLogger`, and prevent TCLAP from terminating before application
  cleanup and exit-code handling.
- [x] Add or retain only observable product-CLI acceptance tests; do not import
  or recreate TCLAP, template, CMake, or generic parser tests.
- [x] Pass strict native build/tests plus standalone YOLO help and real CPU
  inference acceptance before updating the staged review batch.

Acceptance snapshot on 2026-08-28:

- the imported headers and license exactly match local TCLAP commit
  `dcabc731db60b46ef72cdc6f9a4eec5311f6393e`;
- a strict native build passed all 33 registered tests, with only the two
  externally gated centroiding fixture cases skipped;
- a programs-disabled build contained no TCLAP target or dependency, and the
  installed package exported neither TCLAP headers nor a TCLAP CMake target;
  and
- the standalone YOLO consumer passed strict compilation, its help contract,
  and real CPU inference against the recorded external fixture.

## Verification Gates

- [x] Clean configure in `/tmp` with default options.
- [x] Build default target.
- [x] Run `ctest --output-on-failure`.
- [x] Configure and build Python wrapper target with current wrapper interfaces.
- [x] Configure and build MATLAB wrapper target with current wrapper interfaces.
- [x] Confirm no OptiX option or module is required.
- [x] Configure and build `ENABLE_TENSORRT=ON` against local TensorRT install.
- [x] Run optional TensorRT engine inference regression with generated `.engine`.
- [x] Run `benchmark_model` CTest coverage for `.ptafmodel`, raw ONNX, and invalid runtime target handling.
- [x] Add `get_available_providers --require-targets` CTest coverage for ORT provider preflight.
- [x] Add dry-run CTest coverage for the Jetson runtime smoke runner.
- [x] Add parser/CLI regression coverage for TensorRT profile runtime options.
- [x] Install to a staged prefix and verify an external consumer builds/runs against `autoforge_deploy::autoforge_deploy`.
- [x] Generate binary TGZ package with CPack and verify installed export no longer leaks OpenCV linkage.
- [x] Run real centroiding ONNX C++ example/regression with CPU ORT and coherent synthetic labeled fixture.
- [x] Run real centroiding ONNX C++ example/regression with CUDA ORT and coherent synthetic labeled fixture on the local CUDA provider.
- [x] Add CI config for Ubuntu 22.04 x86_64 and arm64 build/test with build-folder cache.
- [x] Add CI config for x86_64 Python wrapper build/test with build-folder cache.
- [x] Add CI config for Doxygen HTML and GitHub Pages deployment.
- [x] Run local CI-style x86_64 Python wrapper build and `autoforge_deploy_python_facade_smoke` against downloaded ORT release layout.
- [x] Build MATLAB wrapper target locally without running MATLAB smoke tests.
- [x] Run TensorRT-enabled CTest locally; optional engine test passes as a no-engine assertion unless `PTAFDEPLOY_TENSORRT_TEST_ENGINE` is set.
- [x] Verify source Python package install from `<repo>/python` in a temporary venv.
- [x] Complete release design/documentation review.
- [ ] Observe a green GitHub Actions run for x86_64, arm64, Python wrapper, and docs jobs.

## Stage 14: Generic Adapter And Source-Owned Tooling Review

This stage revises the protected Stage 12 batch in place following API and
maintainability review. It preserves the approved manager/facade design and the
template-derived wrapper build machinery. Only target-owned adapter, wrapper,
program, test, and tooling behavior belongs to this stage.

### Stage 14.0: Baseline And Scope

- [x] Record protected branch `feature/ort_manager_pipeline_wrapped` at
  `432f39d15f90f8aef6ac6174c50214398a6d6741`.
- [x] Record original cached-diff SHA-256
  `39ba8ae1bf0b29d8c21f0ac02875f29b1bdd9834fffd2ddd6e835b4d348262bc`.
- [x] Preserve reviewer-owned unstaged TODOs and all unrelated dirty paths.
- [x] Classify `HandlePythonWrapper.cmake`,
  `StagePythonRuntimeArtifacts.cmake`, setup/pybind fallbacks, and wrapper
  metadata generation as adopted template mechanics that remain unchanged.
- [x] Classify the Python/MATLAB bodies in the target test CMake file and the
  target Jetson smoke runner as source-ownership corrections in this stage.

### Stage 14.1: Generic Geometry And Task Adapters

- [x] Replace centroid- and detector-specific flat records with composed
  point, size, bounding-box, feature, class-score, and detection records.
- [x] Store boxes canonically as center plus size and provide validated center
  `xywh` and corner `xyxy` conversion surfaces.
- [x] Replace centroid/YOLO-named decoding with schema-driven feature-row and
  dense-detection decoding.
- [x] Keep HWC-to-NCHW conversion generic and allocation-conscious.
- [x] Validate axes, tensor cardinality, schema indices, finite values, box
  geometry, thresholds, class selection, score ordering, and result limits.
- [x] Cover the target-owned behavior with focused Catch2 tests using a
  verified red-green cycle.

### Stage 14.2: Wrapper And MATLAB Boundary

- [x] Wrap the composed geometry records, schemas, conversion helpers, and
  generic task decoders for Python and MATLAB.
- [x] Remove centroid/YOLO policy from `inference_matlab_adapters.h`.
- [x] Keep the MATLAB bridge limited to transparent tensor/vector/shape and
  facade-call conversions.
- [x] Preserve raw-handle, `void*`, and `std::byte` exclusions.
- [x] Verify composed records directly in Python; use a generic schema-driven
  numeric-matrix bridge in MATLAB because gtwrap does not emit a MATLAB class
  for `std::vector<SDetection2D>`, without changing the facade design.

### Stage 14.3: Documentation And Logging

- [x] Review every new or substantially modified C++ file for Doxygen file and
  public-API contracts, ownership, invariants, failure modes, and rationale
  comments.
- [x] Use the existing `CLogger` for human diagnostics and useful load,
  provider, configuration, and dispatch details.
- [x] Keep machine-readable result records and help text stable.
- [x] Route logger output away from stdout and avoid duplicate log-and-throw
  reporting.

### Stage 14.4: Source-Owned Smoke Programs

- [x] Move the target Python facade smoke body from CMake to a tracked Python
  3.12 script with typed arguments and actionable failures.
- [x] Move the target MATLAB facade smoke body from CMake to a tracked MATLAB
  function following repository argument and documentation conventions.
- [x] Replace the target Jetson shell implementation with a typed Python CLI,
  preserving its options, command sequence, dry-run record, and exit behavior.
- [x] Leave imported wrapper-generation code and established template shell
  entrypoints unchanged.

### Stage 14.5: Demo Compatibility And Deferred Design

- [x] Update the unstaged native, Python, and MATLAB YOLO demos to supply the
  generic detection schema at the integration boundary.
- [x] Update the unstaged real-centroiding test to consume generic feature rows.
- [x] Keep demos and external-artifact tests out of the protected staged batch.
- [x] Record model-family inventory/registry, private type-erased execution
  sessions, transactional reload, stateful inference, capability reporting,
  and manifest-v2 work as deferred designs.

### Stage 14.6: Validation And Review Gate

- [x] Pass native strict build and all target-owned CTest cases.
- [x] Pass Doxygen HTML/XML generation with warnings as errors.
- [x] Pass Python wrapper, build-tree smoke, and wheel/import acceptance.
- [x] Pass MATLAB wrapper generation, `checkcode`, and runtime smoke when the
  local licensed runtime is available.
- [x] Pass CUDA/PTX build acceptance without adding placeholder tests when the
  CUDA toolchain is available.
- [x] Pass TensorRT configuration/load-error acceptance and run real engine
  inference only when a compatible engine is supplied.
- [x] Pass install/package and installed-consumer acceptance.
- [x] Re-run available native/Python/MATLAB YOLO demos and record unavailable
  external assets or hardware as explicit unrun boundaries.
- [x] Rebuild the exact staged batch through path/hunk allowlists, inspect the
  full cached diff, run `git diff --cached --check`, and record its new digest.
- [x] Stop for review without committing, pushing, or preparing another batch.

Acceptance snapshot on 2026-08-26:

- staged content SHA-256 excluding this self-referential tracker file is
  `83942d975a3bdb6cc6dffd576596a605cd76b8ddbe97d8f2512977099861e6df`;
- exact-index strict native build completed with `WARNINGS_ARE_ERRORS=ON`; all
  28 registered target-owned tests passed;
- the separate worktree build registered two additional opt-in
  real-centroiding fixture cases, both skipped by contract because the external
  fixture was not supplied;
- Doxygen HTML/XML completed with warnings treated as errors;
- Python build-tree smoke, wheel construction/install, and installed-wrapper
  smoke passed with the extension and project shared library co-located;
- MATLAB R2024b wrapper generation, source `checkcode`, and runtime smoke passed;
- CUDA 12.9 built the retained ordinary CUDA and embedded-PTX examples without
  adding tests for template-owned placeholder mechanics;
- TensorRT 10.7 configured and built with project CUDA disabled, and both
  TensorRT target-owned tests passed; no compatible serialized-engine fixture
  was supplied, so real engine execution was not rerun;
- install, TGZ packaging, and a fresh installed consumer passed; and
- native, Python, and MATLAB YOLO demos ran on both CPU and CUDA against the
  recorded local artifact and image, each returning five score-sorted class-17
  detections.

## Stage 15: ROS 2 YOLO Lifecycle Integration

- [x] Audit the existing four-package ROS 2 Jazzy overlay against current
  `CModelFacade`, manifest, lifecycle, and installed-fixture contracts.
- [x] Keep the reusable lifecycle bridge tensor-generic and place YOLO image
  preprocessing and detection interpretation in integration-owned code.
- [x] Add target-owned ROS 2 tests for lifecycle gating, tensor cardinality,
  inference failure propagation, and standalone/composed launch behavior.
- [x] Exercise a real YOLO inference through the ROS 2 lifecycle/service path
  when the verified external model and image are available.
- [x] Run a clean `colcon` build/test/install acceptance for the complete
  overlay and inspect test results.
- [x] Stage the reviewed ROS 2 package, helper, documentation, and CI-facing
  integration through an explicit allowlist while preserving the YOLO batch.

Acceptance snapshot on 2026-08-27:

- all four Jazzy packages built and installed from a clean overlay with CUDA
  disabled and an explicit local ONNX Runtime package;
- 15 target-owned ROS tests passed with no failures or skips, including the
  real external YOLO request through the lifecycle service and all four
  standalone/composed root/namespaced launch cases; and
- shell syntax/static analysis, Python byte-compilation, XML parsing, explicit
  C++ formatting, and staged-diff whitespace checks passed where available.

Deferred follow-on after ROS 2 consolidation:

- [x] Remind the user to review the design and implementation of the plain
  centroiding model for product use.

## Stage 16: One-Shot ONNX Tensor Inference CLI

Keep `run_ort_inference` backend-generic at the tensor boundary while making its
name truthful: the default operation must execute one ONNX inference, and an
explicit metadata-only mode must preserve the existing inspection use case.

- [x] Add target-owned CLI tests first and observe the expected failures for
  real zero-filled inference, missing input data, and documented help behavior.
- [x] Accept repeated raw float32 inputs using `[name=]path.f32`, repeated
  `[name=]d0,d1,...` dynamic-shape overrides, and repeated `[name=]value`
  deterministic fills.
- [x] Infer concrete static shapes from model metadata; require explicit shape
  overrides for dynamic dimensions and explicit names for ambiguous multi-input
  models.
- [x] Map execution-target, device, fallback, and thread options to the existing
  backend-neutral runtime configuration without introducing ORT handles.
- [x] Print stable metadata and bounded output previews, and optionally persist
  each output as raw float32 under `--output-dir`.
- [x] Validate names, duplicate specifications, tensor dtype, shape cardinality,
  raw file size, output destination, and mutually exclusive file/fill sources
  with actionable logger diagnostics.
- [x] Document raw tensor layout, multi-input rules, dynamic shapes, output
  naming, log-level control, examples, and exit behavior in the user README and
  CLI help.
- [x] Run focused red-green-refactor checks, the complete native target-owned
  suite, strict compilation, and staged-diff validation.
- [x] Stage only the reviewed CLI, its target-owned tests, documentation, and
  this tracker update, then stop for review.

Acceptance snapshot on 2026-08-28:

- the initial metadata-only implementation failed the four new execution/help
  assertions for the intended reasons before production code was replaced;
- the follow-up metadata/result namespace and ignored-option checks each failed
  before their focused corrections and passed afterward;
- a fresh `WARNINGS_ARE_ERRORS=ON` build completed with the explicit local ORT
  config package, and all 38 registered native tests passed; the two external
  FiLM centroiding fixture cases remained skipped by their existing contract;
- an archive reconstructed from `HEAD` plus only the cached patch passed its
  warnings-as-errors build and all 36 tests, proving that the batch does not
  depend on unrelated unstaged headers or the untracked FiLM test;
- both deterministic fill and checked-in raw float32 inputs executed through
  ORT, and the persisted two-float output was verified as eight bytes; and
- no template-conformance, recursive CMake, Python, or generic helper test was
  added.

## Stage 16A: Installed C++ Parsing Utilities

Consolidate the native programs' repeated value parsing behind a small,
framework-independent installed API. Keep TCLAP responsible for command-line
collection and help text while making the reusable subgrammars available to
downstream C++ consumers without exposing TCLAP as a public dependency.

- [x] Confirm that Stage 16 has left the index and record the unrelated dirty
  work that this new commit batch must preserve.
- [x] Add focused failing Catch2 coverage for optional-name specifications,
  integer lists, finite floats, positive tensor shapes, and model-input
  resolution.
- [x] Implement allocation-conscious generic parsing with `std::string_view`
  and `std::from_chars`, including actionable validation for malformed,
  overflowing, trailing, empty, and non-finite values.
- [x] Implement inference-specific tensor-shape parsing and deterministic
  named-input resolution without exposing backend or command-line framework
  types.
- [x] Migrate `run_ort_inference` and `benchmark_model` to the installed API,
  retaining TCLAP for option declaration, collection, and user help.
- [x] Review the complete touched C++ surface for behavior-preserving
  simplification, cohesive logical blocks, four-space Allman formatting, and
  required Doxygen contracts.
- [x] Expand the staged reader-facing pass with functional-unit Doxygen,
  contract-oriented block comments, and less densely packed calls and test
  fixtures.
- [x] Document the API boundary and consumer usage, including the distinction
  between CLI collection and reusable value parsing.
- [x] Run focused and complete target-owned tests, strict compilation, an
  explicit installed-consumer acceptance check, and staged-diff validation.
- [x] Stage only this reviewed parsing/API batch through an explicit allowlist,
  report exclusions and caveats, and stop for review before the plain
  centroiding stage.

Acceptance snapshot on 2026-08-29:

- the initial focused target failed because the planned installed headers did
  not exist, and the leading-plus and hexadecimal-float compatibility assertions
  each failed before their corresponding implementation was added;
- the final focused target passed 36 assertions across seven target-owned
  Catch2 cases, including malformed, overflowing, trailing, non-finite,
  ambiguous, duplicate, and invalid-metadata boundaries;
- a fresh `WARNINGS_ARE_ERRORS=ON` build completed, and the 45-test suite
  reported no failures: 43 passed and the two existing external FiLM fixture
  cases were intentionally skipped;
- an archive reconstructed from `HEAD` plus only the cached patch passed the
  strict build and all 43 registered tests, proving that this batch does not
  depend on the unrelated dirty headers or untracked FiLM test;
- Doxygen generated the new public API; its existing warning for the README's
  Markdown link to `doc/run_ort_inference.md` remains outside this batch; and
- an installed CMake consumer compiled and ran through
  `autoforge_deploy::autoforge_deploy` with the external ONNX Runtime loader path
  supplied, while installed headers and exports remained free of TCLAP.

## Stage 17: Plain Image-Only Centroiding Demo

Implement the plain CNN separately from FiLM integrations. The demo owns its
image semantics above the generic inference facade and accepts an ordinary PNG
as its sole user input.

- [x] Confirm or export the selected plain checkpoint as an external ONNX
  artifact with exactly one float32 image input and one two-value prediction
  output; record its checksum and complete preprocessing/output contract.
- [x] Add a repo-local `.ptafmodel` manifest for the external plain ONNX without
  embedding a machine-local artifact path.
- [x] Add a standalone C++20/OpenCV example that loads any readable PNG,
  converts it to grayscale, resizes it to the model input, scales it to
  `[0,1]`, and creates the NCHW tensor through shared adapter functionality.
- [x] Run inference through `CModelFacade`, interpret normalized `[x,y]`, and
  report normalized coordinates plus coordinates mapped to the original PNG.
- [x] Reuse generic tensor, geometry, runtime-selection, logging, and CLI parsing
  surfaces; keep plain-centroiding policy out of the ORT backend and generic
  MATLAB adapters.
- [x] Add a small tracked black PNG with a bright ellipse and an opt-in real-model
  end-to-end test that rejects non-finite/out-of-range output and verifies the
  predicted center against the known ellipse center with an empirically justified
  tolerance.
- [x] If the exported model does not respond meaningfully to the coherent
  synthetic ellipse, stop for design review rather than weakening the test into
  a non-functional smoke assertion.
- [x] Document build dependencies, artifact placement, manifest editing, CPU/CUDA
  invocation, expected output, resizing semantics, and fixture limitations.
- [x] Review, validate, and stage this example as a separate batch only after
  Stage 16 has left the index through explicit user action.
- [x] Expose generic feature rows to MATLAB as numeric `[x,y,score]` rows
  without adding plain-centroiding policy to the wrapper interface.
- [x] Add Python and MATLAB wrapper demos that own language-specific image
  conversion, use `CModelFacade`, and report the same coordinate spaces as the
  native CLI.
- [x] Extend the existing generated-wrapper smokes with generic feature-row
  coverage and run both wrapper demos end to end on the tracked ellipse.
- [x] Re-review and stage the enlarged native/wrapper example batch without
  absorbing unrelated source, export, MATLAB-program, or FiLM work.

Acceptance snapshot on 2026-08-30:

- the 50,805,325-byte plain checkpoint retained SHA-256
  `8022e92f4de8b86788bc523d5e45d5d09ab414161eac85def1d15c10fc39f12d`;
  its verified opset-11 ONNX export has SHA-256
  `8ba4f46355b0f6b542ec848b4c1130760bea194f9bf3b16e7b36bad9f380af4b`
  and contract `image [-1,1,1536,2048]` to `prediction [-1,2]`;
- the checkpoint followed three coherent synthetic ellipse positions with
  5.0--11.2 px error before export, and the tracked 320x240 PNG produced a
  1.90 px original-image error after the documented OpenCV resize path;
- the missing support header and missing CLI executable were each observed as
  the expected RED state before their corresponding implementation;
- a fresh warnings-as-errors standalone build passed all six native example
  tests with the external ONNX, including manifest CLI and tolerance-based
  ellipse inference; without the test-model environment variable the external
  Catch2 case was reported as skipped rather than failed;
- the current main suite passed 43 tests with its two existing external FiLM
  cases skipped;
- CPU and CUDA CLI runs both applied the requested provider and reported the
  ellipse center near `(217.62, 89.14)` for the labeled `(218, 91)` center; and
- a `HEAD` plus cached-patch reconstruction passed its strict producer build,
  all 43 main tests, strict standalone example build, and all six example tests
  using only the documented external ONNX prerequisite.

Wrapper extension snapshot on 2026-09-01:

- the missing generic MATLAB feature-matrix function and missing Python/MATLAB
  demos were each observed as RED before implementation;
- generated Python and MATLAB wrapper builds passed with the project-owned
  feature-row smokes; global `WARNINGS_ARE_ERRORS=ON` remains unsuitable for
  generated gtwrap/pybind/MATLAB sources because it promotes their existing
  third-party warnings, so the strict native build remains the owned-source
  warning gate;
- the generated Python stub target completed; strict checking of the authored
  demo passed after excluding three pre-existing diagnostics emitted by the
  generated gtwrap stub;
- Python CPU and CUDA demos applied the requested providers and reported
  `(217.6168, 89.1426)` and `(217.6199, 89.1446)`, respectively;
- MATLAB CPU and CUDA inference reported `(217.6175, 89.1478)`, preserved the
  5 px ellipse acceptance limit, returned the concrete submitted input shape,
  and applied the requested providers;
  and
- Python 3.12 compilation and MATLAB R2024b `checkcode` completed without
  source diagnostics.

## Stage 18: Centroiding Sequences And Native JSON Output

Approved on 2026-09-11. Consolidate overlapping existing work before extending
native, Python, and MATLAB image-only demos. Native C++ writes JSON directly
through embedded RapidJSON; overlays are disabled by default. Review locally
without subagents. Do not commit or push without separate authorization.

Execution gates confirmed on 2026-09-12:

- [x] Finish consolidation and stop before starting Stage 18.1 demo changes.
- [ ] After later implementation in Stages 18.1–18.5, stop before running
  Stage 18.6 benchmarks or validation. Consolidation checks remain part of
  pre-stage review.
- [ ] Resume each gated phase only after the user authorizes that phase.

### Pre-stage: Consolidate existing changes

- [x] Inspect the complete worktree and index; the starting index is empty at
  `3a8ca14` on `feature/ort_manager_pipeline_wrapped`.
- [x] Classify existing changed paths and preserve unrelated content, as recorded
  in the review inventory below.
- [x] Reconcile README and CLAUDE guidance with current source, workflows, and
  authoritative AGENTS.md; record this approved plan and cross-link it from TODO.
- [x] Reconcile the separate export tracker against its owning repository;
  qualify historical evidence and leave current owner acceptance pending.
- [x] Assess auxiliary interfaces against current consumers and approved next
  plans; prepare retirement of unused interfaces for review, as recorded below.
- [x] Review the wrapper gitlink change and validate affected wrapper behavior;
  record the tests and existing-binary limitation below. Preserve the checkout.
- [x] Assess the legacy export-script changes against the approved Python
  consolidation plan; preserve them unchanged for later owner-led replacement.
- [x] Classify FiLM, mobile-design, and workspace changes separately; finishing
  these projects is not a prerequisite for image-only centroiding.
- [x] Validate and stage one coherent consolidation batch at a time using an
  explicit path/hunk allowlist; report evidence and the proposed commit message.
- [x] Follow the staged review gates for consolidation batches. On 2026-09-12,
  the user exempted documentation-only stages from blocking review pauses;
  commit authorization remains separate.
- [x] Finish the pre-stage with overlapping work settled, the index clear, and
  a list of deferred changes to preserve. A clean tree is not required.

Review inventory on 2026-09-11. These are static findings; runtime validation remains open:

| Paths | Disposition and evidence |
|---|---|
| `README.md`, `CLAUDE.md` | First documentation batch: retain useful existing updates; correct obsolete template/tracker references, CI caching/publication claims, CUDA policy, and distinguish historical validation from current evidence |
| `src/auxiliary/common_defs.h`, `common_ops.h`, `images_prepro.h` | Separate functional review: name ownership/copying, nested shape accounting, release-mode validation, serial accumulation, and include cleanup; `noexcept` moves call allocating `refreshNameViews()`, requiring correction or justification |
| `lib/wrap` | Clean dependency checkout advanced from `bf9f786` to `4b34b59`; changes affect MATLAB `ImportWrapBuildDir` and its tests; wrapper validation remains open |
| `python/scripts/tmp_to_rework_as_generic/ExportPytorchToONNX.py` | Deferred export-owner review: path migration to `models/pytorch` and `models/onnx` in a protected legacy exporter |
| `doc/development/model_export_and_python_consolidation_plan.md` | Preserve separately: Stage 1 checkbox updates and dated cross-repository export/test evidence have not been revalidated here |
| `examples/model_configs/ml_based_centroiding_film.ptafmodel`, `tests/inference/testRealCentroidingOnnx.cpp` | Deferred FiLM work: image plus prior input and two outputs; test uses fixed temporary manifest names and a broad CUDA exception-to-skip path needing later review |
| `doc/development/mobile_deployment_pipeline_design.md` | Deferred design-only Android/mobile work; no implementation or qualification accepted by this batch |
| `torchAutoForge-deploy.code-workspace` | Preserve separate local workspace additions; no product dependency change |

### Stage 18.1: Record the plan and rename the integration

- [x] Record the approved checklist here and cross-link it from TODO.
- [x] Rename `examples/plain_centroiding/` to `examples/centroiding_models/`.
- [x] Rename `run_plain_centroiding.cpp/.py` to `run_centroiding.cpp/.py`,
  `RunPlainCentroidingFacadeDemo.m` to `RunCentroidingFacadeDemo.m`, and
  `plain_centroiding_support.h` to `centroiding_support.h`.
- [x] Update namespaces, CMake targets, tests, identifiers, and active documentation.
  Retain `plain` for architecture/checkpoint identifiers and historical evidence.
- [x] Preserve manifest/raw-ONNX loading, runtime options, preprocessing, generic
  adapters, and CModelFacade ownership; verify the renamed single-image invocation.

Contract: one image tensor input, independent batch-one execution, and one `[1,2]`
output. FiLM adapters, prior vectors, tracking, and public inference-facade changes
are excluded. Pause for design review if this boundary must change.

### Stage 18.2: Deterministic sequence execution

- [x] Accept positional `input_path` as one image or a directory. Preserve
  decoder-supported explicit single-file inputs.
- [x] Enumerate directories once before creating output, non-recursively, selecting
  regular files with case-insensitive `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, and
  `.tiff` extensions; reject empty selected sequences.
- [x] Match natural filename ordering across languages: compare ASCII digit runs
  numerically without integer overflow, other text case-sensitively by Unicode
  code point, and equivalent natural keys by the complete original filename lexically.
- [x] Load/validate the model once and process each frame independently at batch
  size one; fail clearly on an unreadable selected image.
- [x] Preserve existing grayscale conversion, direct bilinear resize, and unit
  scaling. Retain only current-frame image/tensor buffers, never image arrays.
- [x] Preserve console prediction fields and add frame identity for sequences.
- [x] Time only InferSingleFloatTensor with monotonic elapsed time, including its
  existing facade/wrapper overhead but excluding loading, preprocessing, and output IO.

### Stage 18.3: Embedded RapidJSON and native reports

- [x] Embed the RapidJSON header distribution and license under
  `lib/header_only/rapidjson`, with donor paths and provenance recorded alongside it.
- [x] Reconfirm the inspected raytracer headers match local RapidJSON revision
  `24b5e7a8b27f42fa16b96fc70aade9106cf7102f`; record raytracer import commit
  `b50e068e81d32e7f79bb89089305200b2ec892d1`. Use the matching checkout's license.
- [x] Keep RapidJSON private to the native application and its tests; import no donor
  build machinery or JSON-library conformance tests.
- [x] Add `--output PATH`, writing `PATH/predictions.json` directly from C++ without
  Python or MATLAB and without overlays by default.
- [x] Use schema_version 1 with status, model, input, preprocessing, and ordered
  frames fields, plus error details for failed runs.
- [x] Record model/config paths, role, effective runtime configuration, backend
  detail, and tensor metadata from GetContract(). Record each demo's preprocessing
  and image-library identity separately from declarative manifest pipeline labels.
- [x] Record supplied input path, input kind, selected frame count, and ordering.
- [x] Record each frame's zero-based index, relative source path, original
  dimensions, raw output name/shape/values, normalized coordinates, coordinates
  in model and image pixels, inside_image, inference_ms, and relative overlay path or null.
- [x] Stream native frame records to disk instead of accumulating report records
  in memory; the filename inventory may remain in memory for sorting.

### Stage 18.4: Opt-in overlays and failure handling

- [x] Add `--overlays`, requiring `--output`.
- [x] Accept a new output directory or an existing empty directory; reject
  nonempty destinations and output paths equal to or containing the input location.
  Permit an output child directory because enumeration is fixed and non-recursive.
- [x] Save original-resolution PNG overlays as `overlays/000000_<source-stem>.png`.
  Preserve source appearance outside contrasting crosshair strokes; extra overlay
  decoding must not change inference preprocessing.
- [x] Draw at the true predicted coordinate and clip strokes naturally; fully
  off-image crosshairs remain invisible. Never clamp recorded predictions.
- [x] Accept finite out-of-range coordinates while retaining invalid-shape and
  non-finite-value rejection. Document upper-left origin, x rightward, y downward,
  coordinates multiplied by width/height, and inside bounds `0 <= x < width`,
  `0 <= y < height`; normalized 1 is outside the corresponding upper boundary.
- [x] Create an initial valid incomplete report. Spool frame records and atomically
  publish an assembled report on success or caught failure.
- [x] Stop at the first decoding, inference, validation, or output error; record
  its stage, frame index, source, and actionable message. Preserve completed-frame
  records and mark complete only after all frames and requested overlays succeed.
- [x] On report-publication failure, retain the prior incomplete report and spool,
  identify their locations, and return failure. Abrupt-termination recovery is excluded.

### Stage 18.5: Matching Python and MATLAB behavior

- [x] Match input selection, sorting, model reuse, JSON-only default, opt-in
  overlays, and failure semantics using native Python/MATLAB JSON facilities.
- [x] Preserve language-specific preprocessing conventions and document expected
  small numerical differences.
- [x] Add MATLAB options `strOutputPath=""` and `bOverlays=false`; return one run
  struct matching the report schema for both input modes, including frames.
  Retain compact metadata for that return value, never image arrays.
- [x] Verify matching singleton/empty arrays, numeric arrays, booleans, and null overlays.

### Stage 18.6: Validation and documentation

- [ ] Test single-image compatibility, natural ordering/ties, leading zeros, mixed
  case, long digit runs, non-recursion, empty input, and unreadable images.
- [ ] Verify one model load and independent batch-one inference; test exact
  coordinate mapping using controlled outputs, including boundary/off-image values.
- [ ] Test JSON-only defaults, overlays, original dimensions, marker placement,
  unchanged pixels outside marker strokes, collisions, and output beneath input.
- [ ] Test failure after successful frames, retained partial results, report fields,
  ordering, and completion status across languages.
- [ ] Use synthetic ellipse/blob images to test image preparation and coordinate
  mapping. Replace the learned-model geometric-center assertion with
  contract/output checks; do not
  require arbitrary geometric centers as learned-model ground truth.
- [ ] Confirm user authorization to start validation after the implementation pause.
- [ ] Use COSMICA Itokawa output images as the primary real-image sequence.
  Inspect `cosmica-simulator/output_images/images` and `output_images_ID0` to
  establish the intended run and record source provenance before selecting frames.
- [ ] Select supplementary evaluation cases from datasets referenced by
  ml-based-centroiding and OPERATIVE datasets under `$DATASETS`. The current root
  is `/media/peterc/DatasetsArchive/datasets`; discovered candidates include
  `UniformlyScatteredPointCloudsDatasets/Itokawa`,
  `UniformlyScatteredSequencesDatasets/Itokawa`, and
  `TrajectoriesDatasets/Moon/OPERATIVE_trajectory_test`.
- [ ] Record source paths, selection rules, dimensions, image depth, model artifact,
  runtime configuration, and label provenance. Leave source datasets unchanged.
- [ ] Add the additional folder when the user provides it; proceed with available
  datasets without inventing its location or treating it as a prerequisite.
- [ ] Report per-frame durations and aggregate latency statistics, identifying
  first-frame effects. Compute coordinate-error metrics only after verifying
  label meaning, units, and origin; otherwise report contract and visual checks.
- [ ] Run a fresh native standalone build, focused tests, Python/MATLAB demo tests,
  and available real-ONNX sequence smokes; report unavailable checks explicitly.
- [ ] Document runnable native/Python/MATLAB examples and expected output,
  dependencies, ordering, coordinates, collision policy, and failure guarantees.

Implemented native/Python argument forms:

```bash
run_centroiding model.ptafmodel image.png
run_centroiding model.ptafmodel frames/ --output results/
run_centroiding model.ptafmodel frames/ --output results/ --overlays
```

### Stage 18.7: Review and stage the extension

- [ ] Review correctness, repeated loading, unnecessary abstractions,
  excessive copies, comments, formatting, and public documentation.
- [ ] Update these checkboxes with validation results and limitations.
- [ ] Recheck worktree/index and the review gate; stage one coherent extension
  batch using reviewed paths/hunks while preserving deferred pre-stage work.
- [ ] Inspect the complete cached diff and run `git diff --cached --check`.
- [ ] Report staged paths, evidence, exclusions, limitations, and proposed message;
  stop without committing, pushing, or preparing another batch.

Proposed extension subject: `Extend centroiding demos with sequences and native JSON output`.

Pre-stage documentation review, 2026-09-11:

- [x] Review the full candidate documentation diff and verify local Markdown
  links/anchors, code-fence balance, and `git diff --check`.
- [x] Check current workflow definitions: ORT-package caches remain, CMake-tree
  caches are absent, and the Pages deployment job is disabled. Correct README
  accordingly without changing workflow behavior or claiming current CI success.
- [x] Verify SHA-256 preservation of all nine excluded dirty/untracked files and
  preserve the clean wrapper checkout at `4b34b59`.
- [x] Revise this batch using the scientific-writing and no-ai-slop skills;
  consolidate repeated README capability summaries and retain technical constraints.
- [x] Receive user authorization to commit README, CLAUDE, TODO, and this tracker
  with the subject below and proceed to the next consolidation batch. Runtime
  tests are not rerun for this documentation-only batch.

Approved first-batch subject: `Revise documents and record centroiding stages`.

Pre-stage wrapper review, 2026-09-11:

- [x] Commit the preceding documentation batch as `280c42f` with the approved
  subject, `Revise documents and record centroiding stages`, and verify that
  the index is clear before preparing this batch.
- [x] Review the single dependency commit from `bf9f78617830bfa88c70d81a1a791c0a0c90b548`
  to `4b34b593247f9e365f7685cd430d37a7cc921839`. It changes only
  `matlab/ImportWrapBuildDir.m` and `matlab/tests/testImportWrapBuildDir.m`.
  The helper prefers an explicitly named namespace directory, retains repository
  and generated-wrapper discovery fallbacks, and returns only paths it added.
- [x] Run all three donor MATLAB function tests on R2024b: namespace precedence,
  renamed wrapper discovery, and repository-folder fallback passed.
- [x] Check a temporary MEX-only directory with an empty namespace list. The
  returned value contains exactly the MEX directory, with no empty path entry.
  The helper emits the expected warning for the absent optional wrapper folder.
- [x] Call ImportWrapBuildDir on `build-stage17-wrappers` with
  `ptafdeploy.inference`. Verify the returned wrapper/MEX paths and resolution of
  CModelFacade, then run the project's TestInferenceFacadeSmoke with the tracked
  ONNX, centroiding, and object-detection fixtures. All assertions passed.
- [x] Confirm the dependency checkout is clean, the old revision is an ancestor
  of the new one, and the donor diff passes `git diff --check`. No dependency
  source was edited, fetched, initialized, or committed during this review.
- [x] Receive user authorization and commit the wrapper batch as `2abdb68`;
  verify the index is clear before preparing the auxiliary-header batch.

The target smoke uses existing generated MATLAB/MEX binaries in
`build-stage17-wrappers`, with its `src` directory on LD_LIBRARY_PATH. It validates
this MATLAB discovery change against those binaries; no fresh native build,
wrapper generation, Python test run, or current-source runtime qualification is
claimed. Existing auxiliary-header, export, FiLM, mobile, and workspace changes
remain outside this batch.

Proposed wrapper subject: `Update wrapper revision for MATLAB directory discovery`.


### Pre-stage auxiliary consolidation, 2026-09-12

The earlier ownership-repair candidate was not approved. Its tests established
that the pending allocating noexcept moves could terminate, but did not establish
a consumer requirement for maintaining these interfaces. That candidate and the
complete starting index/worktree patches are preserved in
`/tmp/ptaf-consolidation-vs8f7j6k`; its historical validation remains under
`/tmp/ptaf-legacy-review-qi1pg9ep`.

- [x] Search current sources and known downstream source trees in ML-repos,
  SLAM-repos, projects-DART, rendering-sw, nav-backend, nav-frontend,
  nav-frontend-cpp, and nav-system. Exclude vendored, generated, installed,
  archived, and environment directories. No consumers of the legacy specs or
  AccumProduct were found outside their definitions and the candidate tests.
- [x] Check the approved centroiding and export plans. Neither requires these
  legacy types. Current tensor descriptors, views, buffers, and ComputeElementCount
  provide the inference contracts; generic task adapters provide preprocessing.
- [x] Keep CheckFileExists and CheckFileExistsWithExt, including their behavior
  and namespace. ORT model loading calls the extension check. Retain direct
  standard includes and public documentation in their owning header.
- [x] Remove the unused SInputOutputSpecs and SImagesInputOutputSpecs headers,
  the empty image-preprocessing placeholder, and AccumProduct from the review
  candidate. Withdraw the newly added ownership tests with their retired subject.
- [x] Configure and build a fresh Debug tree with WARNINGS_ARE_ERRORS=ON,
  examples and wrapper generation disabled. CTest reports 44 passed, two external
  FiLM tests skipped, and zero failures. The unchanged untracked FiLM test file
  is discovered by CMake but is excluded from the candidate.
- [x] Install into a fresh temporary prefix: only common_ops.h remains under the
  auxiliary include directory. Compile a C++20 installed-header consumer with
  -Wall -Wextra -Werror and verify existing-file, missing-file, directory, and
  extension-mismatch behavior, including strict failures.
  Evidence: `/tmp/ptaf-consolidation-vs8f7j6k` contains configure/build/install/CTest
  logs and the installed-header probe. GPU execution and regenerated wrappers
  were not tested.
- [x] Review the full candidate and stage only the auxiliary retirement, this
  tracker, and the corresponding TODO correction. Verify unrelated file contents
  are unchanged.
- [x] Verify the reviewed auxiliary batch was committed as `180a5c3` before advancing.

Compatibility: this candidate removes previously installed auxiliary headers and
AccumProduct. Uninspected external consumers may require migration to the current
inference types or a separate product-specific shape calculation. ComputeElementCount
validates one tensor shape; it is not a drop-in replacement for multiplication
across the old nested shapes. Existing installation prefixes may retain obsolete
headers; validate delivery in a fresh prefix. Public inference facades and wrappers
are unchanged. The limited local search cannot establish absence of all external users.

Export-script/tracker, FiLM, mobile-design, and workspace changes remain deferred
and unchanged. The demo rework has not started.

Proposed auxiliary subject: `Retire unused legacy tensor and preprocessing helpers`.

### Pre-stage export reconciliation, 2026-09-12

- [x] Verify that the auxiliary retirement batch was committed as `180a5c3`
  (Remove unused legacy tensor and preprocessing helpers) and that the index
  is clear before preparing this batch.
- [x] Reconcile the export tracker with the staged, uncommitted PTAF owner worktree.
  Preserve the dated implementation report, distinguish historical validation from
  current source inspection, and leave overall Stage 1 acceptance pending.
- [x] Defer the protected legacy export script unchanged. Its path migration follows
  the models layout, but its hard-coded checkpoint is absent locally and its
  output-path assumption conflicts with current PTAF behavior. Replacement and
  relocation remain in the owning export plan; no exporter repair is included.
- [x] Review and stage only the two development trackers; verify the owner index
  and all excluded deployment files are unchanged. Diff checks pass.
- [x] Verify the documentation batch was committed as `57175ee` (Update development
  plans), with the index clear on 2026-09-12.

The remaining export-script, FiLM, mobile-design, and workspace changes stay
outside this batch. All overlapping pre-stage changes have now been assessed;
consolidation is complete at `57175ee`, with the index clear. This closing
tracker update is left unstaged. Stop before starting any centroiding demo rework. The later pause
before benchmarks and validation remains in force.

Proposed subject: `Reconcile export status and defer legacy script migration`.

### Consolidation follow-up: legacy exporter retirement

- [x] Supersede the earlier export-script deferral after verifying committed PTAF
  numerical validation and receiving user confirmation that .mat export is unused.
- [x] Remove ExportPytorchToONNX.py and its pending path edits, with a recovery copy
  under `/tmp/ptaf-retire-export-vceqr0sk`. Update its active documentation reference.
- [ ] Complete the staged source-removal review before further source changes.

Workspace additions and the untracked FiLM test remain separate. The shared
workspace's ml-based-centroiding entry is relevant to the next plan; the two
machine-local worktree entries belong in a personal workspace. Their contents are
preserved pending that separate cleanup. Centroiding demo rework has not started.

### Centroiding implementation handoff, 2026-09-12

The user authorized continuing beyond the workspace-only consolidation pause.
The records below describe implementation before staged review. No build, test,
benchmark, or runtime validation has been run for this candidate; implementation checkboxes
record source changes only. Stage 18.6 remains gated by user authorization.

- [x] Rename the integration, native/Python executables, MATLAB entry point, support
  header, namespace, and native test target. Preserve actual plain-model identifiers.
- [x] Implement deterministic non-recursive sequence selection, one model load,
  independent batch-one frames, and inference-only timing in the three demos.
- [x] Embed RapidJSON headers and matching license, recording donor provenance.
  Add native JSON reporting without any Python/MATLAB runtime dependency.
- [x] Implement versioned reports, optional overlays, collision protection, finite
  out-of-image coordinates, and incomplete-report publication with retained records.
- [x] Add Python/MATLAB reports using their native JSON facilities. Preserve existing
  facade and preprocessing ownership. The wrappers do not expose runtime target
  priority: their report field is null; native reports include the effective vector.
- [x] Write native/Python/MATLAB test sources for selection, coordinate mapping,
  report/overlay behavior, and failures. Remove the learned-model assertion that
  required a synthetic ellipse's geometric centre.
- [x] Document commands, schema, coordinate and failure conventions, and the
  Itokawa, ml-based-centroiding, OPERATIVE, and forthcoming-folder validation inputs.
- [x] Stop after implementation and before the requested validation/benchmark phase.
- [ ] After authorization, configure/build, run the focused tests, review cross-language
  report behavior and filesystem error paths, then run the selected real-image cases.
- [ ] Review the resulting complete diff and stage a tested coherent extension batch.
  No commit or push is authorized by this handoff.

The existing workspace change and untracked FiLM regression remain outside the
extension. PNG overlays support uint8/uint16 image samples; unsupported sample
types fail explicitly. The native and Python report spools bound record memory;
MATLAB retains compact frame structs for its return value as planned.

### Centroiding source review and staged candidate, 2026-09-12

- [x] Review the complete candidate locally against the agreed image-only contract,
  sequence policy, report schema, memory ownership, failure handling, and source
  documentation. Preserve the workspace edit and untracked FiLM test outside it.
- [x] Correct incomplete-report publication to read only successfully closed frame
  records; failed append tails cannot enter native/Python reports. Add regression
  test sources for this case. Consolidate MATLAB JSON write/close checks.
- [x] Correct final-publication error attribution, record the requested model path
  before loading, preserve marker opacity in MATLAB alpha images, and retain
  Pillow palette transparency. Match native JPEG overlay orientation to inference
  decoding and reject mismatched overlay extents.
- [x] Remove redundant test mocks and metadata copies. Complete public helper
  documentation and input/output arguments blocks; format the first-party sources.
- [x] Run Python static syntax/name/import checks (Ruff E9/F): no findings. Compare
  all 38 RapidJSON headers and the license against the donor: unchanged. Confirm
  the renamed PNG fixture retains its original bytes. These checks are not runtime
  qualification of the demos.
- [x] Inspect and prepare the exact staged extension index against the explicit path
  allowlist. Verify staged files match the reviewed worktree and excluded files retain
  their hashes. First-party whitespace checks pass. The full cached check reports
  263 inherited whitespace findings in unmodified RapidJSON headers; preserve the
  verified donor distribution without reformatting it.
- [ ] Run builds, native/Python/MATLAB tests, real-model validation, and benchmarks
  only after the separate validation pause is released. Current test sources have
  not been executed. No commit or push is authorized in this review.

Remaining qualification limits: the existing wrapper cannot expose runtime target
priority, so that field is null in wrapper reports; native reports include it.
Cross-language image decoding and file-publication behavior still require the
planned validation. The staged candidate is for source review, not a claim that
Stage 18.6 has passed.

Proposed subject: `Extend centroiding demos with sequences and native JSON output`.

### Utility extraction and centroiding migration, 2026-09-12

This approved plan supersedes the demo-owned `_io`/`_support` design above.
Breaking auxiliary and parsing include paths and namespaces are authorized.
The previous centroiding candidate was committed as `8304ce1`. The subsequent
`next` instruction advanced this refactor to validation and staged review.

- [x] Stage 0: inspect and preserve the index, worktree, and excluded files.
- [x] Stage 1: move filesystem helpers into utils and value parsing into
  utils/parsing; update namespaces, consumers, installation, and documentation.
- [x] Stage 2: add independent optional images and inference_output compiled
  targets, component exports, and dependency discovery; keep core consumers light.
- [x] Stage 3: extract image operations and sequence selection, typed JSON values,
  raw tensor serialization, and bounded complete/incomplete report publication.
- [x] Stage 4: replace demo helper headers with a compiled centroiding adapter,
  migrate applications and test sources, and document reusable native examples.
- [x] Review implementation locally, then pause before builds/tests/benchmarks.
- [x] Stage 5, after authorization: execute focused tests, installed component
  acceptance checks, and real-image validation with COSMICA Itokawa,
  ml-based-centroiding, OPERATIVE, and the forthcoming input folder.
- [x] Stage 6: review and stage one explicit coherent allowlist; report evidence,
  limitations, exclusions, and a proposed commit message. Do not commit or push.

Implementation evidence: filesystem helpers now reside in `utils/filesystem.h`;
value parsing resides in `utils/parsing` with namespace
`ptafdeploy::utils::parsing`. Optional `images` and `inference_output` components
have independent targets and exports. Core-only package discovery retains the
existing dependency list and does not discover OpenCV. The image component replays
its OpenCV dependency only when requested. RapidJSON appears only in the output
implementation and relevant test code, not in installed public headers.

The compiled centroiding adapter preserves preprocessing and coordinate mapping.
Native frame/run records use typed JSON values; Python helpers are separated into
image selection/annotation, output publication, and centroiding metadata. MATLAB
retains local functions with explicit responsibility boundaries. Public utility
examples and breaking include/namespace migrations are documented in
`doc/image_and_inference_output.md`.

Source review corrected hidden global filesystem-alias dependencies, a Python
module/path naming collision, clipping of thick markers at image borders, and
report append lifecycle handling. New test sources cover filesystem behavior,
image operations, tensor cardinality/type/finiteness, invalid UTF-8, reserved
fields, partial tails, and append failure. Existing centroiding tests use the
compiled components. These tests have not been run.

Checks performed: Ruff E9/F for the Python application, helper modules, and test
source passed. Tracked-diff and new-authored-file whitespace checks passed. All
38 RapidJSON headers and the license remain byte-identical to the donor. The PNG
fixture, workspace edit, and unrelated FiLM test retain their prior bytes.

At the initial implementation handoff, execution checks were paused and the prior
index was unchanged. The validation results below supersede that qualification
status. No commit or push of this refactor has been performed.

### Utility validation and consolidation, 2026-09-12

- [x] Configure and build fresh Debug shared libraries with both optional components.
- [x] Run core/utility CTest: 48 passed; two optional checks from the unrelated
  untracked FiLM test skipped. That source remains excluded from consolidation.
- [x] Install into a fresh prefix; build and run the native centroiding suite:
  10 passed, including the available image-only ONNX checkpoint.
- [x] Run Python sequence tests: 4 passed. Run MATLAB R2024b sequence tests:
  2 passed. Reuse existing generated wrappers with the newly built core runtime;
  no wrapper API change or wrapper regeneration was required.
- [x] Build Release static libraries and compile/run core-only, output-only, and
  combined installed consumers against both shared and static installations.
  Core/output-only consumers disable OpenCV discovery. Check missing required,
  missing optional, and disabled image components. Configure core with OpenCV
  discovery disabled. Build and run the documented native image/output example.
- [x] Process six real images through each of C++, Python, and MATLAB. Verify ordered
  frame identities, raw tensor shapes/finite values, coordinate mappings, inside
  flags, overlay extents, and unchanged pixels outside the crosshair region.
- [x] Verify partial-run failures retain one completed record and an incomplete
  report in all languages. Verify native/Python empty-input rejection and output
  collision preservation. JSON-only runs create no overlay directory.
- [ ] Validate the additional user-supplied input folder when it becomes available;
  larger accuracy/performance campaigns remain separate from this bounded smoke.

Execution exposed two remaining consumers of the retired global filesystem alias;
ORT and model-facade tests now declare their own alias. The natural-order test now
accounts for case-sensitive extension comparison before the spelling tie-breaker.
MATLAB test entry points omit arguments blocks because `functiontests` explicitly
rejects them; application functions retain argument validation.

The six-frame native-versus-Python/MATLAB maximum coordinate difference was
0.699459 pixels. These paths retain different resizing implementations; the smoke
checks schema and mapping consistency rather than requiring identical predictions
or measuring geometric-centre accuracy. Timing samples are recorded in each JSON
report; overlapping validation processes preclude comparative performance claims.
The wrapper execution-target-priority field remains null. Installed shared runtime
consumers require ONNX Runtime on the loader path, as specified by the existing
external dependency contract.

Evidence directory: `/tmp/ptaf-utils-validation-qnfz4s88`. It contains configure,
build, CTest, wrapper, consumer, and failure logs; runnable acceptance scripts;
`input-provenance.json`; and all three real-image reports and overlays.

Checkpoint: `models/onnx/best_model_plain_traveling-goat-68_22b61bbd4ddd.onnx`; SHA-256
`8ba4f46355b0f6b542ec848b4c1130760bea194f9bf3b16e7b36bad9f380af4b`.

Selected source files (symlinked into the temporary input directory):

- `/home/peterc/devDir/projects-DART/cosmica-simulator/output_images/images/000000.png`
  SHA-256 `530a9d019af8d1eefab7b37c50d9864f35e0f762cc6ed90f4fbd5d7e17647d69`
- `/home/peterc/devDir/projects-DART/cosmica-simulator/output_images/images/003952.png`
  SHA-256 `2156745bd2c4c5acbcb590e415f698c28003c1cf2a1451429b935a918ff3cd27`
- `/media/peterc/DatasetsArchive/datasets/UniformlyScatteredPointCloudsDatasets/Itokawa/Dataset_UniformPointCloud_Itokawa_SPECTRAL_OPTIX_RT_1000_WFOV_Farinella_evaluation_ID99/images/000000.png`
  SHA-256 `2ddb0fa63a0f6bde7c182c9548328f6ff4ff5a6cb90b10506adc2512df9993e6`
- `/media/peterc/DatasetsArchive/datasets/UniformlyScatteredPointCloudsDatasets/Itokawa/Dataset_UniformPointCloud_Itokawa_SPECTRAL_OPTIX_RT_1000_WFOV_Farinella_evaluation_ID99/images/000500.png`
  SHA-256 `839b399f1ce27fe2a3db1ed77bd7827172f6cf36059b70ac7481d13fa6977fa7`
- `/media/peterc/DatasetsArchive/datasets/TrajectoriesDatasets/Moon/OPERATIVE_trajectory_test/images/000001.png`
  SHA-256 `2ce224179f42b1ab10c69c6240fe81d9dfd0e89e2f08a3d3fbd77b63c5176656`
- `/media/peterc/DatasetsArchive/datasets/TrajectoriesDatasets/Moon/OPERATIVE_trajectory_test/images/013554.png`
  SHA-256 `e78f9a99b6587e4ad0e9a6e23d950585a69828554bc25d0600f4a031416534e6`

Consolidation result: the reviewed refactor is staged through an explicit 55-path
allowlist (52 diff entries after rename detection). Cached whitespace checks pass;
staged source bytes match the tested worktree. The workspace edit and untracked
FiLM test remain unchanged and excluded. No commit or push was performed.

Proposed commit message:

```text
Consolidate utilities and extract image and inference output

- Move filesystem and parsing helpers into responsibility-specific utils paths
  and update consumers without compatibility aliases

- Add optional image and inference-output components with independent exports
  and private JSON implementation dependencies

- Route centroiding through a compiled adapter while preserving its model
  contract and report schema

- Validate shared/static consumers, native utilities, wrapper sequences, and
  bounded real-image runs; document compatibility changes and remaining limits
```

### Remaining validation and native evaluation, 2026-09-12

Current baseline: commits `8304ce1` and `b6dfeeb` contain the sequence demos and
optional utilities. Earlier implementation-pause and staged-only statements above
are historical. The user has authorized the remaining implementation and validation.
The next review stages only housekeeping; native changes remain unstaged until
the subsequent review batches.

Implement and review the entire change set before staging only batch 1. No Python
evaluation utilities or package changes belong in this repository; Python work,
if requested later, belongs in pyTorchAutoForge. Source datasets remain immutable.

- [x] Stage 1: reconcile recorded status, keep the VS Code workspace locally but
  remove it from tracking, and reserve gitignored data/inputs and data/results
  for linked/copied inputs.
- [x] Stage 2: rename and validate the FiLM test, remove unsupported synthetic
  accuracy assumptions, and distinguish missing CUDA capability from inference errors.
- [x] Stage 3: close selection, contract, JSON, failure, image-format, and delivery
  coverage gaps; reuse passing evidence and rerun checks affected by repairs.
- [x] Stage 4: implement reusable native statistics/point metrics, versioned report
  reading, input preparation, evaluation CLI, shell orchestration, and focused tests.
- [x] Stage 5: select up to 100 frames per available dataset, run three sequential
  CPU passes, exclude the first five frames from steady summaries, and inspect
  ten overlays per dataset. Compute accuracy only for verified reference semantics.
- [x] Review the full implementation and evidence locally without subagents.
- [x] Stage batch 1 only: tracker/workspace reconciliation and /data/ ignore rule.
- [ ] On subsequent next, with index clear, stage batch 2: FiLM test and corrections.
- [ ] On subsequent next, stage batch 3: functional/delivery validation corrections.
- [ ] On subsequent next, stage batch 4: native evaluation tools and evidence.
- [ ] Add the forthcoming user input folder when provided; it does not block the
  available datasets. Larger accuracy and GPU comparison campaigns remain separate.

Implementation and review evidence:

- [x] Implement generic finite statistics and source-matched point errors in core
  `utils/metrics`; add strict report reading to the optional native output component.
- [x] Add native selection/evaluation programs and a sequential shell run driver.
  Do not add Python evaluation modules, packaging changes, or facade changes.
- [x] Rename the external FiLM test and remove geometric-centre accuracy assertions.
  Reject an explicitly invalid artifact path; skip unavailable external prerequisites.
- [x] Build Release shared and static libraries; run installed core/output/combined
  consumers for both. Build output-only programs with OpenCV discovery disabled.
- [x] Pass 54 core/utility tests with two FiLM prerequisites skipped, and all ten
  standalone centroiding tests with the real plain model enabled. Pass known-answer
  evaluation CLI checks and shell syntax/ShellCheck checks. Verify 8/16-bit PNG
  grayscale, RGB, and RGBA round trips and report-publication recovery.
- [x] Reuse the recorded native/Python/MATLAB sequence and failure evidence for the
  unchanged demo paths; do not present those earlier runs as new benchmarks.
- [x] Run three sequential 100-frame CPU passes per dataset and a separate ten-frame
  overlay pass per dataset. Verify all thirty overlays and coordinate mappings.
  Record selections and hashes under ignored data/inputs and data/results.
- [x] Review new sources and the complete candidate diff for ownership, source
  documentation, collision behavior, finite values, repeated loading, and scope.

Evidence: `/tmp/ptaf-evaluation-9mtake_3`; the unstaged native evaluation document
records commands, sources, per-pass statistics, and reference-label limitations.
Other MATLAB workloads were active. The measured latencies are not isolated
performance benchmarks. COSMICA target identity follows the user-selected folder;
its exact run configuration remains unverified. Itokawa and OPERATIVE coordinate
labels were inspected but not adopted as ground truth. No learned accuracy claim
is made. The compatible FiLM ONNX model, CUDA provider, and forthcoming input folder
remain unavailable.

The full implementation remains in the worktree. Batch 1 contains only `.gitignore`,
`torchAutoForge-deploy.code-workspace`, and this tracker. Batch 2 owns the FiLM test;
batch 3 owns the demo validation-status correction; batch 4 owns native evaluation,
its dependent tests, utility coverage, and documentation. This allocation keeps
reader-dependent tests with the reader implementation rather than staging them early.

Proposed batch 1 message:

```text
Reconcile centroiding stages and reserve local evaluation data

- Update completed stages and record native evaluation evidence and limitations

- Keep the workspace file local and ignore evaluation inputs and results
```
