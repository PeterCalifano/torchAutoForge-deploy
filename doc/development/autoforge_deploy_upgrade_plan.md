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
- [x] Remove or deprecate redundant legacy specs where covered by generic descriptors/views/buffers.
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
