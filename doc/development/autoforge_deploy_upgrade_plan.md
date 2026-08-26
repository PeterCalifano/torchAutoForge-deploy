# torchAutoForge-deploy Upgrade Plan

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

## Stage 3: Wrapper Configuration

- [ ] Use `ptafdeploy` as the project namespace and `ptafdeploy::inference` as the nested namespace for inference APIs; do not keep the old `deploy_infer` namespace.
- [ ] Make `src/wrap_interface.i` top-level namespace and includes match `ptafdeploy`.
- [ ] Populate `src/inference/inference.i` around `ptafdeploy::inference::CInferenceManager` and wrapper-safe value types.
- [ ] Expose model loading, metadata query, and float host-buffer inference to Python/MATLAB.
- [ ] Avoid wrapping raw ORT handles, raw `void*`, and `std::byte` storage directly.
- [ ] Add Python import smoke test.
- [ ] Add MATLAB wrapper generation and model-load smoke test.
- [ ] Set MATLAB-safe runtime defaults, especially ORT thread count, while allowing explicit overrides.

## Stage 4: Implementation Review And Cleanup

- [ ] Review `ptafdeploy::inference` tensor types against legacy `SInputOutputSpecs` and `SImagesInputOutputSpecs`.
- [ ] Remove or deprecate redundant legacy specs where covered by generic descriptors/views/buffers.
- [ ] Audit ORT backend helper functions for readability, reuse, and error clarity.
- [ ] Remove useless one-off local helpers that only hide simple operations.
- [ ] Add tests for named input mapping, mixed named/unnamed rejection, dtype mismatch, shape mismatch, byte-count mismatch, null data, and unloaded-session error.
- [ ] Add numeric reference test for ORT output values.

## Stage 5: General Model Facade

- [ ] Add model-role layer above raw tensor inference, not inside ORT backend.
- [ ] Define generic model contract for artifact path, role, input descriptors, output descriptors, preprocessing, and postprocessing.
- [ ] Support initial roles: raw tensor model, centroiding model, object detection model.
- [ ] Keep registry extensible for feature matching, tracking, optical flow, and future learned modules.
- [ ] Avoid hardcoded model names and repo-specific paths in core library code.
- [ ] Provide MATLAB-friendly facade that hides raw tensor plumbing but keeps metadata and errors visible.

## Stage 6: YOLO And Centroiding ONNX Support

- [ ] Locate real YOLO and centroiding ONNX artifacts from related repos or repo-local model folders.
- [ ] Add repo-local example configs instead of absolute model paths.
- [ ] Implement YOLO through generic object-detection role.
- [ ] Implement centroiding through generic centroiding role.
- [ ] Add C++ smoke tests for role config parsing and facade dispatch.
- [ ] Add MATLAB smoke tests for loading both roles and running one fixture inference.

## Verification Gates

- [x] Clean configure in `/tmp` with default options.
- [x] Build default target.
- [x] Run `ctest --output-on-failure`.
- [x] Configure and build Python wrapper target with current stub interfaces.
- [x] Configure and build MATLAB wrapper target with current stub interfaces.
- [x] Confirm no OptiX option or module is required.

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
