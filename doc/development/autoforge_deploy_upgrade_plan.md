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

- [ ] Make `src/wrap_interface.i` top-level namespace and includes match this package.
- [ ] Populate `src/inference/inference.i` around `deploy_infer::CInferenceManager` and wrapper-safe value types.
- [ ] Expose model loading, metadata query, and float host-buffer inference to Python/MATLAB.
- [ ] Avoid wrapping raw ORT handles, raw `void*`, and `std::byte` storage directly.
- [ ] Add Python import smoke test.
- [ ] Add MATLAB wrapper generation and model-load smoke test.
- [ ] Set MATLAB-safe runtime defaults, especially ORT thread count, while allowing explicit overrides.

## Stage 4: Implementation Review And Cleanup

- [ ] Review `deploy_infer` tensor types against legacy `SInputOutputSpecs` and `SImagesInputOutputSpecs`.
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
