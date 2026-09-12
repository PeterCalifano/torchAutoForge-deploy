# PTAF Deploy Release Design Review

Local release review status: Stage 12 exact-index candidate verified and staged
for consolidation review.

## Design Review

- [x] Public API is centered on `ptafdeploy::inference::CInferenceManager` for
  backend dispatch and `ptafdeploy::inference::CModelFacade` for role-level
  prototype use.
- [x] Backend-specific ORT handles stay inside the ORT backend. Wrappers expose
  facade/value types, not raw ORT objects, `void*`, or byte buffers.
- [x] Runtime configuration uses enums for backend, artifact, execution target,
  and model role. Text parsing is isolated to manifest/CLI boundaries.
- [x] Invalid scalar configuration is rejected before backend load: device id,
  thread counts, and TensorRT profile index.
- [x] Model-role adapters stay above raw tensor inference. ORT and TensorRT
  backends remain raw-tensor executors.
- [x] Shared adapter helpers cover HWC-to-NCHW conversion, centroid extraction,
  and YOLO-style raw detection decoding without duplicating local conversions.
- [x] TensorRT serialized-engine backend supports CUDA device selection,
  optimization-profile selection, and reusable host-staged CUDA buffers for
  concrete tensor shapes. DLA selection and dynamic-output allocation remain
  unqualified.
- [x] Python packaging installs from `<repo>/python` after wrapper build. The
  generated source-package link file is ignored and is not included in the
  installed wheel.
- [x] MATLAB runtime smoke tests are intentionally local-only and not part of
  hosted CI.
- [ ] CI, ROS 2, and broad README changes remain outside this product-API batch
  and require their own consolidation review.
- [ ] Confirm remote workflow execution and GitHub Pages publication before
  claiming hosted CI or published documentation availability.

## Documentation Review

- [ ] The broad README refresh remains outside this batch so its CI/ROS claims
  are reviewed with the files they describe.
- [x] `.ptafmodel` schema documents required keys, optional keys, enum domains,
  TensorRT profile selection, defaults, and path resolution.
- [x] Upgrade plan records completed stages and keeps external/manual
  validations visible without hiding them as implementation work.
- [x] Doxygen HTML/XML targets build from the exact exported index with warnings
  treated as errors.
- [x] API headers contain development comments for facade, backend, tensor,
  adapter, and wrapper-safe value blocks.

## Fresh Exact-Index Verification Evidence

The staged index was exported to a fresh source tree. The clean `lib/wrap`
checkout was linked into that export only as the wrapper generator dependency;
ordinary configuration did not modify it.

- [x] Default configure/build passes with `CPU_ENABLE_NATIVE_TUNING=OFF`.
- [x] Default target-owned CTest passes: 22/22.
- [x] Strict `WARNINGS_ARE_ERRORS=ON` build and CTest pass: 22/22.
- [x] TensorRT-enabled configure/build passes against local TensorRT 10.7 while
  project CUDA remains disabled.
- [x] TensorRT-enabled CTest passes: 23/23. The optional engine test exercised
  its no-fixture branch; this is not evidence of serialized-engine inference.
- [x] Python wrapper target builds and `autoforge_deploy_python_facade_smoke`
  passes: 1/1.
- [ ] `pip install ./python` from a temporary venv installs
  `autoforge_deploy.so` plus `libautoforge_deploy.so`.
- [x] MATLAB wrapper target builds and its target-owned runtime smoke passes:
  1/1 on MATLAB R2024b.
- [x] Doxygen target generates HTML and XML with `DOC_WARN_AS_ERROR=ON`.
- [x] Install plus fresh external consumer configure/build/run passes; the
  consumer reports 11 tensor elements through the installed target.
- [x] Jetson runner Python CLI and exact cached-diff checks pass.

## Conditional Or External Follow-Up

- [ ] Run `scripts/run_jetson_runtime_smoke.py` on Jetson hardware.
- [ ] Run ORT TensorRT Execution Provider path when an ORT build exposing
  `TensorrtExecutionProvider` is available.
- [ ] Confirm GitHub Actions jobs are green after pushing workflow changes.
- [ ] Enable/confirm GitHub Pages in repository settings after first docs upload.
- [ ] Add NMS/alternate box-format adapters when a concrete detector contract
  requires them.
- [ ] Add TensorRT dynamic-output allocator support when a target serialized
  engine exposes data-dependent output shapes.
- [ ] Add TensorRT engine-building precision tooling only if this repo takes
  ownership of engine generation.
- [ ] Make backend/model reload transactional before advertising preservation of
  a previously loaded model after a failed reload.
