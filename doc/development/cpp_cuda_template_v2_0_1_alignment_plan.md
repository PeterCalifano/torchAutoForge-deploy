# cpp_cuda_template v2.0.1 Build Alignment Plan

This plan tracks only build/configuration consolidation. Inference, wrapper API,
TensorRT runtime, model-role, program, test, and ROS 2 design remains subject to
change and is not consolidated here.

## Baseline and authority

- [x] Preserve the target branch `feature/ort_manager_pipeline_wrapped` and its
  pre-existing staged, unstaged, untracked, and nested-repository state.
- [x] Verify signed donor tag `v2.0.1` resolves to
  `1d87153b2d060bf03c2c9adcd1df6c6d4f40ea09`.
- [x] Review donor `main` at
  `12041acb19433dfe1b98b34203f4721b5a568797`; treat its post-tag delta as
  instruction guidance rather than production code.
- [x] Keep the target product version at `0.1.0`; `v2.0.1` records template
  alignment and is not a target release number.
- [x] Keep the historical v1.11.3/v1.12.2 plan and dirty-patch snapshot as
  provenance only.

## Donor reconciliation ledger

| Donor behavior | Decision | Target treatment |
|---|---|---|
| Exact VERSION serialization | Adopt | Write the composed five-field payload once, with one terminal newline. |
| Source archive `-source` suffix | Adopt | Keep binary and source archive names distinct. |
| Generated-artifact exclusions | Adapt | Retain donor exclusions plus target ROS, log, profiling, wrapper, and development artifacts. |
| MATLAB output ownership | Adopt | Keep generated toolbox output under the build/install prefix, never the source tree. |
| Installed gtwrap include bridge | Adopt | Create a build-owned `wrap/matlab.h` compatibility bridge without modifying gtwrap. |
| Qualified build options | Adapt | Use `autoforge_deploy_*` canonically and consume legacy aliases only for top-level compatibility. |
| CUDA and embedded PTX | Retain | Keep ordinary and `.ptx.cu` examples separate and usable without OptiX. |
| Standalone TensorRT discovery | Adapt | Keep TensorRT optional and independent of the project CUDA language. |
| ONNX Runtime discovery/export | Retain | Preserve the target-owned required dependency and list-driven installed replay. |
| Python/MATLAB build machinery | Adapt | Preserve both Python identities, target RPATH, and read-only wrapper resolution. |
| OptiX and donor-only dependencies | Skip | Do not restore removed features or dependencies. |
| Donor conformance tests | Skip | Use explicit acceptance builds; add no generic project tests. |
| Donor ROS metadata versions | Skip literal import | Keep all future target ROS metadata tied to target version `0.1.0`. |
| Donor process instructions | Adapt | Preserve this repository's stricter staging, authority, and tailoring rules. |

## Build/configuration batch

- [x] Fold the applicable v2.0.1 fixes into the protected build/template index.
- [x] Keep approved build support utilities: CLogger, ordinary CUDA/PTX
  examples, Python runtime staging, development container, and container runner.
- [x] Use project-qualified options in new build automation.
- [x] Keep `build_docs` active while disabling only its Pages deployment job.
- [x] Remove the generated root `log/` artifact from the worktree.
- [x] Exclude product implementation, feature-owned CMake registration, native
  and CUDA/ROS workflows, and evolving design documentation from this batch.

## Acceptance gates

- [ ] Configure metadata-only from a fresh checkout representation without ORT
  and without modifying source VERSION metadata.
- [ ] Configure, build, install, and consume the exact staged native candidate
  with project tests disabled.
- [ ] Build ordinary CUDA plus embedded PTX with one explicit architecture.
- [ ] Configure/build TensorRT with project CUDA disabled, and the combined
  CUDA/TensorRT path, when local SDKs are available.
- [ ] Generate Python and MATLAB wrappers without changing the source tree or
  the resolved gtwrap checkout.
- [ ] Build Doxygen HTML/XML and validate the docs workflow syntax and disabled
  deployment condition.
- [ ] Generate binary/source packages and verify VERSION content, archive names,
  source exclusions, and an extracted-source configure/build.
- [ ] Audit the exact index for OptiX, donor conformance tests, unrelated product
  sources, whitespace errors, and unintended staged paths.
- [ ] Stage the single reviewed build/configuration batch and stop without
  committing, pushing, or preparing a later batch.

## Known boundary

The protected config-only checkpoint intentionally excludes the current
inference/test fixes. Its native acceptance therefore uses `ENABLE_TESTS=OFF`.
This does not waive target-owned tests for the later product batch and must not
be represented as full inference validation.
