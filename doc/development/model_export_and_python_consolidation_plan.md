# Model Export And Python Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to
> implement this plan one review batch at a time. Do not start a later staging batch
> while an earlier batch remains in an index.

**Goal:** Establish a clear, tested path from PyTorch checkpoints to ONNX and from
ONNX to deployment artifacts, consolidate this repository on the single
`autoforge_deploy` Python package, and validate the resulting workflow on supported
desktop GPUs and a user-selected Jetson device without changing the approved C++
facade design.

**Architecture:** `pyTorchAutoForge` owns trusted reconstruction of PyTorch models
and generic Torch/TorchScript-to-ONNX export. A model-owning repository supplies the
trusted checkpoint loader and owns model-specific numerical equivalence checks.
`torchAutoForge-deploy` starts from traced artifacts, owns ONNX-to-TensorRT conversion
and runtime consumption, and provides one combined pure-Python/native-wrapper package.
TensorRT engines remain hardware- and toolchain-specific generated artifacts.

**Tech Stack:** Python 3.12+, PyTorch, ONNX, ONNX Runtime, TensorRT `trtexec`, pytest,
CMake, C++20, gtwrap Python/MATLAB bindings, Catch2, CUDA, and Jetson Linux.

**Spec:** Approved decisions and contracts are recorded in this document under
[Decision record](#decision-record).

## Global Constraints

- [ ] Re-read the nearest `AGENTS.md` before changing each repository and follow its
  local naming, testing, documentation, and staging rules.
- [ ] Preserve all user-owned staged and unstaged work. Never reset, overwrite, or
  absorb unrelated changes.
- [ ] Treat the current staged `torchAutoForge-deploy` product-wrapper batch as
  protected. Do not stage this plan or any new deployment work until that index has
  been reviewed and cleared by the user.
- [ ] Implement and stage one coherent batch at a time. After staging, inspect the
  entire cached diff, run `git diff --cached --check`, report evidence and exclusions,
  and stop for review.
- [ ] Advance only after the exact keyword `next`. A commit, tag, push, remote build,
  or system dependency upgrade requires separate explicit authorization.
- [ ] Keep generated `.onnx`, `.engine`, `.plan`, `.mat`, benchmark output, parity
  output, and conversion reports under ignored artifact directories. Do not stage
  them with source/configuration batches.
- [ ] Add only product-owned tests. Do not add donor-template conformance scripts,
  recursive CMake configure/build/install tests, package-name tests, or tests of
  behavior already owned and tested by `cpp_cuda_template_project`.
- [ ] Preserve the approved `CInferenceManager` and `CModelFacade` public design.
  Pause for discussion if implementation requires a large facade, manager, manifest,
  wrapper, or execution-session redesign.
- [ ] Keep ONNX Runtime required, standalone TensorRT optional, CUDA/PTX independent,
  and OptiX absent.
- [ ] Do not add PyTorch as a dependency of `torchAutoForge-deploy`.
- [ ] Do not modify `MachineLearningGears_for_SpaceNav`; its current dirty migrated
  files are discovery evidence only, not authoritative implementation sources.
- [ ] Use explicit errors at conversion boundaries. Do not silently change shapes,
  precision, execution target, device selection, or artifact format.
- [ ] Stop and ask the user before a system TensorRT/CUDA/driver upgrade, model contract
  change, destructive legacy cleanup, unsupported GPU workaround, or unplanned API
  compatibility break.

## Decision Record

### Repository ownership

- [x] `pyTorchAutoForge` owns PyTorch/TorchScript loading contracts, trusted loader
  plugins, generic ONNX export, and its existing TensorRT exporter API.
- [x] `ml-based-centroiding` owns exact architecture reconstruction for centroiding
  checkpoints and PyTorch-to-ONNX / ONNX-to-TensorRT numerical equivalence evaluation.
- [x] `torchAutoForge-deploy` owns conversion beginning from ONNX, the lightweight
  `ptafdeploy` artifact CLI, runtime loading, and future traced-format conversions.
- [x] `MachineLearningGears_for_SpaceNav` is outside the change set.
- [x] The deployment CLI may independently invoke `trtexec`; it does not import or
  duplicate the PyTorch-facing `TRTengineExporter` API.

### Python package contract

- [x] Publish and import one package: `autoforge_deploy`.
- [x] Keep the generated native extension private at
  `autoforge_deploy.autoforge_deploy` and re-export its public wrapper API from the
  package root when available.
- [x] Produce one combined wheel containing pure Python modules, the native extension,
  typing assets, and colocated project runtime libraries.
- [x] Do not publish a separate pure-only wheel.
- [x] Make pure imports and `ptafdeploy --help` work without the native wrapper,
  ONNX Runtime libraries, TensorRT, CUDA, or a GPU by using lazy optional imports.
- [x] Replace the obsolete `autoforgeDeployPy` package identity only after useful
  content has been classified, replaced, and preserved in its owning location.

### Model artifact layout

- [x] Use `models/` as the only repository-local artifact root.
- [x] Use `models/pytorch/` for source checkpoints, `models/onnx/` for ONNX artifacts,
  `models/tensorrt/` for `.engine`/`.plan` artifacts and sidecars, and `models/matlab/`
  for MATLAB artifacts.
- [x] Use exactly `matlab`, `onnx`, `pytorch`, and `tensorrt` as directory names; do
  not restore `.model_checkpoint`, `matlab_models`, `onnx_models`, or parallel
  top-level model directories.
- [x] Keep all directories and generated artifacts ignored unless a small licensed
  fixture is deliberately approved for source control.

### Checkpoint and export contract

- [x] Export the current checkpoint from
  `models/pytorch/best_model_plain_traveling-goat-68_22b61bbd4ddd.pth`.
- [x] Reconstruct model type `plain` with prior-vector size `0`, load its state dict
  strictly, and preserve the verified parameter count of `4,187,919` as a diagnostic
  invariant.
- [x] Write the ONNX artifact to
  `models/onnx/best_model_plain_traveling-goat-68_22b61bbd4ddd.onnx`.
- [x] Export input `image` as float32 `[1, 1, 1536, 2048]` with a dynamic batch axis.
- [x] Export output `prediction` as float32 `[batch, 2]`, representing the normalized
  centroid result.
- [x] Use explicit ONNX opset 11 for this exact checkpoint to match the current
  model-owner export path.
- [x] Validate PyTorch-to-ONNX and ONNX-to-TensorRT predictions with `rtol=1e-2` and
  `atol=1e-3`, while also reporting maximum absolute and relative error.
- [x] Use the Itokawa evaluation dataset currently available at
  `/media/peterc/SCRATCH_PRO/datasets/UniformlyScatteredPointCloudsDatasets/Itokawa/`
  `Dataset_UniformPointCloud_Itokawa_SPECTRAL_OPTIX_RT_1000_WFOV_FUTX_evaluation_ID99`.
- [x] Treat these paths as execution inputs, not library defaults or committed config.

### Device policy

- [x] Desktop GPU selection is automatic by default and considers all locally
  supported GPUs rather than assuming CUDA device 0.
- [x] `--all-supported-desktop-gpus` builds one engine per compatible desktop GPU.
- [x] Jetson never uses automatic device selection; the user must pass `--device N`.
- [x] Run `trtexec` with a child-specific `CUDA_VISIBLE_DEVICES=<physical-index>` so
  the isolated process addresses the selected GPU as its local device 0.
- [x] Include TensorRT version, compute capability, and precision in engine names;
  record GPU UUID/name and complete conversion provenance in a JSON sidecar.
- [x] Treat TensorRT engines as device/toolchain artifacts, not portable binaries.

### Deferred conversion formats

- [x] Keep the immediate implementation limited to PyTorch/TorchScript-to-ONNX in
  `pyTorchAutoForge` and ONNX-to-TensorRT in `torchAutoForge-deploy`.
- [x] Record ONNX-to-LiteRT/TFLite through optional `onnx2tf` and ONNX-initializers-
  to-MAT as future deployment edges.
- [x] Record direct Torch-to-LiteRT through LiteRT Torch as a future
  `pyTorchAutoForge` edge.
- [x] Exclude `onnx-tf` because its upstream project is inactive/deprecated.
- [x] Defer quantization, OpenVINO, CoreML, ExecuTorch, and mobile packaging until a
  target-owned requirement defines their contracts.

## Stage Tracker

- [x] Stage 0A: Record the approved plan on disk without staging it.
- [x] Stage 0B: Capture and protect the live multi-repository baseline.
- [ ] Stage 1: Complete owner review of generic trusted ONNX export and TensorRT
  compatibility in `pyTorchAutoForge`. Implementation is staged in the isolated
  owner worktree; see the 2026-09-12 reconciliation below.
- [ ] Stage 2: Add the centroiding loader adapter and export the exact ONNX model.
- [ ] Stage 3: Consolidate the deployment Python package and add ONNX-to-TensorRT CLI.
- [ ] Stage 4: Build and validate the desktop TensorRT artifact matrix.
- [ ] Stage 5: Relocate preserved legacy conversion material.
- [ ] Stage 6: Validate the explicit-device Jetson workflow.
- [ ] Stage 7: Perform independent final review and the complete validation matrix.

## Stage 0B: Baseline Protection And Path Realignment Audit

**Purpose:** Establish exact ownership and ensure the model-directory move did not
leave stale runtime, demo, configuration, or documentation paths.

### Live baseline

- [x] Record `git rev-parse --show-toplevel`, branch, `HEAD`, worktrees, submodules,
  `git status --short`, and `git diff --cached --stat` for:
  - `/home/peterc/devDir/ML-repos/pyTorchAutoForge`
  - `/home/peterc/devDir/ML-repos/ml-based-centroiding`
  - `/home/peterc/devDir/ML-repos/torchAutoForge-deploy`
  - `/home/peterc/devDir/ML-repos/MachineLearningGears_for_SpaceNav` as read-only
    exclusion evidence
- [x] Save the baseline in the implementation log section at the end of this file,
  including which changes are user-owned and which future paths belong to this plan.
- [x] Confirm the existing deployment index is unchanged. If it remains populated,
  continue work only in unstaged files and do not prepare Stage 3 for review.

### Model path audit

- [x] Search tracked and untracked text for `.model_checkpoint`, `model_checkpoint/`,
  `matlab_models`, `onnx_models`, `pytorch_models`, and old top-level model paths.
- [x] Classify each match as runtime path, configuration, documentation, generated
  cache, archived provenance, or unrelated external repository usage.
- [x] Update only active deployment-owned paths to the canonical `models/<format>/`
  layout; retain historical prose only when clearly marked as historical.
- [x] Confirm example manifests use repo-relative artifact paths and never embed
  `/home`, `/media`, or a developer-specific checkout.
- [x] Confirm `.gitignore` ignores model artifacts and conversion sidecars while
  allowing the intended directory structure/documentation to remain visible.
- [x] Remove generated `__pycache__`, `.egg-info`, conversion logs, and build output
  only when their exact paths are proven disposable and the operation is authorized.
- [x] Run a path-only smoke check that resolves each active example model config from
  the repository root and from an out-of-tree working directory.

### Stage 0B review gate

- [x] Review the complete path-only diff and confirm no model bytes or unrelated
  source files are included.
- [x] If path corrections form a coherent standalone batch, stage them by exact
  allowlist, inspect the cached diff, run `git diff --cached --check`, and stop.
- [x] Confirm no path correction must travel with Stage 3 package configuration;
  keep future package/configuration changes in their later owning batch.

## Stage 1: `pyTorchAutoForge` Generic Export CLI And TensorRT Compatibility

**Purpose:** Provide the owner-level `ptaf` command that can safely export generic
Torch/TorchScript models to ONNX without embedding model-specific reconstruction in
the framework.

The checked items below record the 2026-09-06 implementation and review report.
They do not certify the current owner index. Current acceptance remains open until
the owner reviews and validates that exact candidate; see the dated reconciliation
at the end of this document.

### Files

- [x] Create `pyTorchAutoForge/api/export/__init__.py`.
- [x] Create `pyTorchAutoForge/api/export/model_export.py` for typed export contracts,
  loader resolution, ONNX export orchestration, validation, and reporting.
- [x] Create `pyTorchAutoForge/cli/__init__.py`.
- [x] Create `pyTorchAutoForge/cli/main.py` with the `ptaf model export-onnx` command.
- [x] Modify `pyproject.toml` to publish `ptaf = "pyTorchAutoForge.cli.main:main"`.
- [x] Modify `pyTorchAutoForge/api/tensorrt/TRTengineExporter.py` only for the agreed
  builder-optimization compatibility fix.
- [x] Keep the known downstream centroiding call-site update to the canonical
  `workspace_pool_size_bytes` spelling in Stage 2, outside this batch.
- [x] Create `tests/api/export/test_model_export.py`.
- [x] Create `tests/cli/test_ptaf_cli.py`.
- [x] Extend `tests/api/tensorrt/test_TRTengineExporter.py`.
- [x] Create `doc/developments/ptaf_model_bundle_design.md` as a design-only artifact.
- [x] Update the nearest user-facing PTAF documentation with the new command and a
  runnable example.

### Public export contracts

- [x] Add immutable `ModelInputSpec` with `name: str`, `dtype: str`, and
  `shape: tuple[int, ...]`.
- [x] Add immutable `ModelExportRequest` with checkpoint path, input specs, selected
  `torch.device`, opset, and validation settings.
- [x] Add immutable `ModelExportBundle` with reconstructed `torch.nn.Module`, sample
  input tensors, input/output names, and dynamic-axis mapping.
- [x] Define a typed loader callable accepting `ModelExportRequest` and returning
  `ModelExportBundle`.
- [x] Keep framework naming aligned with PTAF conventions: public functions begin
  with a capital letter and use snake case; internal helpers begin with `_`.
- [x] Validate input names, supported dtype names, positive static dimensions,
  non-empty outputs, opset, source extension, destination extension, and overwrite
  policy before invoking PyTorch.
- [x] Parse CLI inputs as `name:dtype:dim,dim,...`, including the concrete form
  `image:float32:1,1,1536,2048`.
- [x] Support direct `.pt` TorchScript loading without a model-specific plugin.
- [x] Require `--loader module:function` for `.pth` state dictionaries.
- [x] Import the loader only after explicit user selection, verify it is callable,
  and present import/signature failures as actionable errors.
- [x] Do not call a generic `LoadModel(model=None)`, execute checkpoint-embedded code,
  infer an architecture from parameter names, or accept arbitrary code stored in a
  checkpoint.
- [x] Set the reconstructed model to evaluation mode, move it and inputs to the
  selected device, and export under inference/no-grad semantics.
- [x] Run ONNX checker validation and an ONNX Runtime smoke inference when requested.
- [x] Write a JSON export report containing source/output hashes, loader identifier,
  input/output contract, opset, tool versions, device, timestamp, and validation
  results.
- [x] Return exit code `0` on success, `2` for invalid user input/configuration, and
  `1` for runtime/export failure.

### TensorRT compatibility fix

- [x] Preserve `workspace_pool_size_bytes` as the sole public spelling and retain
  rejection of the removed `workspace_size_bytes` alias.
- [x] Add `builder_optimization_level: int | None` to `TRTengineExporterConfig` and
  the constructor, validating the inclusive range `0..5`.
- [x] In Python TensorRT mode, assign
  `IBuilderConfig.builder_optimization_level` only when the runtime exposes it;
  otherwise issue a clear warning and continue without changing other settings.
- [x] In `trtexec` mode, probe `trtexec --help` once per exporter instance and emit
  `--builderOptimizationLevel=<N>` only when advertised.
- [x] Preserve existing compatibility for `--memPoolSize` versus `--workspace` and
  `set_memory_pool_limit` versus `max_workspace_size`.
- [x] Do not change the established public exporter class/mode/precision/profile API
  or make Jetson depend on x86-only discovery.

### PTAF bundle design draft

- [x] Specify `.ptafbundle` so it cannot be confused with deployment `.ptafmodel`
  manifests.
- [x] Define a versioned JSON manifest with a registered PTAF architecture ID,
  factory configuration, input/output contracts, hash/provenance data, and safe
  non-pickle weight storage.
- [x] Forbid arbitrary import strings and embedded executable code in the bundle.
- [x] Record migration/versioning/security questions and acceptance criteria without
  implementing bundle loading in this plan.

### Stage 1 tests

- [x] Write failing focused tests first for input-spec parsing, `.pth` loader
  requirement, bad loader identifiers, typed loader result validation, overwrite
  protection, and documented exit codes.
- [x] Export a tiny traced fixture to a pytest temporary directory; validate its graph
  and execute one ONNX Runtime inference.
- [x] Export a tiny state-dict fixture through a test-only trusted loader plugin.
- [x] Test builder optimization levels `0`, `5`, `-1`, and `6`.
- [x] Test Python TensorRT behavior with and without the optimization-level property.
- [x] Test `trtexec` command construction with and without the advertised flag.
- [x] Run the focused export, CLI, and TensorRT tests, then the PTAF default pytest
  suite according to its `AGENTS.md` marker policy.
- [x] Run static typing/lint checks configured by the repository for all new public
  code and verify `ptaf --help` and `ptaf model export-onnx --help` in a clean venv.

### Stage 1 review gate

- [x] Review new modules as a public API reader, checking type annotations, Google-
  style docstrings, lazy imports, examples, error messages, and absence of redundant
  abstractions.
- [x] Stage only the reviewed PTAF source, tests, documentation, and packaging paths.
- [x] Inspect the complete cached diff and run `git diff --cached --check`.
- [x] Report exact tests, excluded dirty paths, and proposed commit message, then stop.

**Proposed subject:** `Add generic model export CLI`

**Proposed body:**

- Add trusted plugin-based checkpoint reconstruction and TorchScript export

- Validate ONNX contracts and record reproducible export provenance

- Extend TensorRT builder optimization compatibility without changing exporter API

## Stage 2: Model-Owned Centroiding Adapter And Exact ONNX Export

**Purpose:** Keep architecture-specific checkpoint reconstruction in
`ml-based-centroiding` while making it callable through the generic PTAF export API.

### Files

- [ ] Create `python/ml_based_centroiding/deployment/__init__.py`.
- [ ] Create `python/ml_based_centroiding/deployment/checkpoint_export.py`.
- [ ] Refactor `python/scripts/export_evaluate_onnx.py` to consume the shared
  reconstruction/export adapter rather than maintaining a second export contract.
- [ ] Update `python/scripts/export_evaluate_tensorrt.py` to use
  `workspace_pool_size_bytes` and builder optimization level `0` through the PTAF
  public exporter API.
- [ ] Create `python/tests/test_checkpoint_export.py` or extend the closest existing
  checkpoint-loading test when that produces a clearer cohesive suite.
- [ ] Update model-owner documentation with the exact checkpoint, input/output
  contract, export command, and artifact location.

### Loader adapter

- [ ] Implement the generic PTAF loader callable using the repository's current
  checkpoint/config reconstruction utilities.
- [ ] Accept the `ModelExportRequest`, validate the `.pth` source and requested input
  contract, reconstruct the exact model class, and call strict state-dict loading.
- [ ] Reject architecture/configuration mismatches instead of attempting partial or
  non-strict loading.
- [ ] Verify model type `plain`, prior-vector size `0`, and parameter count
  `4,187,919` for the selected checkpoint before export.
- [ ] Return a single float32 sample input named `image`, one output named
  `prediction`, and dynamic batch axes for both.
- [ ] Keep image preprocessing, output interpretation, and dataset evaluation in the
  model-owner repository; do not move them into PTAF or the deploy backend.

### Exact ONNX export

- [ ] Run `ptaf model export-onnx` against
  `torchAutoForge-deploy/models/pytorch/best_model_plain_traveling-goat-68_22b61bbd4ddd.pth`
  with the explicit centroiding loader, opset 11, and
  `image:float32:1,1,1536,2048`.
- [ ] Write to
  `torchAutoForge-deploy/models/onnx/best_model_plain_traveling-goat-68_22b61bbd4ddd.onnx`
  and refuse an overwrite unless explicitly requested.
- [ ] Run `onnx.checker`, inspect graph I/O names/dtypes/shapes, and record artifact
  SHA-256 and file size.
- [ ] Execute PyTorch and ONNX Runtime on identical Itokawa samples.
- [ ] Assert `rtol=1e-2`, `atol=1e-3`; report sample count, maximum absolute error,
  maximum relative error, and any failing sample identifiers.
- [ ] Retain the ignored ONNX and JSON report locally for Stage 3/4; do not stage them.

### Stage 2 tests and review gate

- [ ] Run the existing checkpoint-loading suite plus new adapter contract tests.
- [ ] Run the model-owner ONNX export/evaluation tests and the exact artifact parity
  command on the available dataset.
- [ ] Confirm the refactored scripts preserve model choices, prior-vector models,
  device handling, and existing evaluation behavior beyond the selected plain model.
- [ ] Review and stage only model-owner source, tests, and documentation by explicit
  allowlist; leave generated artifacts unstaged.
- [ ] Inspect the cached diff, run `git diff --cached --check`, report evidence and
  exclusions, and stop.

**Proposed subject:** `Add checkpoint adapter for deployment export`

**Proposed body:**

- Reconstruct centroiding checkpoints through a typed PTAF loader contract

- Share export configuration without weakening strict checkpoint validation

- Preserve model-owned PyTorch, ONNX, and TensorRT equivalence evaluation

## Stage 3: Unified Deployment Python Package And ONNX-To-TensorRT CLI

**Entry gate:** The protected `torchAutoForge-deploy` index must be reviewed and
cleared. If it is still populated, stop before staging or modifying overlapping
packaging/wrapper paths.

**Purpose:** Combine pure deployment utilities and generated bindings under one
package while keeping the converter independent of PyTorch and optional native
runtime dependencies.

### Files

- [ ] Create `python/autoforge_deploy/cli.py`.
- [ ] Create `python/autoforge_deploy/__main__.py`.
- [ ] Create `python/autoforge_deploy/conversion/__init__.py`.
- [ ] Create `python/autoforge_deploy/conversion/devices.py`.
- [ ] Create `python/autoforge_deploy/conversion/tensorrt.py`.
- [ ] Create `python/autoforge_deploy/conversion/reports.py`.
- [ ] Modify `python/autoforge_deploy/__init__.py` to preserve lazy wrapper loading
  and expose pure package metadata without importing conversion backends eagerly.
- [ ] Modify `python/pyproject.toml.in` so the distribution and import package are
  both `autoforge_deploy`, all pure subpackages are included, and
  `ptafdeploy = "autoforge_deploy.cli:main"` is installed.
- [ ] Modify `python/setup.py.in` only if the CMake wrapper build still consumes it;
  keep setup metadata generated from the same project identity/version source.
- [ ] Update CMake install/wheel inputs only where needed to colocate pure modules,
  extension, typing files, and runtime libraries.
- [ ] Create focused Python tests under the active package test location for CLI
  parsing, device selection, command construction, sidecar generation, and lazy
  import behavior.
- [ ] Create `doc/development/artifact_conversion_graph_design.md`.
- [ ] Update README/package documentation with installation prerequisites and
  desktop/Jetson examples.

### Package consolidation behavior

- [ ] Keep `import autoforge_deploy` successful when the native extension is absent;
  expose `HAS_WRAPPER` and a useful wrapper import error as the current package does.
- [ ] Make `python -m autoforge_deploy --help` and `ptafdeploy --help` succeed in a
  clean environment without TensorRT, ONNX Runtime, CUDA, or the compiled extension.
- [ ] Import GPU discovery and conversion dependencies inside the commands that need
  them.
- [ ] Include `autoforge_deploy/conversion/**` and all pure modules in the generated
  wheel; verify installed package contents instead of relying on build-tree imports.
- [ ] Keep external ONNX Runtime libraries as prerequisites and preserve the target's
  build/install RPATH policy.
- [ ] Eliminate active imports of `autoforgeDeployPy`, `AutoForgeDeployPy`, and other
  case variants.
- [ ] Do not delete the old package tree in this stage; remove it from active packaging
  only after Stage 5 has relocated its useful material.

### `ptafdeploy convert onnx-to-trt`

- [ ] Accept `--onnx`, `--output-dir`, `--precision`, `--workspace-mib`,
  `--builder-optimization-level`, repeated shape specifications, `--device`,
  `--all-supported-desktop-gpus`, `--trtexec`, `--overwrite`, and report options.
- [ ] Default to FP16, 256 MiB workspace, builder optimization level `0`, desktop
  device `auto`, and no overwrite.
- [ ] Require explicit input shapes for dynamic ONNX inputs. Permit one `--opt-shape`
  to populate min/opt/max for a fixed-shape engine, while preserving separate
  min/opt/max flags for genuine dynamic profiles.
- [ ] Use exact current shape `image:1x1x1536x2048` for the selected model.
- [ ] Resolve `trtexec` in this order: explicit `--trtexec`,
  `$TENSORRT_ROOT/bin/trtexec`, then `PATH`.
- [ ] Run `trtexec --help` once and select supported modern/legacy workspace flags and
  the optimization-level flag from observed capabilities.
- [ ] Use argument lists without shell interpolation, preserve stdout/stderr in the
  report, and surface the failing command and exit status.
- [ ] Return exit code `0` on success, `2` for invalid user input/configuration, and
  `1` for discovery/build/runtime failure.

### Desktop GPU selection

- [ ] Detect desktop versus Jetson before resolving `auto`.
- [ ] Query desktop GPUs without initializing PyTorch; prefer `nvidia-smi` query data
  and isolate the optional NVML implementation.
- [ ] Record physical index, UUID, name, compute capability, free memory, total
  memory, and driver version for each visible candidate.
- [ ] Rank automatic candidates deterministically by supported compute capability,
  available memory, then stable physical index.
- [ ] Probe TensorRT compatibility per candidate without treating an unrelated ONNX
  parser/configuration error as an unsupported GPU.
- [ ] For default `auto`, build on the highest-ranked compatible GPU and report every
  rejected candidate with its reason.
- [ ] For `--all-supported-desktop-gpus`, build one artifact per compatible GPU and
  fail the overall command on unexpected model/configuration errors.
- [ ] Never hardcode RTX model names or ordinal 0 into library policy.

### Jetson selection

- [ ] Detect Jetson from `/etc/nv_tegra_release` or `/proc/device-tree/model`.
- [ ] Reject `--device auto` on Jetson with a message requiring `--device N`.
- [ ] Validate that the requested ordinal exists before launch.
- [ ] Avoid `nvidia-smi` as a required Jetson dependency and keep memory-conscious
  defaults suitable for unified-memory systems.

### Artifact naming and sidecar

- [ ] Name engines using source stem, normalized TensorRT version, compute capability,
  and precision, for example
  `<stem>_trt10.7_sm89_fp16.engine`.
- [ ] Avoid collision between different devices/toolchains; require `--overwrite` to
  replace an existing engine or sidecar.
- [ ] Write a versioned JSON sidecar beside each engine with source/engine SHA-256,
  command arguments, resolved executable, TensorRT/CUDA/driver versions, GPU
  UUID/name/capability, precision, workspace, builder level, shapes, timestamp,
  process status, and validation results.
- [ ] Write reports atomically after successful engine creation; retain a separate
  failure report without presenting a partial engine as valid.

### Future conversion graph design

- [ ] Define a registry/graph rooted at ONNX with immediate ONNX-to-TensorRT and
  deferred ONNX-to-LiteRT/TFLite and ONNX-initializers-to-MAT edges.
- [ ] Assign direct Torch-to-LiteRT ownership to PTAF through LiteRT Torch.
- [ ] Record optional dependency boundaries, artifact/report contracts, capability
  discovery, error policy, and extension criteria.
- [ ] Use these primary references in the design:
  - <https://github.com/google-ai-edge/ai-edge-torch>
  - <https://github.com/PINTO0309/onnx2tf>
  - <https://github.com/onnx/onnx-tensorflow>
- [ ] Do not implement deferred graph edges in this plan.

### Stage 3 tests

- [ ] Write failing tests first for parsing, exit codes, binary resolution precedence,
  help probing, workspace compatibility, optimization flag compatibility, shapes,
  overwrite protection, desktop ranking, Jetson explicit-device rejection, artifact
  naming, hashes, and report schema.
- [ ] Mock subprocess/GPU discovery for deterministic unit tests; add one explicit
  local integration command using the real `trtexec` outside default test collection.
- [ ] Build a combined wheel, install it into a clean venv, inspect wheel contents,
  import `autoforge_deploy`, import the native facade, and run both help commands.
- [ ] Run existing Python wrapper/facade smoke tests and target-owned C++ tests affected
  by package changes.
- [ ] Do not register pytest, wheel-build, recursive CMake, or template-verification
  checks as ordinary CTest entries.

### Stage 3 review gate

- [ ] Review source/API docs, lazy-import boundaries, subprocess safety, deterministic
  selection, generated metadata, wrapper coexistence, and packaging contents.
- [ ] Stage only the unified package, conversion CLI, focused product tests, build
  configuration, and directly supporting documentation through an explicit allowlist.
- [ ] Inspect the full cached diff, run `git diff --cached --check`, rerun checks from
  the staged representation, report exclusions, and stop.

**Proposed subject:** `Unify Python deployment package and conversion CLI`

**Proposed body:**

- Package pure deployment tools with the generated native wrapper

- Add reproducible ONNX-to-TensorRT conversion through `trtexec`

- Enforce automatic desktop and explicit Jetson device selection

## Stage 4: Desktop ONNX-To-TensorRT Artifact Matrix

**Purpose:** Generate and validate local hardware-specific artifacts after the Stage 3
source batch is reviewed. Generated outputs remain ignored and unstaged.

### Environment preflight

- [ ] Record `nvidia-smi`, driver, CUDA runtime/toolkit, TensorRT, `trtexec --version`,
  `trtexec --help`, ONNX Runtime provider, and free-space information.
- [ ] Confirm the currently discovered desktop devices: RTX 5090 compute capability
  12.0 and RTX 4070 Ti SUPER compute capability 8.9.
- [ ] Confirm the current `trtexec` path
  `/usr/local/tensorrt10.7-cuda12.6/bin/trtexec` and TensorRT version 10.7, or record
  the live replacement if the environment changed.
- [ ] Do not install or upgrade TensorRT, CUDA, drivers, ONNX, ONNX Runtime, or system
  packages during preflight.

### Build and validate

- [ ] Invoke `ptafdeploy convert onnx-to-trt --all-supported-desktop-gpus` for the
  exact ONNX model using FP16, 256 MiB workspace, builder optimization level 0, and
  fixed shape `image:1x1x1536x2048`.
- [ ] Confirm the converter selects supported devices automatically and generates one
  uniquely named engine/sidecar per successful GPU.
- [ ] If TensorRT 10.7 cannot support the RTX 5090 toolchain/compute capability, stop
  and report the exact parser/builder diagnostics before proposing any upgrade.
- [ ] Do not misclassify ONNX graph, shape, memory, or command errors as unsupported
  GPU architecture.
- [ ] Load each successful engine with the deploy TensorRT runtime and execute a smoke
  inference on its owning GPU.
- [ ] Compare ONNX Runtime and TensorRT predictions on identical Itokawa samples using
  `rtol=1e-2`, `atol=1e-3`; record error extrema and sample identifiers.
- [ ] Run the existing generic benchmark CLI through `CModelFacade` for ORT CPU, ORT
  CUDA, and each TensorRT engine; record warmup, iteration count, latency statistics,
  provider/device, and artifact hash.
- [ ] Keep all engines, sidecars, benchmark results, and parity output under ignored
  `models/tensorrt/` or a dedicated ignored report directory.

### Stage 4 review gate

- [ ] Review reports for source hash consistency, correct physical device isolation,
  engine naming, expected I/O, and numerical equivalence.
- [ ] Record evidence in this plan's implementation log.
- [ ] Stage nothing from Stage 4 unless a source defect is found; handle any source
  correction as a separately reviewed Stage 3 follow-up.

## Stage 5: Selective Legacy Relocation

**Purpose:** Remove the obsolete second package from active code without discarding
unique historical or prototype material.

### Classification and destinations

- [ ] Treat `python/scripts/tmp_to_rework_as_generic/ExportPytorchToONNX.py` as a
  hard-coded predecessor replaced by PTAF plus the centroiding loader; preserve a
  concise provenance note rather than active implementation.
- [ ] Treat `python/scripts/tmp_to_rework_as_generic/TestOnnxAccuracy.py` and its
  NeuralCOB-specific `.mat` result as model-owner material; relocate them to a clearly
  named archive under `ml-based-centroiding` only after the current parity workflow
  covers their useful behavior.
- [ ] Preserve `python/scripts/tmp_to_rework_as_generic/SaveOnnxWeights.py` as the
  prototype for future ONNX-initializers-to-MAT conversion and move it to a clearly
  marked experimental/legacy deployment documentation area.
- [ ] Move `python/scripts/tflite-tutorial.ipynb` to an experimental deployment
  tutorial/archive area and label it as reference material outside current support.
- [ ] Classify every file in `python/autoforgeDeployPy/`; preserve useful design notes
  or algorithms, but do not preserve broken imports, empty placeholders, generated
  versions, caches, or duplicate packaging metadata as active code.

### Relocation rules

- [ ] Prove each useful behavior has a tested replacement or an explicit archived
  destination before removing the original active path.
- [ ] Use history-preserving moves where repository boundaries permit; when crossing
  repositories, retain source path, source revision, and migration date in the
  destination provenance note.
- [ ] Do not copy dirty `MachineLearningGears_for_SpaceNav` versions into either owner.
- [ ] Remove obsolete package/build references only after the unified wheel and clean
  imports pass.
- [ ] Delete generated `__pycache__` and `.egg-info` artifacts from the worktree after
  verifying they are not tracked; keep deletion scope explicit and narrow.
- [ ] Search again for `autoforgeDeployPy`, `AutoForgeDeployPy`, legacy model folders,
  old converter script paths, and stale documented commands.

### Stage 5 review gate

- [ ] Review relocation provenance, replacements, docs, packaging, and all deletions.
- [ ] Stage the deployment relocation/removal batch separately from the earlier CLI
  implementation. If model-owner archive changes are staged, prepare a separate
  `ml-based-centroiding` review batch.
- [ ] Inspect each full cached diff, run `git diff --cached --check`, report preserved
  and removed material, and stop.

**Proposed deployment subject:** `Relocate legacy model conversion prototypes`

**Proposed body:**

- Preserve unique conversion references outside the active Python package

- Remove obsolete package identity after verified replacement

- Document ownership of model-specific and future conversion workflows

## Stage 6: Explicit-Device Jetson Hardware Gate

**Purpose:** Validate deployment on the target platform without transferring a desktop
engine or assuming desktop GPU discovery behavior.

### Access and preflight

- [ ] Wait until the configured `jetson_jammy` target is reachable; connection timeout
  is a pending hardware gate, not a reason to weaken validation.
- [ ] Obtain the user-selected Jetson device ordinal for the run and pass it explicitly.
- [ ] Record Jetson model, JetPack/L4T, architecture, Python, CUDA, TensorRT,
  `trtexec`, ONNX Runtime providers, memory, and free-space information.
- [ ] Transfer or otherwise make available the verified ONNX artifact and source hash;
  never reuse a desktop `.engine`/`.plan`.
- [ ] Do not change remote system packages, services, drivers, or repository state
  without explicit authorization.

### Build and runtime validation

- [ ] Confirm `ptafdeploy convert onnx-to-trt` rejects omitted/automatic device
  selection on Jetson.
- [ ] Build locally on Jetson with `--device <user-ordinal>`, FP16, 256 MiB workspace,
  builder optimization level 0, and `image:1x1x1536x2048`.
- [ ] Verify artifact/sidecar hashes, target metadata, I/O contract, and engine naming.
- [ ] Load the engine through `CInferenceManager`/`CModelFacade`, run one inference,
  then run the established Jetson benchmark/smoke path.
- [ ] Compare ONNX Runtime and TensorRT outputs with `rtol=1e-2`, `atol=1e-3` on an
  identical bounded sample set.
- [ ] Record memory use, warmup/latency statistics, errors, and device ordinal.
- [ ] Copy back only reports when authorized; keep engine artifacts local/ignored.

### Stage 6 gate

- [ ] Mark Jetson validation complete only with live hardware evidence.
- [ ] If access remains unavailable, report final consolidation as pending Jetson
  validation rather than passing by inference from desktop tests.
- [ ] Stage no source solely for hardware evidence. Any discovered defect returns to
  the owning earlier stage as a separate reviewed fix.

## Stage 7: Independent Final Review And Validation

**Purpose:** Reassess the complete multi-repository result from clean environments and
from the exact staged representations before consolidation.

### 7A. Scope and ownership review

- [ ] Re-read every applicable `AGENTS.md` and this plan.
- [ ] Capture branch, `HEAD`, full status, staged diff, unstaged diff, submodules, and
  worktrees for PTAF, model owner, deploy, and the read-only excluded repository.
- [ ] Confirm every changed path maps to one approved stage and every checked box has
  concrete evidence in the implementation log.
- [ ] Confirm no `MachineLearningGears_for_SpaceNav` change was made by this work.
- [ ] Confirm no generated ONNX, TensorRT, MATLAB, benchmark, dataset, cache, wheel,
  or report artifact is tracked or staged unintentionally.
- [ ] Confirm there is one active distribution/import package,
  `autoforge_deploy`, and no active legacy-case imports remain.
- [ ] Confirm Torch/checkpoint reconstruction lives only in PTAF/model-owner code and
  deploy conversion begins from ONNX.
- [ ] Confirm `CInferenceManager`, `CModelFacade`, ORT, optional TensorRT, wrappers,
  independent CUDA/PTX, ROS 2 overlay, and OptiX absence retain their approved roles.
- [ ] Confirm no template conformance, recursive CMake, package-name, or generic donor
  tests were added.

### 7B. Code and documentation review

- [ ] Review all new/substantially modified source files as a reader receives them:
  module/public API docs, precise types, error contracts, lazy imports, ownership,
  subprocess safety, path handling, and non-obvious invariants.
- [ ] Simplify redundant helpers, branches, loops, state, and indirection only where
  behavior is preserved and the change stays inside the approved design.
- [ ] Check command help, README examples, package docs, conversion graph design,
  bundle design, sidecar schema, and model-owner docs against live behavior.
- [ ] Check all paths from repository root, installed environments, and out-of-tree
  working directories.
- [ ] Check formatting/line length and run each repository's configured static tools.

### 7C. PTAF validation

- [ ] Create a clean Python environment and install PTAF with its required export/test
  dependencies.
- [ ] Run focused export/CLI/TensorRT tests, including direct TorchScript export and a
  trusted state-dict loader fixture.
- [ ] Run the relevant default PTAF pytest suite under its marker policy.
- [ ] Run configured typing/static checks on the changed modules.
- [ ] Verify `ptaf --help`, `ptaf model export-onnx --help`, documented exit codes,
  overwrite behavior, ONNX checker, ORT smoke inference, and JSON report schema.
- [ ] Verify existing TensorRT exporter public imports and caller compatibility.

### 7D. Model-owner validation

- [ ] Run the complete existing checkpoint-loading tests plus the new loader adapter
  tests.
- [ ] Strictly reconstruct the selected checkpoint and confirm `4,187,919` parameters.
- [ ] Re-export the exact ONNX model from a clean artifact directory and verify its
  hash/report, I/O contract, and opset.
- [ ] Run the model-owner PyTorch/ONNX parity evaluation on the Itokawa dataset and
  record `rtol=1e-2`, `atol=1e-3`, sample count, and error extrema.
- [ ] Run the model-owner ONNX/TensorRT parity path against each validated engine.
- [ ] Verify refactoring did not regress prior-informed or other supported model types.

### 7E. Deployment native, package, and wrapper validation

- [ ] Configure/build/test a fresh RelWithDebInfo CPU build through `./build_lib.sh`.
- [ ] Configure/build/test the optional TensorRT build with the live `TENSORRT_ROOT`.
- [ ] Run existing target-owned CTest suites; do not add generic template checks.
- [ ] Build/install the combined Python wheel from a clean build tree and inspect its
  file list and dynamic-library dependencies.
- [ ] In a clean venv, import pure `autoforge_deploy`, verify no wrapper case, then
  install the wrapper wheel and run facade/model-config/inference smoke behavior.
- [ ] Run `ptafdeploy --help` without optional runtime dependencies and the real
  ONNX-to-TensorRT command with explicit artifact inputs.
- [ ] Run existing MATLAB wrapper generation/load/inference smoke checks because
  package/wrapper paths changed; if MATLAB is unavailable, report this gate as unrun.
- [ ] Run existing ROS 2, CUDA, and PTX gates only if their files/configuration were
  touched or their packaging contract changed; report intentional non-runs precisely.
- [ ] Verify install/package/consumer behavior through explicit fresh acceptance
  commands rather than registering recursive tests in CTest.

### 7F. Hardware and numerical validation

- [ ] Repeat desktop automatic selection and all-supported-GPU builds from clean
  ignored output directories.
- [ ] Verify each engine on its owning GPU through the public deploy facade and generic
  benchmark CLI.
- [ ] Verify the user-selected Jetson build/runtime/parity gate on live hardware.
- [ ] Check that artifact sidecars are reproducible, complete, and match source/engine
  hashes and actual hardware/tool versions.
- [ ] Distinguish unit, integration, local desktop, and live Jetson evidence in the
  final report.

### 7G. Final staged-review gate

- [ ] Reconcile plan checkboxes with evidence; leave unavailable hardware or external
  service gates unchecked.
- [ ] For each repository, review the entire cached diff, run
  `git diff --cached --check`, and rerun checks affected by the staged representation.
- [ ] Confirm no earlier protected batch remains mixed with a later batch.
- [ ] Report staged paths, functional purpose, validation commands/results, caveats,
  unrun gates, ignored artifacts, excluded dirty work, and exact proposed commit
  subject/body for each repository.
- [ ] Stop for user consolidation review. Do not commit, tag, push, alter remotes, or
  clear an index without explicit authorization.

## Acceptance Criteria

- [ ] A trusted-loader `.pth` workflow and a direct TorchScript workflow export valid
  ONNX through `ptaf model export-onnx`.
- [ ] The selected plain centroiding checkpoint exports to the agreed ONNX path with
  the exact I/O contract and passes model-owner parity tolerances.
- [ ] `autoforge_deploy` is the only active Python distribution/import package and its
  combined wheel contains pure utilities plus the native wrapper.
- [ ] `ptafdeploy convert onnx-to-trt` works without PyTorch, resolves compatible
  `trtexec` flags, and writes reproducible hardware-specific sidecars.
- [ ] Desktop default selection is automatic; all-supported mode covers every locally
  compatible GPU; Jetson requires an explicit device ordinal.
- [ ] TensorRT builder optimization level 0 works across supported Python/trtexec
  compatibility paths without restoring removed workspace aliases.
- [ ] Legacy useful content is preserved with provenance, while broken duplicate
  package/configuration paths are no longer active.
- [ ] No generated model artifact, dataset, result, cache, or unrelated dirty file is
  included in a source/configuration batch.
- [ ] Native C++, Python/wheel, wrappers, ONNX, TensorRT desktop, and Jetson gates are
  either passed with fresh evidence or explicitly reported as pending.
- [ ] The approved facade/backend/adapter design is unchanged and no generic template
  test has been introduced.

## Implementation Log

Use one entry per work session or review batch. Do not mark a stage complete until
its evidence is entered here.

### 2026-08-26 — Plan creation

- [x] Recorded the approved repository ownership, package, path, checkpoint, TensorRT,
  desktop selection, Jetson selection, legacy preservation, review, and validation
  decisions.
- [x] Confirmed the current deploy artifact layout contains `models/matlab/`,
  `models/onnx/`, and `models/pytorch/`, including the selected `.pth` checkpoint.
- [x] Confirmed the deployment repository has a pre-existing populated index and broad
  unstaged work; this new plan is intentionally left unstaged.
- [x] Confirmed PTAF, model-owner, and excluded legacy repositories are independently
  dirty and require exact path allowlists before any future staging.
- [x] Added the Stage 0B baseline identifiers and path-audit findings before
  implementation began.

### 2026-09-01 — Stage 0B live baseline and path audit

- [x] Captured all four repositories with clear indexes and one ordinary worktree
  each:
  - `pyTorchAutoForge`: branch `feature/upgrade-formalize-explainer-module`, HEAD
    `5fd58b123ef0dca541cc861223d00a681caa6a12`; submodules
    `CommManager4MATLAB@4f72439` and nested `matlab-msgpack_PeterCdev@3c4492a`;
  - `ml-based-centroiding`: branch `develop`, HEAD
    `ba34bdd8d7fe09e663b7c506f9b4b92afdfc8333`, with no submodules;
  - `torchAutoForge-deploy`: branch `feature/ort_manager_pipeline_wrapped`, HEAD
    `5334aa33bae0f119ba00890e43c63d0c531a9f89`, with
    `lib/wrap@bf9f786`; and
  - read-only `MachineLearningGears_for_SpaceNav`: branch
    `dev_refactor_conv_ncof_for_onnx`, HEAD
    `eadd49d5965e62c7219ecbd2b3317e8abf8b9f46`, with
    `EfficientPose_PeterCdev@64b7119`.
- [x] Saved the exact pre-Stage0B `git status --short` baseline:

```text
[pyTorchAutoForge]
 M .vscode/c_cpp_properties.json
 M AGENTS.md
 M doc/datasets.md
 M pyTorchAutoForge/__init__.py
 M pyTorchAutoForge/datasets/AugmentationsManager.py
 M pyTorchAutoForge/datasets/__init__.py
 M pyTorchAutoForge/datasets/vector_error_models/VectorErrorsBaseClasses.py
 M pyTorchAutoForge/datasets/vector_error_models/__init__.py
 M pyTorchAutoForge/evaluation/ModelExplainer.py
 M pyTorchAutoForge/evaluation/__init__.py
 M pyTorchAutoForge/extra/experimental_pysr_module.py
 M pyTorchAutoForge/extra/xgboost_regression_module.py
 M pyproject.toml
 M tests/datasets/test_VectorErrorsBaseClasses.py
 M tests/optimization/test_model_training_manager.py
?? CLAUDE.md
?? doc/developments/library_implementation_review_2026-07-18.md
?? doc/developments/neuralcob_mlgears_migration.md
?? doc/developments/reports/
?? doc/developments/sequence_and_esa_loader_implementation.md
?? examples/example_ModelExplainer.py
?? pyTorchAutoForge/evaluation/explainability/
?? tests/datasets/test_AugmentationsManager.py
?? tests/evaluation/explainability/
?? tests/extra/

[ml-based-centroiding]
 M ml-based-centroiding.code-workspace

[torchAutoForge-deploy]
 M .gitignore
 M CLAUDE.md
 M README.md
 M TODO
 D doc/development/cpp_cuda_template_v1_11_3_upgrade_plan.md
 D doc/development/cpp_cuda_template_v1_12_2_dirty.patch
 D doc/development/cpp_cuda_template_v2_0_1_alignment_plan.md
 M matlab/programs/ModelCodegenProgram.m
 M matlab/programs/ModelImportProgram.m
 M python/scripts/tmp_to_rework_as_generic/ExportPytorchToONNX.py
 M src/auxiliary/common_defs.h
 M src/auxiliary/common_ops.h
 M src/auxiliary/images_prepro.h
 M src/inference/inference_common.h
 M src/inference/inference_manager.h
?? .github/workflows/build_linux.yml
?? .github/workflows/build_linux_cuda.yml
?? doc/development/mobile_deployment_pipeline_design.md
?? doc/development/model_export_and_python_consolidation_plan.md
?? examples/model_configs/ml_based_centroiding_film.ptafmodel
?? tests/inference/testRealCentroidingOnnx.cpp

[MachineLearningGears_for_SpaceNav, read-only]
 M .vscode/c_cpp_properties.json
 M MachineLearningGears_for_SpaceNav/centroiding_based_nav/datasets/__init__.py
 M MachineLearningGears_for_SpaceNav/centroiding_based_nav/datasets/dataset_processing.py
 M MachineLearningGears_for_SpaceNav/centroiding_based_nav/models/__init__.py
 M MachineLearningGears_for_SpaceNav/centroiding_based_nav/models/blob_analysis_module.py
 M MachineLearningGears_for_SpaceNav/centroiding_based_nav/models/centroid_range_convolutionalNeuralCOF.py
 M MachineLearningGears_for_SpaceNav/centroiding_based_nav/models/centroid_range_neuralCOB.py
 M MachineLearningGears_for_SpaceNav/experimental/experiment_regressor_search.py
 M MachineLearningGears_for_SpaceNav/experimental/experiment_regressor_search_combinedRegr.py
 M MachineLearningGears_for_SpaceNav/experimental/pysr_modular/example_SymbolicRegression.py
 M MachineLearningGears_for_SpaceNav/experimental/pysr_modular/experiment_sswcob_regressor_search_selector.py
 M MachineLearningGears_for_SpaceNav/scripts/centroiding_based_nav/centroid_range_single_training.py
 M MachineLearningGears_for_SpaceNav/scripts/centroiding_based_nav/xgboost_cob_regressor_training.py
 M MachineLearningGears_for_SpaceNav/scripts/extra/experiment_regressor_search.py
 M MachineLearningGears_for_SpaceNav/scripts/extra/experiment_regressor_search_combinedRegr.py
 M MachineLearningGears_for_SpaceNav/scripts/start_mlflow_server.sh
 M pyproject.toml
 M tests/centroid_based_nav/test_blob_analysis_module.py
 M tests/centroid_based_nav/test_models_export.py
 M tests/centroid_based_nav/test_models_integration.py
 M tests/centroid_based_nav/test_models_unit.py
?? MachineLearningGears_for_SpaceNav/centroiding_based_nav/datasets/augmentations.py
?? MachineLearningGears_for_SpaceNav/centroiding_based_nav/datasets/datasets_handling.py
?? MachineLearningGears_for_SpaceNav/centroiding_based_nav/models/film_centroiding_convnn.py
?? MachineLearningGears_for_SpaceNav/centroiding_based_nav/models/photocentre_correction_module.py
?? MachineLearningGears_for_SpaceNav/centroiding_based_nav/models/prior_informed_centroiding_convnn.py
?? MachineLearningGears_for_SpaceNav/centroiding_based_nav/utils/
?? MachineLearningGears_for_SpaceNav/rot_state_learning/
?? MachineLearningGears_for_SpaceNav/scripts/centroiding_based_nav/pysr_cob_regressor_search.py
?? MachineLearningGears_for_SpaceNav/scripts/centroiding_based_nav/upgraded/
?? MachineLearningGears_for_SpaceNav/scripts/rot_state_learning/
?? configs/
?? tests/centroid_based_nav/test_classical_regression_scripts.py
?? tests/centroid_based_nav/upgraded/
?? tests/rot_state_learning/
?? tests/utils/test_mlflow_server_script.py
```

- [x] Classified all pre-existing changes as user-owned and excluded them from
  Stage 0B:
  - PTAF has modified editor, guidance, dataset, explainer, experimental-backend,
    packaging, and existing test files, plus untracked explainer/dataset reports,
    examples, implementations, and tests;
  - the model owner has only `ml-based-centroiding.code-workspace` modified;
  - deployment has modified `.gitignore`, `CLAUDE.md`, `README.md`, `TODO`, MATLAB
    programs, one legacy export prototype, auxiliary/inference headers, three removed
    historical template-plan files, and untracked CI, design, FiLM, and inference-test
    files; and
  - the excluded legacy repository has broad modified and untracked centroiding,
    regression, dataset, experiment, script, configuration, and test work. It was
    inspected read-only and remains outside every implementation stage.
- [x] Reserved future change ownership explicitly: Stage 1 may touch only the planned
  PTAF export/CLI/TensorRT paths; Stage 2 may touch only the model-owner loader,
  export/parity, and documentation paths; Stage 3 may touch only deployment package,
  converter, build, focused test, and directly supporting documentation paths.
- [x] Classified legacy-name matches:
  - PTAF has no `.model_checkpoint`, `model_checkpoint/`, `matlab_models`,
    `onnx_models`, or `pytorch_models` match;
  - model-owner `benchmark_onnx_models.py` matches are command/module names, while
    `CLAUDE.md` documents a model-owner `scripts/onnx_models/` layout to reassess with
    Stage 2 rather than rewrite from deployment;
  - deployment matches in `SaveOnnxWeights.py` and `TestOnnxAccuracy.py` are protected
    Stage 5 legacy prototypes; `convert_onnx_models_to_ort` is an ONNX Runtime command
    name; remaining matches are this plan's policy text; and
  - the excluded legacy repository retains an active absolute
    `.model_checkpoints/matlab_models` MATLAB path plus benchmark module-name matches.
    No change was made outside deployment.
- [x] Confirmed active example manifests contain repo-relative artifact paths and no
  `/home` or `/media` default. `centroiding_traced_sample`, YOLOv7, and plain
  centroiding each loaded and ran once from both the repository root and `/tmp`.
  `template.ptafmodel` is deliberately non-runnable, while the untracked FiLM
  manifest and its unavailable external artifact remain excluded user work.
- [x] Narrowed the deployment ignore policy so generated contents under `models/`
  remain ignored while `models/README.md` records the canonical `matlab`, `onnx`,
  `pytorch`, and `tensorrt` layout. Representative `.onnx`, `.pth`, `.engine`,
  `.engine.json`, and `.mat` paths remained ignored.
- [x] Identified existing Python caches, egg metadata, example/Python/ROS build trees,
  ROS logs, and test caches. None was deleted because Stage 0B did not receive
  destructive-cleanup authorization, and none is included in the review batch.

### 2026-09-06 — Stage 1 generic export review batch

Historical report preserved from the pending tracker changes. The test counts,
wheel checks, and remote PR observations below were not rerun or refreshed during
the 2026-09-12 deployment consolidation; they describe the earlier candidate.

- [x] Created the isolated `feature/model-export-cli` worktree from PTAF revision
  `b307d8491eb41c777401b862831776553fd9c665`, which includes the prerequisite
  test-suite bugfix branch without modifying the original dirty PTAF checkout.
- [x] Added immutable export request, input, and loader-bundle contracts; direct
  TorchScript export; explicitly selected state-dictionary reconstruction; ONNX
  checker and ONNX Runtime validation; provenance reports; and documented CLI exit
  statuses.
- [x] Made ONNX/report publication collision-safe when overwrite is disabled and
  rollback-safe when overwrite is enabled. Regression tests cover a destination
  created during export and failures at both pair-publication steps.
- [x] Added `ptaf model export-onnx`, its installed entry point, runnable CLI/API
  examples, and the design-only `.ptafbundle` security/versioning contract.
- [x] Added TensorRT builder optimization levels `0..5` through the Python API and
  capability-probed `trtexec` flag while retaining the former seven positional
  `TRTengineExporterConfig` fields and existing workspace/profile behavior.
- [x] Completed an independent review, fixed its three blocking findings, and passed
  the follow-up review with no remaining critical, important, or minor findings and
  no approved API, ownership, or package-identity change.
- [x] Passed the focused export, CLI, and TensorRT matrix with `110 passed`; strict
  flake8 reported `0`, Ruff reported no findings, isolated mypy reported no issues,
  the new source stayed within the 100-column soft limit, and configured complexity
  checks reported no findings.
- [x] Passed the complete default suite in `autoforgeV2` with `733 passed, 72 skipped`
  and in the older `autoforge` environment with `731 passed, 74 skipped`; the latter
  has two expected Torch-version skips for unavailable dynamo export.
- [x] Built `pytorchautoforge-0.6.1.dev20-py3-none-any.whl`, installed it without
  dependencies in a fresh environment, confirmed both help commands, inspected the
  wheel for the export/CLI packages and entry-point metadata, and passed the public
  export module doctest. The disposable evidence directory is
  `/tmp/ptaf-stage1-final.EfxDKP`.
- [x] Audited prerequisite PR `pyTorchAutoForge#46` at
  `b307d8491eb41c777401b862831776553fd9c665`. Its only manual workflow run remains
  red: four unchanged dataset tests require an unset `DATASETS` path, and the
  repository-wide lint job reports 23 existing `F821` findings. The PR is mergeable,
  but its workflow does not target `develop`, so no automatic PR status is attached.
- [x] Recorded the one unresolved PR review thread separately from Stage 1:
  `StartMLflowUI` in `examples/example_mlflow_optuna_cifar10_demo.py` needs an
  explicit `subprocess.Popen` return annotation before the prerequisite PR merges.
- [ ] The Sphinx site was not rebuilt because the existing `autoforgeV2` environment
  does not include Sphinx. Runnable documentation examples were covered by doctest,
  source tests, and the clean installed CLI checks.
- [x] Staged exactly the eleven Stage 1 PTAF source, test, documentation, and
  packaging paths. The deployment tracker remains an unstaged change in its owning
  repository, and the prerequisite PR finding, model-owner Stage 2 work, generated
  artifacts, original dirty PTAF checkout, and all unrelated deployment work remain
  excluded.

### 2026-09-12 — Deployment consolidation and owner-state reconciliation

- [x] Inspect the owner worktree at
  `/home/peterc/devDir/ML-repos/.worktrees/pyTorchAutoForge-model-export`, branch
  `feature/model-export-cli`, HEAD `b307d8491eb41c777401b862831776553fd9c665`.
  The export implementation is staged and uncommitted. Its index contains twelve
  paths, including AGENTS.md alongside the eleven implementation-related paths.
  No unstaged owner diff was present. Leave the owner worktree and index unchanged.
- [x] Confirm the export contracts/module, CLI entry point, TensorRT optimization
  controls, tests, and bundle-design document exist in that candidate. This is
  source-presence evidence; it does not renew the historical behavioral or review
  claims. Return the overall Stage 1 checkbox to pending owner acceptance.
- [x] Assess the legacy ExportPytorchToONNX.py path migration. Its relative root
  resolves to this repository's models directory, consistent with the approved
  layout. Preserve the pending script unchanged and exclude it from this batch.
- [x] Record why that script is not accepted as a supported export path: it uses a
  hard-coded NeuralCOB import and a locally absent thoughtful-shark checkpoint.
  It passes a suffix-less output path while expecting a sibling .onnx file; the
  current main PTAF checkout treats that argument as a directory. Do not execute
  this script or replace its workflow during deployment consolidation.
- [x] Keep Stage 5 relocation pending until an owner-approved replacement covers
  the useful comparison and MATLAB-output behavior. Neither this legacy script
  nor completion of the export project is required by the image-only demo.
- [ ] Complete the owner review and validation before accepting Stage 1 or starting
  its dependent model-export stages. This deployment batch grants no authority to
  edit, stage, or commit in the owner repository.

Validation for this documentation batch consists of local source/path inspection,
owner status and index inspection, and diff checks. No model export, inference,
TensorRT build, benchmark, remote PR refresh, or Python test suite was run.

### 2026-09-12 — Authorized owner permission fix and Stage 1 review

- [x] Review the persistent goal after explicit user authorization to proceed with
  fixes. Retain dependency order, one-batch review gates, unchanged public contracts,
  and separate commit/push authority. Perform this review locally without subagents.
- [x] Confirm the prior single-writer overwrite documentation and checkpoint provenance
  verification are already staged in the owner export worktree.
- [x] Fix published ONNX/report permissions: new artifacts use exclusive same-directory
  creation with mode 0666 filtered by the OS umask. Replacement artifacts remain
  private during writing and receive the existing destination's read/write bits before
  publication, including read-only destinations. Do not copy executable/special bits,
  ownership, or extended ACLs. Never change the process umask in production code.
- [x] Clean up temporary report files when report serialization or writing fails.
- [x] Add real export regressions for umasks 0002, 0022, and 0077 and distinct existing
  ONNX/report permissions. Observe the permission defect before implementing the fix.
- [x] Validate the final candidate: 123 focused export/CLI/TensorRT tests pass; the
  Python 3.12 full suite reports 746 passed and 72 skipped. Ruff and isolated mypy on
  the affected export code pass. These are local results, not renewed remote CI proof.
- [x] Review the existing staged owner AGENTS.md for export ownership, naming,
  documentation, and review-gate alignment; preserve its contents unchanged.
- [x] Stage only the updated owner model_export.py, export test file, and runtime guide.
  The existing twelve-path owner batch remains staged and uncommitted, with no
  unstaged owner changes; cached whitespace checks pass.
- [ ] Accept/consolidate the owner Stage 1 batch before advancing. PR #46 is already
  merged; its annotation, dataset independence, active undefined-name defects, and
  develop CI trigger corrections remain a separate follow-up batch.
- [ ] Continue model-owner Stage 2 and deployment Stages 3–7 after their review gates.

The two deployment plan files already staged when this work began are preserved in
the index. This log addition is deliberately unstaged so it does not alter that
existing documentation review batch. No commit, push, or later batch was prepared.
