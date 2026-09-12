# Build, Packaging, And Template Sync

This repository is semantically aligned with the applicable mechanics from
`dev-tools/cpp_cuda_template_project` signed tag `v2.0.1` at
`1d87153b2d060bf03c2c9adcd1df6c6d4f40ea09`. The target remains authoritative
for inference APIs, ONNX Runtime/TensorRT dependencies, wrappers, package
identity, ROS 2 behavior, and permanent tests. The upgrade is a semantic port,
not a donor-tree overlay.

The earlier `v1.12.2` transition and its target-specific patch remain below as
historical provenance for the protected staged batch; they are not the current
alignment boundary.

## Donor Provenance

- Current `v2.0.1` annotated tag object:
  `e4cde034d5af7e1d5972f8cbfed71154fc476578`.
- Current `v2.0.1` signed tag commit:
  `1d87153b2d060bf03c2c9adcd1df6c6d4f40ea09`.
- Reviewed post-tag donor instruction commit:
  `12041acb19433dfe1b98b34203f4721b5a568797`. It changes agent-process
  guidance only and does not extend the production alignment boundary.
- Previous `v2.0.0` signed tag commit:
  `e42ddca9256b31010f47c6cb8074d9debaa7d5f5`.
- Historical `v1.12.2` annotated tag object:
  `1f60cebe98557bbba73a978f5a1243eb3c21f533`.
- Historical `v1.12.2` signed tag commit:
  `490de59d9fc5ec09ab4d57a9dbf5d6e238c7209a`.
- Approved gtwrap revision:
  `bf9f78617830bfa88c70d81a1a791c0a0c90b548` from `origin/master`.
- Exact post-tag donor snapshot:
  `doc/development/cpp_cuda_template_v1_12_2_dirty.patch`.
- Snapshot SHA-256:
  `1e251f34b353b393f9a3463565fb22ba553084cb49403513c133147b7a8db041`.
- Stable patch ID:
  `4645c110f7a48da523575f95faac5cb1500a87ce`.

The archived patch contains every tracked donor delta plus the untracked
`cmake/StagePackageVersion.cmake.in` as a full-index Git patch. Donor changes to
`VerifyTemplateProject*` scripts are retained only as provenance in that
archive: they are neither copied into this repository nor registered with
CTest.

The v1.12.2 follow-up decisions are:

| Donor area | Decision | Target treatment |
|---|---|---|
| Safe build cleanup | Adopt | Require a conventional in-repository path and exact `CMAKE_HOME_DIRECTORY` ownership before deletion. |
| Prepared-checkout source packaging | Adopt | Synchronize source `VERSION` before configuration, preserve caller hooks, and exclude the exact active build tree. |
| Python `FULL_VERSION` | Adopt | Use one Git-derived version for CPack and wheel metadata. |
| Split wrapper modules | Adapt | Preserve target interface files, package name, ORT linkage, and target-specific RPATH. |
| Wrapper checkout maintenance | Adopt | Default update/init operations to off, prohibit automatic submodule creation, and require explicit maintenance authorization. |
| Container launcher | Adapt | Retain ownership-safe Docker/Podman, VS Code attachment, MATLAB mount, and CUDA behavior under this repository identity. |
| CUDA/PTX | Retain | Keep the ordinary CUDA and embedded-PTX example sources; do not add OptiX. |
| CI | Adapt | Separate C++, docs, CUDA/TensorRT, and ROS jobs using target acceptance commands. |
| Donor conformance tests | Skip | Generic template behavior stays tested only in the donor repository. |
| Donor product identity and reports | Skip | Keep `autoforge_deploy`, `ptafdeploy`, and this repository's own tracking record. |

## Build And Dependency Rules

- CMake 3.21+ and C++20 are required.
- ONNX Runtime is required; discovery precedence is explicit
  `onnxruntime_DIR`, CMake `ONNXRUNTIME_ROOT`, environment
  `ONNXRUNTIME_ROOT`, then ordinary config-package discovery.
- Canonical options are `autoforge_deploy_METADATA_ONLY`,
  `autoforge_deploy_ENABLE_CUDA`, and
  `autoforge_deploy_ENABLE_TENSORRT`. Top-level legacy cache aliases
  `PROJECT_METADATA_ONLY`, `ENABLE_CUDA`, and `ENABLE_TENSORRT` are consumed on
  every configure for compatibility; nested consumers must use the canonical
  names.
- TensorRT remains optional and is not discovered when
  `autoforge_deploy_ENABLE_TENSORRT=OFF`. Enabling TensorRT does not enable the
  project CUDA language or its PTX examples; TensorRT independently discovers
  the CUDA runtime required by its imported targets.
- TensorRT discovery accepts `TensorRT_ROOT` and `TENSORRT_ROOT`, including
  x86_64 and aarch64 SDK archive layouts. Installed quiet package lookups
  remain quiet when the SDK is unavailable.
- CUDA is optional. Dedicated `.ptx.cu` sources are compiled and embedded, not
  compiled as ordinary CUDA translation units.
- CUDA CI always enables TensorRT in the same build. It is manual and guarded
  by `CI_USE_SELF_HOSTED=true` because both SDKs and a suitable GPU are runner
  responsibilities.
- OptiX support is absent.
- Source-tree `VERSION` writes are opt-in.

`build_lib.sh` resolves relative build and wrapper paths against the helper's
own checkout, independent of the caller's working directory. Its `--clean`
contract remains deliberately narrower than CMake configuration: only `build`,
`build*`, or `out/*` paths below that checkout are accepted. If the directory
exists, its cache must name the exact checkout as `CMAKE_HOME_DIRECTORY`; a
cache owned by the donor or another clone is refused and preserved.

## Wrapper Ownership And Delivery

Wrapper resolution may use an explicit root, local checkout, installed gtwrap
package, or declared submodule. Resolution is read-only by default. Use
`./build_lib.sh -p --wrap-update` to intentionally update a resolved checkout,
or `--wrap-submodule-init` to intentionally initialize a declared fallback.
Direct CMake callers must pair `GTWRAP_SYNC_TO_MASTER=ON` with
`GTWRAP_MAINTENANCE_UPDATE=ON`. Configuration never adds an undeclared
submodule; repository creation remains an explicit maintenance operation
outside ordinary CMake configuration.

The Python wheel contains the extension and the direct project-owned shared
runtime `libautoforge_deploy`; build-only `_wrapper_build.py` metadata remains
under `<build>/python` and is excluded from the wheel.
ONNX Runtime is intentionally external and is not copied into the wheel. The
extension uses a loader-relative path for the co-located project library while
the build tree retains its explicit ONNX Runtime lookup path. This keeps local
tests functional without claiming that the wheel vendors ORT.

MATLAB discovery accepts an explicit CMake `Matlab_ROOT_DIR` first and then the
`MATLAB_ROOT_DIR` environment variable used by the container launcher. Both
wrappers expose the generic inference/model facade and wrapper-safe values, not
backend handles or byte storage.

MATLAB toolbox outputs install below the selected CMake prefix. When an
installed gtwrap package provides `include/gtwrap/matlab.h`, configuration
copies that header into a build-owned `wrap/matlab.h` compatibility path
without modifying the package or source checkout.

## Version And Package Contract

CMake writes the exact five-field VERSION payload below, with one terminating
newline and no blank sixth line:

```text
Project version: 1.2.3
Project version core: 1.2.3
Project version prerelease: stale
Project version metadata: source
Full version: 1.2.3-stale+source
```

Binary packages install that exact build-tree file. Before configuring a source
package, run `./generate_version.sh`; CPack then packages the synchronized
source `VERSION` unchanged without claiming a public extension hook. Wheel
metadata uses the same `FULL_VERSION` and lets the Python build backend
normalize it to PEP 440 when necessary.

Source packaging appends anchored exclusions for generated build/install,
code-generation, MEX, cache, and Python native artifacts while retaining
caller-owned CPack exclusions and the target-specific development-plan policy.

Generated install/log trees, Git metadata, Python caches, generated wrapper
bootstrap files, and the exact active binary tree are excluded from source
packages. Foreign nested CMake caches are not scanned or treated as
project-owned.

## CI Branch Policy

- Native C++/Python, documentation, and ROS 2 run automatically for pushes to
  `main` and pull requests targeting `main` or `develop`.
- `develop` push validation is manual through `workflow_dispatch`.
- Documentation is built separately for `main`, pull requests, and manual
  dispatches. Pages deployment is deliberately disabled until repository
  publication settings and policy are reviewed.
- CUDA/TensorRT is manual-only and disabled unless the repository variable
  `CI_USE_SELF_HOSTED` is `true`.
- Permanent CI/CTest coverage exercises target inference, wrappers, programs,
  and ROS behavior. It does not inspect placeholder symbols or re-run donor
  template conformance matrices.

## Acceptance Checks

Template-derived build mechanics are accepted with explicit fresh integration
gates:

- metadata-only configure without ORT;
- default, Debug, RelWithDebInfo, and explicit-architecture CUDA+TensorRT
  builds;
- installed consumer linked only to
  `autoforge_deploy::autoforge_deploy`;
- Doxygen HTML/XML, Python/MATLAB wrapper generation, wheel installation and
  real facade inference;
- CPack binary/source archives and an extracted source-package build;
- ROS 2 build and real standalone/composed inference tests;
- shell/YAML/JSON syntax, CTest-inventory, OptiX-absence, and staged-diff
  audits.

These checks do not become generic permanent tests in this repository.
Permanent tests must exercise `torchAutoForge-deploy` inference, wrappers,
programs, future project-owned CUDA kernels, or ROS behavior.
