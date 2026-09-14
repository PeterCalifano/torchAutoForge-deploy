# Experimental Mobile Deployment Pipeline Design

Status: approved design direction; implementation has not started

Date: 2026-08-26; expanded 2026-08-30

## Summary

Add an Android-first deployment and validation pipeline around the existing
`ptafdeploy::inference::CModelFacade`. Package the C++ runtime behind a thin JNI/Kotlin
library, keep ONNX Runtime as the required inference backend, and use the original
OnePlus Nord as the first physical qualification device.

The portable baseline is ONNX Runtime Mobile with CPU and XNNPACK execution. A
Qualcomm QNN/HTP lane is optional and must pass an explicit compatibility probe on the
Nord before it becomes an implementation or support claim. NNAPI may be retained as a
legacy comparison experiment, but it is not a new default because Android deprecated
NNAPI in Android 15. iOS is a later, optional lane using the same C++ facade with
XNNPACK/CoreML and an Objective-C++/Swift bridge.

Performance and power are product requirements, not after-the-fact diagnostics. The
pipeline must report cold start, model load, warm latency distributions, throughput,
memory, binary/model size, sustained thermal behavior, and energy per inference when
the physical device exposes a trustworthy energy source. Emulator runs establish
correctness and packaging only; they never establish mobile performance or energy.

## Decision Record

### Decision

Build the existing backend-neutral `CModelFacade` with the Android NDK and package it
behind a small JNI/Kotlin API in an AAR. Establish correctness with the full ONNX
Runtime Android package, then qualify CPU and XNNPACK on the OnePlus Nord. Introduce
ORT-format models, a reduced runtime, QNN/HTP, or a different mobile runtime only at
explicit evidence gates.

This decision separates four concerns that can otherwise be conflated:

1. **API ownership:** where manifests, task contracts, validation, and lifecycle live
2. **Inference runtime:** ORT, LiteRT, ExecuTorch, or a vendor runtime
3. **Delivery format:** AAR composition, model format, and model distribution
4. **Qualification:** host, emulator, physical-device, and optional device-farm evidence

Changing one concern does not silently change the other three. For example, a future
LiteRT backend could still sit behind the native role-level facade, and Docker emulator
testing remains required even when physical-device benchmarking is added.

### Evaluation criteria

The following criteria order the alternatives. The assessment is qualitative until
the physical campaign supplies measurements; labels such as "lower overhead" describe
the expected architecture, not a benchmark result.

| Criterion | Priority | What must be demonstrated |
|---|---|---|
| Contract and result parity | Critical | Identical manifest interpretation, tensor validation, task semantics, errors, and checked outputs |
| Sustained latency | Critical | Cold, first, warm percentiles, and thermal drift on the original Nord |
| Energy efficiency | Critical | Energy per correct inference under controlled conditions, or an explicit power-unvalidated result |
| Fallback truthfulness | Critical | Requested acceleration is used for the claimed graph, or session creation fails |
| Delivery risk | High | A working pinned AAR/APK path without redesigning unrelated desktop consumers |
| Model portability | High | Reproducible conversion, operator coverage, quantization accuracy, and debuggable artifacts |
| Footprint | High | APK/AAR, native runtime, model, private-storage duplication, and peak memory |
| Cross-platform reuse | High | Shared C++ behavior across Android, desktop, wrappers, ROS, and optional iOS |
| Testability | High | Deterministic host, `x86_64` emulator, `arm64-v8a` device, and failure-path tests |
| Maintenance and supply chain | High | Pinned tools, licenses, security updates, ABI compatibility, and one owner for each policy |
| Reversibility | Medium | A failed experiment can be removed without breaking the public facade or artifact schema |

No alternative wins solely on microbenchmark latency. Correctness, sustained behavior,
energy, footprint, integration cost, and maintenance are reviewed together. Unknown
Nord compatibility is recorded as unknown rather than scored optimistically.

### API-ownership alternatives

| Design | Semantic reuse | Data-path potential | Delivery and maintenance cost | Decision |
|---|---|---|---|---|
| **A. Existing C++ facade plus thin JNI** | Reuses manifests, value types, task adapters, validation, and backend policy | One application-owned JNI boundary; direct buffers can avoid bulk copies | Moderate Android work; lowest long-term policy duplication | **Chosen** |
| **B. Direct ORT Java/Kotlin application path** | Reuses ORT only; duplicates the product facade and task behavior | Raw ORT execution can be similar, but Java-side adapters or a second native bridge can add work | Fastest demo; Android-specific product logic and two behavior suites | Diagnostic reference only |
| **C. Android-specific native ORT fork** | Reuses some C++ helpers but bypasses `CModelFacade` | Similar ceiling to A and may initially link less code | Creates a second native lifecycle, config, and fallback implementation | Rejected |
| **D. General multi-runtime core abstraction first** | Could eventually host ORT, LiteRT, and ExecuTorch uniformly | Potentially strong after a larger redesign | Changes core ownership, backend dispatch, wrappers, TensorRT, ROS, and tests before Android evidence exists | Deferred separate project |

#### A. Existing C++ facade plus thin JNI: chosen

The main advantage is behavioral ownership. `CModelFacade`, wrapper-safe task values,
schema-driven task adapters, model configuration, and backend diagnostics remain the
authoritative implementation. Android adds lifecycle and buffer transport, not another
interpretation of the model contract. The same native path can later be packaged for
iOS with a different language bridge.

JNI still has costs: native failures are harder to diagnose, ABI packaging is more
complex, and each crossing has fixed overhead. Those costs are bounded by using one
coarse inference call, deterministic native ownership, and direct buffers. JNI is not
assumed to dominate; profiling must attribute preprocessing, copies, inference, and
postprocessing before a lower-level transport is added.

The main implementation risk is accidental import of desktop-only dependencies or
assumptions into the mobile target. Android target composition therefore has to be
proven independently rather than forcing Android conditionals throughout the desktop
library.

#### B. Direct ORT Java/Kotlin path: fastest demonstrator

ORT's Android Java API already enters a native runtime through ORT's JNI layer, so this
option removes the project-owned JNI bridge rather than making inference purely Java.
It is attractive for proving that a model and upstream ORT package work on Android,
and it provides a useful reference when diagnosing a project bridge failure.

As a product path, it would require Kotlin implementations of manifest resolution,
task adapters, validation, session reload, errors, and provider policy. Keeping those
operations in C++ would add another project JNI layer and remove most of the simplicity.
Raw inference performance may be close to design A when the same ORT provider and
buffers are used, but Java array conversion or duplicated preprocessing can worsen
end-to-end latency, allocation pressure, and energy. Only measurement can establish
the difference.

Use this design for a small, disposable upstream sanity application if needed. Do not
promote it unless maintaining an Android-specific semantic implementation is explicitly
accepted.

#### C. Android-specific native ORT fork: apparent simplicity, durable divergence

This design would call ORT C/C++ directly from an Android-only JNI library and copy
only selected helpers from the existing facade. It could produce a narrow first binary,
but it has nearly the same native build and JNI work as design A while creating a new
owner for configuration, task results, fallback, and lifecycle. Any initial footprint
advantage should instead be pursued by target-level linking and dependency slimming.

Because it offers no distinct accelerator or transport advantage, this design is
rejected.

#### D. General multi-runtime abstraction first: strategically useful, wrong sequence

A private execution-session interface could eventually remove fixed backend knowledge
from the public manager and allow ORT, LiteRT, ExecuTorch, or future runtimes to plug in
behind one facade. That may be the clean long-term architecture if measured mobile
results justify multiple runtimes.

Doing it first would expand the experiment into a core redesign involving desktop ORT,
standalone TensorRT, wrappers, programs, ROS, error contracts, reload transactionality,
and installed consumers. It would delay the first physical evidence and make Android
failures difficult to distinguish from redesign regressions. This option requires a
separate approved design and implementation plan after the ORT Android baseline.

### Relative implementation size and code shape

The size bands below describe ownership and verification surface, not lines of code or
calendar estimates:

| Band | Meaning |
|---|---|
| Small | Isolated feasibility application or comparison with no supported reusable contract |
| Medium | New supported platform deliverable using the existing runtime and semantic contract |
| Large | New runtime or duplicated product behavior with its own artifacts, parity, and support matrix |
| Very large | Cross-platform core redesign affecting multiple backends and public integrations |

| API design | Initial size | Durable size | Principal change surface |
|---|---|---|---|
| A. Native facade plus JNI | Medium | Medium | Android Gradle/CMake modules, Kotlin API, JNI ownership, two ABIs, instrumentation, container and device tests; focused core build seams |
| B. Direct ORT Java/Kotlin | Small as a demo | Large as a product | Android app plus Kotlin copies of config, validation, task adapters, errors, provider policy, and parity tests |
| C. Android native ORT fork | Medium | Large | Android JNI/native runtime plus duplicated native config, lifecycle, task, and fallback behavior |
| D. Multi-runtime abstraction first | Very large | Very large | `CInferenceManager`, every concrete backend, facade/reload behavior, wrappers, programs, ROS, installed consumers, and their tests |

The following sketches illustrate the amount and location of glue. They are deliberately
incomplete pseudocode; they define responsibility, not final names or implementation.

#### Design A code shape

The public Android surface remains small and resource-owned:

```kotlin
PtafSession.open(modelConfigPath, runtimePolicy).use { session ->
    val metadata = session.contract()
    val output = session.infer(inputDirectBuffer, inputShape)
    val diagnostics = session.diagnostics()
}
```

The bridge delegates product behavior to the existing facade:

```cpp
// Illustrative JNI ownership shape, not a proposed final signature.
jlong NativeOpen(std::string config_path)
{
    auto facade = std::make_unique<ptafdeploy::inference::CModelFacade>();
    facade->LoadModelConfig(config_path);
    return session_registry.Insert(std::move(facade));
}

NativeOutput NativeInfer(jlong handle, DirectBuffer input, Shape shape)
{
    auto& facade = session_registry.Get(handle);
    return Encode(facade.InferSingleFloatTensor(Decode(input, shape)));
}
```

The first correct implementation may materialize the current wrapper-safe
`SFloatTensor`. If profiling proves that the copy is material, a native-only view path can
reuse the non-owning `STensorView`/`CInferenceManager::Infer` behavior without exposing
raw pointers to Kotlin, Python, or MATLAB. Current ownership to reuse is in:

- `src/inference/model_facade.h/.cpp`
- `src/inference/inference_manager.h/.cpp`
- `src/inference/inference_common.h`
- `src/inference/task_value_types.h/.cpp`
- `src/inference/task_adapters.h/.cpp`

#### Design B code shape

The inference call itself is compact because ORT supplies Android Java bindings:

```kotlin
val environment = OrtEnvironment.getEnvironment()
environment.createSession(modelPath, sessionOptions).use { session ->
    OnnxTensor.createTensor(environment, inputFloatBuffer, shape).use { input ->
        session.run(mapOf(inputName to input)).use { result ->
            consume(result)
        }
    }
}
```

That snippet is a valid model-execution demo, but not a substitute for the project API.
A supported product path also needs Android-owned equivalents of approximately this
responsibility graph:

```text
PtafModelParser.kt -> RuntimePolicy.kt -> OrtSessionFactory.kt
         |                    |                  |
         +-> TensorValidator.kt                  +-> FallbackDiagnostics.kt
         +-> TaskAdapterRegistry.kt -> wrapper-safe task results
```

Every change to the C++ manifest, task schema, fallback rule, or error behavior then
requires a coordinated Kotlin change and cross-language parity test. That duplicated
graph is why the demo is small while the supported alternative is large.

#### Design C code shape

An Android-only native fork is superficially close to the existing backend:

```cpp
Ort::Env environment{ORT_LOGGING_LEVEL_WARNING, "ptafdeploy-android"};
Ort::SessionOptions options;
RegisterRequestedProvider(options, android_policy);
Ort::Session session{environment, model_path.c_str(), options};
auto outputs = session.Run(run_options, input_names, input_values,
                           output_names);
```

The short call omits the difficult parts already owned elsewhere: manifest parsing,
artifact dispatch, tensor ordering and cardinality, task result decoding, transactional
reload, provider truth, and stable errors. Rebuilding those around this snippet produces
nearly design A's implementation size with an additional long-term fork.

#### Design D code shape

The multi-runtime design begins with an ownership change in the core, not an Android
module:

```cpp
class IExecutionSession
{
  public:
    virtual ~IExecutionSession() = default;
    virtual const SModelMetadata& GetMetadata() const = 0;
    virtual std::vector<STensorBuffer>
    Infer(std::span<const STensorView> inputs) const = 0;
};

class OrtExecutionSession final : public IExecutionSession { /* ... */ };
class TensorRtExecutionSession final : public IExecutionSession { /* ... */ };
class LiteRtExecutionSession final : public IExecutionSession { /* ... */ };
```

This would replace or hide the current fixed `std::variant` in
`src/inference/inference_manager.h`, define capability and error contracts, and prove
transactional construction for every backend. The interface is small; the migration
and regression matrix make the alternative very large.

### Inference-runtime and acceleration alternatives

| Runtime path | Model/runtime change | Main opportunity | Principal cost or uncertainty | Nord decision |
|---|---|---|---|---|
| **Full ORT, CPU/XNNPACK** | Existing ONNX plus pinned full Android runtime | Lowest conversion risk and fastest honest baseline | Largest initial runtime and broad unused operator set | **Start here** |
| **Reduced ORT, CPU/XNNPACK** | ORT-format model plus model-set-specific runtime | Lower binary footprint; same facade and providers | Runtime must be regenerated and tested whenever required operators/types change | Promote after baseline |
| **ORT QNN/HTP** | Fixed-shape calibrated QDQ model, QNN-enabled ORT, vendor libraries | Potential NPU latency/energy improvement while retaining ONNX/ORT ownership | Old-device SDK/firmware support, operator coverage, quantization loss, redistribution, and CPU fallback | Gated physical probe |
| **Direct Qualcomm QNN** | Vendor graph/toolchain and a new native backend | Maximum vendor control and possibly fewer framework layers | Highest lock-in, new backend semantics, no useful emulator qualification, and separate packaging/support burden | Reject unless ORT QNN has a measured blocker |
| **LiteRT `CompiledModel`** | Convert and validate `.tflite`; add LiteRT backend and packages | Android-focused CPU/GPU/NPU path with Kotlin and C++ APIs | Second model format, conversion/quantization parity, operator coverage, and dual-runtime maintenance | Future competitive spike |
| **ExecuTorch** | Export/lower PyTorch to `.pte`; add ExecuTorch runtime/backend | PyTorch-native export with XNNPACK, Vulkan, and Qualcomm paths | Bypasses the established ONNX pipeline; export compatibility and backend-specific lowering become new contracts | Future competitive spike |
| **ORT NNAPI** | Existing ORT path with system NNAPI drivers | Low-effort historical comparison on compatible OS images | Deprecated in Android 15, variable driver behavior, and no durable support direction | Legacy, default-off |

#### Full ORT before reduced ORT

The published full package accepts ONNX models and minimizes the number of variables
during first integration. Its larger footprint is knowingly accepted for the baseline.
After the complete model set is known, ORT-format conversion supplies the required
operator/type configuration for a custom runtime. The reduced runtime can materially
shrink the package, but it couples a runtime build to the exact promoted model set.

The pipeline therefore retains both lanes: full ORT is the diagnostic compatibility
oracle, while reduced ORT is the candidate deliverable. A model update that changes
operators or types must regenerate the configuration, rebuild both Android ABIs, and
rerun parity and packaging acceptance. XNNPACK requires an extended minimal build, so
size reduction must not silently remove the selected provider.

#### ORT QNN versus direct QNN

ORT QNN preserves the existing C++ and ONNX boundaries and exposes strict CPU-fallback
control, provider profiling, performance modes, and context caching. Its HTP path
requires a quantized model, representative calibration data, fixed shapes, and supported
operators. The current public ORT documentation does not list the Snapdragon 765G among
its tested QNN SoCs, so Hexagon 696 presence is not a compatibility claim.

A direct QNN backend could expose vendor buffers and controls sooner than ORT does, but
would make this project own a Qualcomm-specific graph, lifecycle, error-recovery, SDK,
and redistribution contract. It is considered only if a successful Nord probe shows a
material, measured ORT QNN limitation that cannot be corrected upstream or configured
away.

#### LiteRT

Current LiteRT provides a modern `CompiledModel` API for Android with Kotlin and C++
integration and CPU/GPU/NPU acceleration paths. That makes it a credible competitor,
not merely a TensorFlow-only historical option. It may be attractive if its accelerator
stack supports the Nord and produces a better sustained latency-energy result.

For this repository, however, adopting LiteRT requires a reproducible `.tflite` export,
new operator and quantization qualification, a new backend implementation, and task-level
parity with the ONNX artifact. ONNX Runtime remains the required project backend, so
LiteRT would initially increase rather than replace maintenance. It advances from a
spike only if it is Pareto-superior by a predeclared meaningful margin and its model
conversion remains reproducible.

#### ExecuTorch

ExecuTorch supplies Android AAR and native C++ integration, and offers XNNPACK, Vulkan,
and Qualcomm backends. It aligns naturally with source models still owned in PyTorch
and could avoid an ONNX conversion for those models.

The cost is a separate `.pte` export/lowering pipeline tied to PyTorch-export support,
new runtime value and memory contracts, and reduced applicability to models delivered
only as ONNX. Published Qualcomm guidance is verified primarily on newer Snapdragon
families than the 765G, so the Nord again requires a real compatibility probe. As with
LiteRT, it remains a controlled same-model comparison until evidence justifies product
ownership.

### Runtime change surface and integration sketches

| Runtime path | Relative incremental size after design A | New owned pipeline or support surface |
|---|---|---|
| Full ORT CPU/XNNPACK | Medium baseline | Pinned Android ORT, provider configuration, AAR packaging, two ABIs, parity and device benchmarking |
| Reduced ORT CPU/XNNPACK | Medium increment | ORT conversion, required-operator/type config, custom runtime build, model-set/runtime lockstep |
| ORT QNN/HTP | Large | QDQ calibration, fixed shapes, QNN SDK/runtime packaging, strict assignment, context cache, profiling, SSR recovery |
| Direct QNN | Very large | New backend, vendor graph/toolchain, native buffers, SDK licensing/redistribution, device-only qualification |
| LiteRT | Large | `.tflite` conversion, LiteRT backend, accelerator selection, Android packages, dual-runtime parity and support |
| ExecuTorch | Large | `.pte` export/lowering, ExecuTorch backend and packages, backend delegation, dual-runtime parity and support |
| ORT NNAPI | Small experiment | Provider flag and device comparison, with no long-term promotion path |

The full-to-reduced ORT sequence reuses one runtime API but adds a reproducible build
pipeline. The rough command shape is below; `ORT_RELEASE_TAG` comes from the checked
dependency lock rather than the ambient environment:

```bash
# Establish the full Android package first; exact pinned arguments belong in the plan.
./build.sh --android --android_abi arm64-v8a --build_java --config MinSizeRel

# Later convert the promoted model set and generate required operator/type metadata.
python -m onnxruntime.tools.convert_onnx_models_to_ort \
  --enable_type_reduction path/to/promoted_models

# Build a pinned custom AAR from that checked configuration.
python tools/android_custom_build/build_custom_android_package.py \
  --onnxruntime_branch_or_tag "$ORT_RELEASE_TAG" \
  --include_ops_by_config path/to/required_operators_and_types.config \
  --build_settings path/to/android_build_settings.json \
  path/to/generated_work_root
```

QNN adds more than a provider name. The qualifying session must make fallback failure
observable and bind a verified HTP runtime:

```cpp
// Schematic ORT QNN qualification settings; provider API details are version-pinned.
session_options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
session_options.AddConfigEntry("ep.context_enable", "1");
AppendQnnProvider(session_options,
                  {.backend = "htp", .performance_mode = "sustained_high_performance"});
```

The surrounding work fixes shapes, produces a calibrated QDQ model, packages compatible
QNN libraries, checks complete graph assignment, compares cached and uncached load, and
recovers by recreating the session after an HTP subsystem restart. None of those steps
can be established by the `x86_64` emulator.

LiteRT can be integrated from Kotlin or native C++. A native backend would resemble:

```cpp
auto compiled_model = litert::CompiledModel::Create(environment, tflite_path, options);
auto input_buffers = compiled_model.CreateInputBuffers(signature_index);
auto output_buffers = compiled_model.CreateOutputBuffers(signature_index);
compiled_model.Run(signature_index, input_buffers, output_buffers);
```

The small runtime call sits after a large new responsibility: PyTorch/TensorFlow-to-
`.tflite` conversion, signatures, quantization, operator compatibility, task parity,
accelerator selection, packaging, and another backend adapter behind the project facade.

ExecuTorch's Android call is similarly compact:

```kotlin
val module = Module.load(ptePath)
val input = Tensor.fromBlob(inputValues, inputShape)
val outputs = module.forward(EValue.from(input))
```

Its larger work is upstream: make the source model compatible with `torch.export`,
lower it for XNNPACK/Vulkan/Qualcomm, create and inspect the `.pte`, map runtime values
to project tensors, and demonstrate the same task-level result and physical protocol as
ORT. These snippets explain why API brevity is not a reliable estimate of product size.

### Packaging and artifact alternatives

#### Runtime packaging

| Packaging | Benefits | Costs and failure modes | Decision |
|---|---|---|---|
| Self-contained `ptafdeploy` AAR | One consumer artifact, pinned native closure, deterministic offline installation | Larger artifact; must inspect licenses, duplicate libraries, symbols, and ABI/page alignment | Preferred experiment delivery |
| Version-locked bridge AAR plus ORT AAR | Clear upstream package separation and simpler ORT replacement | Gradle/native version skew, duplicate `.so` risk, and less control over consumer resolution | Acceptable fallback if one release manifest locks both |
| Consumer-supplied ORT runtime | Small project artifact and flexible app dependency choice | Unsupported ABI combinations and unbounded runtime/provider mismatch | Rejected for qualification |
| APK-only native integration | Fast reference application | No reusable library contract and hides consumer packaging problems | Test harness only |

The experiment first attempts a self-contained AAR containing the Kotlin API, project
JNI/core libraries, and exactly one pinned ORT native runtime for each ABI. If Gradle or
upstream packaging makes that composition brittle, ship a version-locked pair whose
release manifest and acceptance test prohibit independent version substitution. An APK
inspection gate rejects duplicate `libonnxruntime` or C++ runtime copies in either case.

#### Model distribution and format

| Choice | Startup/footprint behavior | Update and integrity behavior | Decision |
|---|---|---|---|
| Bundled ONNX asset | Simplest full-runtime baseline; APK and private-copy storage may both contain the model | App release controls the model; hash can be fixed at build time | Baseline fixture and first campaign |
| Bundled ORT-format asset | Works with reduced runtime and may improve load characteristics | Converter/runtime versions and operator config become a locked unit | Candidate deliverable |
| Downloaded versioned package | Smaller app update and independent model rollout | Requires signature/hash verification, atomic install, rollback, storage policy, and network lifecycle | Later product feature |
| Play Asset Delivery or dynamic feature | Store-managed large-asset delivery | Adds Play-specific packaging and test surface | Out of first experiment |

The first experiment copies a checked bundled model into private storage because the
existing facade accepts filesystem paths. It measures the APK-size and duplicated-
storage cost. A memory-mapped asset or downloaded package is introduced only in response
to measured startup, storage, or update requirements.

#### Tensor transport

| Transport | Advantages | Costs | Decision |
|---|---|---|---|
| Java/Kotlin primitive arrays | Simple API and easy tests | Conversion and allocation copies are likely for large tensors | Metadata and tiny diagnostic values only |
| Direct `ByteBuffer`/native buffer | Explicit capacity and dtype; supports a single coarse JNI call with fewer copies | Requires strict lifetime, position, alignment, and concurrency validation | **Initial tensor path** |
| `AHardwareBuffer`, camera surface, or provider I/O binding | Possible camera-to-accelerator copy reduction | Android/provider coupling, synchronization and lifetime complexity, limited emulator evidence | Profile-gated optimization |

Zero-copy is a measured optimization, not a design slogan. The baseline records bytes
copied and stage timings. A lower-level transport is considered only when data movement
is a material fraction of end-to-end latency or energy and the selected provider can
consume the buffer without an equally expensive hidden conversion.

### Qualification alternatives are complementary

| Lane | Strength | Blind spot | Role in the decision |
|---|---|---|---|
| Host native tests | Fast behavioral regression and golden generation | No Android loader, JNI, ABI, or device behavior | Required before Android packaging |
| Docker builder and KVM emulator | Reproducible SDK/NDK, APK installation, JNI/lifecycle/errors, `x86_64`, and 16 KB-page checks | Not the Nord CPU, firmware, accelerator, battery, or thermal system | Required correctness gate |
| Owned OnePlus Nord | Authoritative ABI, provider, sustained performance, power, and thermal evidence for the first target | One aged device does not represent the Android market | Required qualification gate |
| Hosted physical-device farm | Broader API/OEM compatibility and regression reach | Limited low-level power control and does not replace the owned target | Optional after the Nord lane is stable |

Docker and the physical phone are therefore not competing approaches. The emulator
shortens feedback for deterministic failures; the Nord answers the performance and
energy questions the emulator cannot answer.

### Final selection and reconsideration triggers

The initial selection is:

- native `CModelFacade` ownership behind one coarse JNI/Kotlin AAR API
- full ORT plus CPU/XNNPACK for the first correct physical baseline
- direct buffers before specialized Android zero-copy transports
- bundled checked models before dynamic model delivery
- a self-contained or strictly version-locked runtime package
- Docker/KVM emulator correctness plus separate Nord qualification
- reduced ORT only after the supported model set is known
- QNN, LiteRT, and ExecuTorch only through gated, same-device comparisons

Reconsider the selected architecture only when evidence shows at least one of the
following:

- JNI or native buffer transport is a material end-to-end bottleneck after profiling
- the native core cannot be made Android-portable without distorting desktop ownership
- ORT misses an approved sustained latency, energy, memory, or footprint budget
- another runtime demonstrates parity and a meaningful Pareto improvement on the Nord
- QNN requires a vendor-native capability unavailable through ORT and the benefit
  justifies the support burden
- model ownership moves away from ONNX strongly enough that maintaining ONNX and another
  export is no longer the cheaper contract

Any reconsideration is a new design decision. A promising single microbenchmark does
not authorize a second production runtime.

## Scope

### In scope

- Android `arm64-v8a` AAR and APK production from the existing C++20 facade
- Android `x86_64` artifacts for containerized emulator testing only
- ONNX and ORT-format model loading through ONNX Runtime
- CPU and XNNPACK execution-provider selection with explicit fallback semantics
- optional QNN/HTP feasibility and performance qualification on the OnePlus Nord
- a headless/reference benchmark application and instrumentation tests
- a Docker-based Android builder and KVM-accelerated emulator test lane
- numerical parity, latency, memory, footprint, thermal, and power reporting
- reproducible model preparation and reduced-operator ORT builds
- a documented optional iOS extension point

### Out of scope for the first experiment

- a production user interface or Play Store release
- background or always-on inference services
- Android TensorRT, CUDA, DLA, or PTX
- NNAPI as a default or supported long-term acceleration contract
- unqualified QNN claims based only on the presence of Hexagon hardware
- LiteRT, ExecuTorch, or direct QNN as an ungated prerequisite or first supported deliverable
- 32-bit Android ABIs
- iOS implementation or performance claims from this Linux host
- a general execution-session/type-erasure redesign
- committing generated models, AARs, APKs, ORT runtimes, or benchmark reports

## First Physical Target

The first target is the original OnePlus Nord, not a newer Nord variant.

| Property | Qualification value |
|---|---|
| Product | OnePlus Nord, original generation |
| SoC | Qualcomm Snapdragon 765G |
| CPU | 64-bit Kryo 475 family |
| GPU | Adreno 620 |
| AI hardware | Hexagon 696 with tensor acceleration |
| Memory | 8 or 12 GB LPDDR4X, discovered on the actual device |
| Battery | 4115 mAh nominal; present health/capacity must be measured |
| Android build | Discovered from the connected device; do not assume launch Android 10 |
| Runtime ABI | `arm64-v8a` |

The phone is intentionally an older mid-range thermal and power envelope. A pipeline
that is usable there provides a strong portability baseline, but its results must not
be generalized to newer Snapdragon devices.

The preflight report records at least:

- manufacturer, model, board, SoC, ABI, Android API level, build fingerprint, and
  security patch level
- total memory and current free storage
- battery level, health status, charge counter support, and charging state
- thermal API availability and initial thermal state/headroom
- supported ABIs, memory page size, and native-loader diagnostics
- ORT version, packaged execution providers, and resolved provider options

## Architecture

### Offline artifact flow

```text
trusted ONNX model + representative inputs + golden outputs
                         |
                         v
        validate shapes, operators, parity, and provenance
                         |
            +------------+-------------+
            |                          |
            v                          v
      FP32/FP16 candidate        calibrated INT8 QDQ candidate
            |                          |
            +------------+-------------+
                         |
                         v
              convert to ORT format
                         |
                         +--> required operators/types config
                         |
                         v
       pinned full or reduced ONNX Runtime Android build
                         |
                         v
       model package + runtime report + Android AAR/APK
```

The full published ORT Android package is used first to establish correctness and a
performance baseline. Operator/type reduction is introduced only after model
conversion produces a checked configuration. Converter and runtime use the same
pinned ORT release; `latest.release`, nightly packages, and an unpinned main branch
are not reproducible inputs.

Generated mobile artifacts live under an ignored output root. Each artifact report
records source and output SHA-256 hashes, ORT version, converter arguments, operator
configuration hash, ABI, API level, NDK version, compiler, build type, providers, and
model parity results.

### Runtime flow

```text
Kotlin application / instrumentation test
                |
                v
       Closeable Kotlin session
                |
                v
      narrow JNI ownership layer
                |
                v
        CModelFacade contract
                |
                v
       CInferenceManager -> ORT
                |
       +--------+--------+
       |        |        |
      CPU    XNNPACK   QNN/HTP (optional, qualified)
```

The mobile bridge always enters through `CModelFacade`; it does not call backend
classes directly. Role adapters stay above raw ORT execution. Provider-specific
configuration remains inside the ORT backend.

Android packaged assets are not assumed to be ordinary files. The first experiment
copies a versioned model package once into application-private storage, verifies its
hash, and loads it through the existing filesystem contract. Model load time is
reported separately from inference. An `AAssetManager`/memory-mapped load path is a
later optimization only if first-use latency, duplicated storage, or flash traffic is
material.

### Android library surface

The Kotlin-facing API remains deliberately small:

- create a session from a packaged manifest and runtime policy
- inspect model input/output metadata and resolved backend/provider diagnostics
- run typed dense tensors through direct buffers
- retrieve structured timing/fallback diagnostics
- close the native session deterministically

Every JNI entry validates the native handle, buffer directness, dtype, shape,
cardinality, byte count, and session state. No C++ exception crosses JNI. Native
errors become stable typed Java/Kotlin exceptions with actionable messages. A closed
session rejects later calls. The initial concurrency contract is one inference at a
time per session; parallel callers use separate sessions until a measured need
justifies shared-session synchronization.

## Runtime and Provider Policy

### CPU baseline

CPU execution is the correctness fallback and the first baseline for quantized CNNs.
Thread count and spinning are explicit benchmark dimensions. The implementation does
not assume that all eight logical cores are energy-optimal on the Nord's heterogeneous
CPU.

### XNNPACK baseline

XNNPACK is the preferred first candidate for unquantized mobile models. ORT's normal
intra-op pool and XNNPACK's internal pool must not compete accidentally. The resolved
configuration records both pool sizes and whether ORT intra-op spinning is disabled.
The benchmark sweeps a small bounded set of thread counts rather than hard-coding the
desktop default or the number of logical cores.

### QNN/HTP optional lane

The Snapdragon 765G contains Hexagon tensor hardware, but that does not establish
compatibility with a current QNN SDK, ORT QNN build, firmware, or model graph. The QNN
lane begins with a read-only capability probe covering:

- device/firmware and QNN SDK compatibility
- availability and redistribution rules for required QNN libraries
- ORT QNN Android build success for `arm64-v8a`
- QDQ model requirements and complete graph assignment
- strict failure when CPU fallback is disabled
- context-cache creation and reload behavior

Only a successful probe authorizes QNN implementation. The experiment never reports
HTP performance when nodes silently execute on CPU. QNN performance modes, graph
finalization options, and context caching are benchmark variables, not hidden
defaults.

### NNAPI legacy comparison

NNAPI is deprecated in Android 15. If the Nord's installed OS and the selected ORT
package permit it, NNAPI may be measured as an explicitly labelled legacy comparison.
It is default-off, excluded from the long-term support contract, and cannot be the
only accelerated route.

### Fallback truthfulness

`allow_fallback=false` means session creation must fail when the requested provider
cannot execute the required graph under its strict contract. Provider registration is
not execution evidence. Reports include available providers, requested priority,
applied provider options, graph-assignment/fallback diagnostics, and the provider used
for every qualified result.

The current desktop fallback behavior is insufficient for this claim and must be
corrected or isolated before mobile performance results are accepted.

## Model Workloads

The pipeline separates functional fixtures from representative performance models.

1. A small checked ONNX fixture proves packaging, JNI, shape validation, inference,
   exception translation, and numerical parity on every emulator/device lane.
2. The existing fixed-shape YOLOv7 contract (`images [1,3,640,640]` to
   `output [1,25200,85]`) is an initial stress workload because its C++ adapter and
   demos already exist. Its roughly 148 MB source model is not presumed to be a good
   mobile production model.
3. SuperPoint, LightGlue, and large centroiding candidates enter only after the ORT
   mobile usability checker and fixed/dynamic-shape analysis. Their operator
   partitions and task-level accuracy define whether acceleration is meaningful.

Inputs and expected outputs are checksummed. Mobile results are compared with
model-owner golden outputs where available, with desktop CPU ORT as a secondary
cross-platform reference. Quantized models also require a model-owner task metric and
an approved accuracy-loss budget; raw tensor proximity alone is not sufficient.

## Containerized Android Test Lane

Use Google's experimental Android Emulator Container Scripts or a directly derived,
pinned emulator image. The design uses two services:

1. `android-builder`: pinned JDK, Android command-line tools, SDK platforms, NDK,
   CMake, Ninja, and Gradle wrapper; builds AAR/APK artifacts without special device
   privileges.
2. `android-emulator`: pinned Google system/emulator image with `/dev/kvm` passed
   through; exposes ADB only to the local Compose network or loopback.

An ephemeral ADB key is shared only between the test runner and emulator. The runner
waits for both `adb` connectivity and `sys.boot_completed`, installs the release-like
test APK, runs instrumentation tests, retrieves JUnit/logcat/results, and then destroys
the emulator state. The emulator container must not run `--privileged`; `/dev/kvm` is
the narrow required device grant.

The initial matrix contains:

- an `x86_64` system image near the Nord's discovered API level for compatibility
- an API 35 `x86_64` 16 KB-page image to validate every packaged native library
- an optional current API image for forward-compatibility smoke testing

The AAR therefore contains `arm64-v8a` for devices and `x86_64` for tests. An ARM
system image on an x86 host is not used as a performance substitute. QNN, NNAPI
hardware behavior, thermal response, and energy are never qualified in the container.

The current development host has Docker, membership in the Docker group, and an
accessible `/dev/kvm`. It does not currently expose `adb`, an Android SDK, or Gradle.
Those tools belong in the pinned builder image rather than becoming undocumented host
prerequisites.

## Android Build and Packaging Contract

Gradle owns AAR/APK assembly and invokes CMake through `externalNativeBuild`. CMake
continues to own the native library graph. Proposed source ownership is:

```text
mobile/android/
  ptafdeploy-android/       Kotlin API, JNI source, AAR packaging
  benchmark-app/            headless/reference runner
  instrumentation-tests/    device and emulator behavioral tests
  benchmark/                Macrobenchmark and sustained-run orchestration
  docker/                   pinned builder/emulator composition
scripts/mobile/             model preparation and host/device orchestration
```

Android builds use the NDK toolchain and project-qualified options. CUDA, standalone
TensorRT, PTX examples, programs, ROS, Python wrappers, and MATLAB wrappers remain
disabled. Desktop defaults and install/package behavior must remain unchanged.

The first supported ABIs are:

- `arm64-v8a`: release and physical-device qualification
- `x86_64`: emulator correctness only

Every shared library in the APK, including ONNX Runtime and `libc++_shared.so`, must
be 16 KB-page compatible. The pipeline inspects ELF alignment and also installs/runs
on a 16 KB emulator image. Duplicate ORT or C++ runtime libraries are rejected during
APK inspection.

The Android ORT dependency is pinned and exposed to CMake as one
`onnxruntime::onnxruntime` target without changing desktop discovery precedence. If
the upstream AAR does not provide a usable Prefab package, the build stages its
headers and per-ABI native library into an explicit Android SDK root and creates an
imported target. Ordinary configuration never downloads or updates ORT implicitly.

The current core links Eigen unconditionally although mobile inference sources do
not use it outside MATLAB-specific adapters. Dependency slimming should separate the
mobile core from MATLAB-only Eigen/GTSAM adapter requirements instead of cross-
compiling unused dependencies.

## Performance, Power, and Thermal Method

### Measurements

Each candidate reports:

- package, native-library, and model sizes
- session creation and model-load time
- first-inference latency
- warmed p50, p90, p95, and p99 latency
- throughput at the intended concurrency
- peak resident memory and allocation/copy counts where observable
- preprocessing, inference, postprocessing, and end-to-end latency separately
- average power and energy per inference when the device counter is trustworthy
- thermal state/headroom and latency drift through a sustained run
- exact provider, thread, optimization, quantization, and fallback configuration
- output parity or task-level accuracy

Average latency alone is not an acceptance metric.

### Physical-device protocol

Use a non-debuggable release-like APK and representative, checksummed inputs. Record
ambient conditions, battery level and health, display state/brightness, radios,
charging state, background activity, and initial thermal state. Do not compare runs
that begin from materially different thermal or battery conditions.

Run cold-load tests separately. Warm the runtime to steady state, then execute at
least three independent sustained trials per candidate. A first ten-minute trial is
used to reveal thermal throttling; duration increases if battery counters need a
longer integration interval. Compare early steady-state and final-window latency, not
the unrepresentative first iterations.

Android Macrobenchmark/system tracing provides reproducible orchestration and trace
correlation. The Nord is not assumed to expose Pixel-style On Device Power Rails.
Attempt battery charge/current/energy counters only after checking their resolution,
sampling rate, sign, and consistency. Subtract an equally controlled idle baseline
and label the result as device-level energy. If the counters are absent or too noisy,
energy remains unvalidated until an external power instrument is available; battery
percentage deltas are not accepted as precise energy measurements.

The phone's aged battery makes within-device A/B comparisons more credible than
absolute generalization. Battery health is part of every report.

### Candidate selection

The initial tuning matrix is intentionally small:

- CPU FP32 and, where accuracy permits, static INT8 QDQ
- XNNPACK unquantized with bounded thread-count/spinning variants
- QNN HTP QDQ variants only after qualification

Select from the measured latency-energy Pareto frontier. A candidate that is slower
and consumes more energy than another equally correct candidate is rejected. The
first physical campaign establishes evidence-backed absolute budgets for each real
workload; production support is not declared until those budgets are reviewed.

## Test Matrix and Evidence Boundaries

| Lane | Required evidence | Prohibited claim |
|---|---|---|
| Host native | existing C++ behavior and golden-output generation | Android integration |
| Android cross-build | `arm64-v8a`/`x86_64` compile, link, AAR/APK inspection | runtime correctness |
| Docker emulator | install, JNI load, lifecycle, inference, errors, parity, 16 KB compatibility | latency, energy, thermal, QNN/NNAPI hardware |
| OnePlus Nord | arm64 runtime, provider truth, parity, sustained latency, memory, power/thermal where measurable | newer-device generalization |
| iOS later | device/simulator build and physical-device XNNPACK/CoreML evidence | support inferred from Android |

Permanent project tests exercise target-owned inference and Android behavior. They do
not import template-conformance tests or register recursive Android builds inside
ordinary native CTest. Android build/device/emulator matrices remain explicit Gradle,
container, or CI acceptance commands.

## Failure Handling

- Reject missing, corrupt, wrong-hash, unsupported, or incompatible model packages
  before replacing an active session.
- Construct and validate a candidate session transactionally; swap only after model
  metadata and facade contracts are valid.
- Fail clearly when the requested provider, model format, ABI, API level, or QNN
  runtime is unavailable.
- Never convert provider failure into unreported CPU fallback.
- Reject non-direct JNI buffers on fast-path APIs and mismatched shape/cardinality.
- Translate allocation failures and ORT errors without aborting the Android process.
- Preserve the old session after a failed reload and test recovery explicitly.
- Capture native crash tombstones, logcat, ORT profiling output, and structured test
  results as separate artifacts.

## Implementation Plan Requirements and Delivery Phases

This section defines what the later implementation plan must contain. It is not an
authorization to implement, install an SDK, run a remote service, publish a container,
stage, commit, tag, or push. At implementation time, every phase becomes one or more
coherent review batches under the repository's staged-review workflow.

### Required structure of every planned task

Each implementation task must state:

1. **Objective and non-goals:** one observable outcome and the behavior deliberately
   left unchanged
2. **Prerequisites:** approved prior gates, device/tool availability, input artifacts,
   licenses, and repository state
3. **Owned paths:** exact files or directories allowed to change and adjacent surfaces
   that must remain untouched
4. **Contract change:** API, ABI, model, package, provider, lifecycle, or evidence
   behavior introduced by the task
5. **Implementation steps:** small ordered actions with the exact build or orchestration
   entry point they use
6. **Produced artifacts:** AAR/APK, native libraries, models, reports, hashes, logs, or
   test results and whether each is committed or generated/ignored
7. **Verification:** exact host, cross-build, emulator, or device command; expected
   observable result; and retained evidence
8. **Failure and rollback:** how partial outputs are removed or ignored and how the old
   working path remains usable
9. **Exit gate:** a binary review decision with no support claim inferred from a weaker
   lane
10. **Review batch:** staged allowlist, exclusions, validation repeated on the staged
    representation, and proposed commit message without implicit commit authorization

A task entry should read approximately as follows:

```text
Task: Build the x86_64 reference AAR
Objective: Produce an emulator-loadable AAR through the pinned builder
Owned paths: Android library module, Android CMake adapter, focused build documentation
Inputs: Pinned ORT package, checked fixture, dependency lock, builder digest
Actions: Configure -> build -> inspect archive and ELF closure -> publish locally
Verify: Consumer APK assembles; loader test passes; one ORT and one C++ runtime are present
Gate: Pass packaging checks before adding JNI inference behavior
Exclusions: No arm64 performance, provider, power, or production-support claim
```

The final plan must prefer exact commands and expected outputs over prose such as
"test Android." When exact version values depend on the connected Nord or a selected
ORT release, the discovery task writes them into a checked lock/qualification record
before downstream tasks consume them.

### Critical path and optional branches

```text
0 repository baseline
        |
1 Nord/toolchain preflight
        |
2 checked model and golden pipeline
        |
3 pinned Android builder
        |
4 native cross-build and package skeleton
        |
5 JNI/Kotlin AAR -> 6 reference app -> 7 Docker emulator gates
                                      |
                                      v
                              8 Nord CPU/XNNPACK baseline
                                      |
                         +------------+-------------+
                         |                          |
                         v                          v
             9 measured data-path tuning   10 reduced ORT/quantization
                         |                          |
                         +------------+-------------+
                                      |
                         +------------+-------------+
                         |                          |
                         v                          v
                  11 QNN gated branch      12 alternate-runtime spike
                         |                          |
                         +------------+-------------+
                                      |
                              13 promotion review
                                      |
                              14 optional iOS plan
```

Phases 11, 12, and 14 are not part of the minimum Android critical path. Phase 12 tests
one alternative runtime at a time only after a reviewed trigger; it does not add LiteRT
and ExecuTorch together by default.

### Phase 0: establish repository and behavioral baselines

**Objective:** start mobile work without absorbing unrelated user work and establish a
reproducible desktop oracle.

The plan must:

- record the branch, HEAD, index, unstaged/untracked paths, worktrees, submodules, and
  current generated build roots
- identify every pre-existing change that overlaps the proposed Android/native seams
- use an isolated worktree when authorized, or wait until overlapping work has an owner;
  never require unrelated changes to be discarded merely to make the mobile tree clean
- inventory the actual facade, tensor-view, task-value, task-adapter, runtime-config,
  ORT, TensorRT, wrapper, program, and ROS consumers
- rerun the smallest authoritative native baseline plus the selected model fixture and
  record exact commands, versions, outputs, and known failures
- record the ONNX Runtime source/package version and discovery path used by the desktop
  oracle

**Outputs:** a repository-state record, change-surface inventory, native baseline report,
and initial dependency lock. Generated logs remain under an ignored results root.

**Gate:** no mobile edit begins until overlapping changes have an ownership strategy and
the desktop oracle is reproducible or every pre-existing failure is explicitly recorded.

### Phase 1: qualify the target and freeze Android requirements

**Objective:** replace assumed launch specifications with the actual OnePlus Nord and a
pinned Android support envelope.

The plan must acquire the phone non-destructively through ADB and record:

- exact model/board/SoC, ABI list, API level, build fingerprint, security patch, kernel,
  memory page size, available storage, and loader namespace behavior
- battery level, health, charge/current/energy counter availability and resolution,
  charging state, thermal APIs, initial thermal state, and repeatability limitations
- CPU topology/frequency visibility, available performance hints, and whether required
  tracing counters are accessible without rooting the phone
- whether the installed OS can run the selected minimum/target SDK combination
- whether QNN runtime/firmware compatibility can even be investigated under applicable
  SDK licensing, without treating hardware marketing names as support proof

In parallel, record host Docker/KVM capability and select pinned JDK, Android command-
line tools, platform, build-tools, NDK, CMake, Ninja, and Gradle wrapper versions.

**Outputs:** a machine-readable device preflight report, human-readable support envelope,
and Android toolchain lock.

**Gate:** the critical path requires a usable `arm64-v8a` phone and a viable SDK/API
combination. Missing trustworthy energy counters do not block correctness work, but
they set the later energy result to power-unvalidated unless external instrumentation
is supplied. QNN uncertainty does not block CPU/XNNPACK.

### Phase 2: create the reproducible model and golden-data pipeline

**Objective:** make model behavior independent of Android packaging before introducing
mobile variables.

The plan must:

- choose one small repository-appropriate functional model and one representative
  performance workload, with provenance, license, source hash, shapes, dtypes, and role
- capture checksummed representative inputs and model-owner golden outputs; use desktop
  CPU ORT as a secondary execution reference, not the sole semantic oracle
- validate fixed and dynamic dimensions, operator/opset requirements, tensor names,
  preprocessing, postprocessing, and task-level metric definitions
- define numeric tolerance by output/task and reject one global tolerance chosen only
  because it makes a test pass
- add a pinned preparation command that can validate ONNX, optionally create optimized
  ONNX/ORT candidates, emit the required-operator configuration, and write a manifest
  containing every tool argument and hash
- keep large or generated models, runtime packages, and result artifacts outside Git
  unless a separately reviewed fixture policy permits a small artifact

**Outputs:** checked model package(s), input/golden set, artifact manifest, parity report,
and deterministic preparation entry point under `scripts/mobile/`.

**Gate:** the full ONNX candidate must pass desktop contract and parity checks before it
is used to diagnose Android. Quantized or converted candidates remain separate and may
not replace the baseline oracle.

### Phase 3: build the pinned Android builder environment

**Objective:** make Android builds independent of undocumented host SDK state.

The plan must:

- create the `android-builder` image with pinned downloads and verified checksums for
  JDK, SDK command-line tools, platform/build-tools, NDK, CMake, Ninja, and other build
  dependencies
- use the repository Gradle wrapper and dependency locking; prohibit `latest.release`,
  unpinned nightlies, or implicit network downloads in the reproducibility acceptance
  command
- separate normal dependency acquisition from an offline rebuild check so missing
  cache inputs fail clearly
- build as a non-root user where practical, mount only explicit source/output/cache
  paths, and require no KVM or privileged access in the builder
- record the image digest, package/version inventory, Gradle dependency graph, CMake
  cache summary, and license/notice inputs

**Outputs:** builder recipe, local image, dependency locks, provenance report, and a
single documented build entry point.

**Gate:** two clean builder runs must resolve the same locked inputs and produce AAR/APK
contents with the same ABI/dependency closure. Bit-identical archives are desirable but
not claimed until timestamp and tool nondeterminism have been audited.

### Phase 4: cross-compile the native core and assemble the package skeleton

**Objective:** prove that the existing C++ ownership can become an Android native
dependency before adding product JNI behavior.

The plan must:

- add Android-qualified CMake entry points without changing desktop option defaults or
  ONNX Runtime discovery precedence
- disable CUDA, standalone TensorRT, PTX examples, programs, ROS, Python/MATLAB wrappers,
  tests that cannot execute under cross-compilation, and unrelated template utilities
- expose the pinned full Android ORT package as one `onnxruntime::onnxruntime` target;
  ordinary configuration must not download or mutate it
- separate mobile inference sources from unused Eigen/GTSAM or MATLAB-only adapters
  instead of cross-compiling their complete dependency closure
- compile `arm64-v8a` and `x86_64` release-like libraries and assemble an AAR skeleton
- inspect archive contents, ELF class/machine, SONAME/NEEDED closure, exported symbols,
  STL selection, duplicate libraries, stripping/debug symbols, and 16 KB load alignment
- rerun the existing desktop build/tests to detect accidental target-composition changes

**Outputs:** two-ABI native libraries, package skeleton, dependency/ELF inspection report,
and a minimal Android consumer link test.

**Gate:** both ABIs link and package with one ORT and one C++ runtime; desktop behavior is
unchanged; no Android result is yet called runtime-correct.

### Phase 5: implement the narrow JNI/Kotlin AAR contract

**Objective:** expose the existing role-level facade with deterministic Android ownership
and bounded data movement.

The plan must cover:

- a thread-safe native handle registry with create, lookup, close, stale-handle rejection,
  and cleanup after partial construction failure
- one `Closeable`/`AutoCloseable` Kotlin session whose state machine is explicit and whose
  `Cleaner`, if used, is only a leak backstop, not the normal ownership path
- manifest/runtime-policy loading, contract metadata, inference, diagnostics, and close;
  raw ORT handles and backend classes never cross JNI
- direct-buffer validation for address, directness, dtype, shape, cardinality, byte size,
  alignment assumptions, lifetime, and read/write direction
- stable exception categories for invalid input, missing/corrupt model, unsupported
  provider, native allocation, ORT execution, and closed session
- one-inference-at-a-time semantics per session and explicit behavior for concurrent
  calls, close-during-run, failed load, and failed reload
- a first correct `SFloatTensor` path plus instrumentation capable of showing whether a
  later `STensorView` fast path is warranted

**Outputs:** reusable AAR, API documentation, JNI/native unit tests where host-testable,
Kotlin unit tests, symbol mapping, and lifecycle/error contract.

**Gate:** a minimal consumer can create, inspect, infer, fail predictably, and close
without leaking or aborting. Performance is not accepted until the reference app and
physical lane use the same API.

### Phase 6: add the release-like reference and benchmark application

**Objective:** exercise the reusable AAR exactly as an external Android consumer would.

The plan must:

- create a small non-production application with no backend-private calls
- copy a versioned model/config package from assets to app-private storage atomically,
  verify hashes, and retain the old valid package after failed replacement
- expose deterministic instrumentation entry points for load, first inference, repeated
  inference, task-level output, diagnostics, memory sampling, and sustained runs
- separate preprocessing, input transfer, inference, output transfer, postprocessing,
  and end-to-end timing without changing the production call graph being measured
- build a non-debuggable release-like benchmark variant and a diagnosable test variant
- inspect the final APK for ABI filters, one ORT/STL closure, native symbols, compression,
  model duplication, signature, and 16 KB compatibility

**Outputs:** reference APK, instrumentation APK, package inspection report, and stable
machine-readable result schema.

**Gate:** package assembly and static inspection pass for both ABIs. Runtime claims wait
for the emulator and device phases.

### Phase 7: implement Docker/KVM emulator qualification

**Objective:** make Android packaging and behavior reproducible without consuming the
physical phone for every change.

The plan must:

- compose the pinned `android-builder` with an independently pinned emulator image;
  pass only `/dev/kvm`, never use `--privileged`, and keep ADB on the local network or
  loopback with ephemeral keys
- wait for ADB authorization, package manager readiness, and `sys.boot_completed`
  before installation rather than relying on a fixed sleep
- test an `x86_64` API near the Nord, API 35 with 16 KB pages, and an optional current
  forward-compatibility image
- install the release-like app, execute JNI load/lifecycle/inference/error/reload/parity
  tests, and uninstall or destroy emulator state after the run
- retain JUnit/XML, structured results, logcat, native tombstones, package/ELF reports,
  system-image digest, and exact emulator arguments
- prove that QNN, NNAPI hardware behavior, latency, energy, and thermal results are not
  emitted from this lane

**Outputs:** container composition, boot/test orchestrator, emulator matrix, and archived
correctness evidence.

**Gate:** all functional tests pass on the normal and 16 KB lanes. Emulator success
authorizes physical testing, not support or performance claims.

### Phase 8: establish the OnePlus Nord CPU/XNNPACK baseline

**Objective:** obtain the first authoritative end-to-end performance, power, thermal,
and provider evidence from the actual target.

The plan must:

- rerun device preflight before each campaign and reject incomparable initial battery,
  charging, background-load, or thermal conditions
- install the exact inspected release-like APK and verify its artifact/model hashes
- run correctness first, then separate cold load, first inference, warmed percentiles,
  intended-concurrency throughput, memory, and sustained thermal trials
- sweep a bounded, predeclared CPU/XNNPACK thread and spinning matrix; do not search an
  unlimited tuning space until one result looks favorable
- capture provider availability, requested options, resolved backend detail, fallback
  evidence, Android/system traces, and application stage timings
- assess counter resolution and idle subtraction before reporting energy; otherwise mark
  the campaign power-unvalidated and retain all other valid measurements
- run at least three controlled sustained trials per promoted candidate and compare early
  steady-state with the final measurement window

**Outputs:** immutable baseline report, raw results/traces, device-state record, accuracy
result, and an initial reviewed set of workload-specific budgets or explicitly deferred
budgets.

**Gate:** CPU and XNNPACK must be correct and their resolved provider configuration must
be truthful. The next phase uses the measured bottleneck and Pareto frontier; it does
not assume XNNPACK wins.

### Phase 9: optimize only measured data-path and runtime bottlenecks

**Objective:** reduce latency or energy without adding Android-specific complexity that
the baseline does not justify.

The plan must rank observed cost in preprocessing, copies/allocations, inference,
postprocessing, model load, and storage duplication. Candidate changes are introduced
one at a time and may include:

- reuse of preallocated direct buffers and output storage
- a native-only `STensorView` facade path that preserves wrapper-safe Python/MATLAB APIs
- bounded thread count, affinity only when Android permits a stable contract, and ORT/
  XNNPACK spinning configuration
- memory-mapped or asset-manager model input when private-copy cost is material
- provider I/O binding or `AHardwareBuffer` only after proving compatible buffer ownership
  and avoiding hidden conversions

Every candidate repeats correctness, warm/sustained latency, memory, and trustworthy
energy measurements against the unchanged baseline. The plan must predeclare the minimum
meaningful improvement and complexity budget before seeing the result.

**Outputs:** stage-level profiles, one-change-at-a-time comparisons, accepted/rejected
optimization ledger, and updated Pareto report.

**Gate:** retain only behavior-preserving candidates that improve an approved objective
without an unacceptable regression. Remove experimental complexity that fails the gate.

### Phase 10: reduce runtime/model footprint and qualify quantized candidates

**Objective:** create a smaller deliverable without confusing footprint work with the
first correctness baseline.

The plan must:

- convert the complete promoted model set to ORT format with the same pinned ORT version
  used to build the custom runtime
- generate, review, hash, and retain the required-operator/type configuration
- build full and reduced Android AARs for both ABIs from pinned settings
- run clean model-load and parity tests that detect a missing kernel/type rather than
  silently returning to a full runtime
- inspect AAR/APK/native/model/private-storage size and memory, not just the compressed
  APK number
- produce CPU INT8 or other quantized candidates only with representative calibration,
  model-owner task metrics, and an approved accuracy-loss budget
- repeat the physical latency/energy/thermal protocol because a smaller or quantized
  candidate is not automatically faster or more efficient

**Outputs:** ORT model(s), operator/type config, custom runtime package, provenance and
size reports, accuracy results, and updated physical Pareto comparison.

**Gate:** promote reduced ORT only when every supported model loads and passes parity and
the footprint benefit justifies model/runtime lockstep maintenance. Promote quantization
only when it meets both task accuracy and physical-device objectives.

### Phase 11: probe and optionally qualify QNN/HTP

**Objective:** determine whether the original Nord can use a current, redistributable,
truthful ORT QNN path before owning product code for it.

The phase has two separately reviewed parts.

**11A, compatibility probe:**

- verify the Nord SoC/firmware against the selected QNN SDK and ORT build requirements
- resolve SDK/runtime licenses and redistribution before packaging vendor libraries
- build or acquire a pinned `arm64-v8a` QNN-enabled ORT without changing the CPU baseline
- fix dynamic shapes, produce representative calibrated QDQ input, and check operator
  coverage
- create the session with CPU fallback disabled and prove complete assignment or retain
  the exact rejection
- test context creation/cache reload and capture QNN/ORT profiling capability

**11A gate:** if compatibility, redistribution, complete graph assignment, or stable
session creation fails, stop the branch and record QNN as unsupported for this device.
Do not add a silent partial-offload path.

**11B, implementation and qualification, only after 11A passes:**

- expose QNN options through backend-neutral validated runtime policy and diagnostics
- package the exact required libraries with ABI/version checks
- implement context-cache integrity/version behavior and HTP subsystem-restart recovery
- benchmark explicit performance modes and graph-finalization choices using the same
  sustained protocol, accuracy metric, and input set as CPU/XNNPACK
- measure load/cache trade-offs, memory, thermals, and energy rather than reporting only
  peak inference latency

**Outputs:** compatibility record, QDQ/context artifacts, provider-assignment proof,
license/package report, profiles, and same-device Pareto comparison.

**11B gate:** QNN is promoted only if it is correct, supportable, fully assigned under
the claimed strict contract, and materially improves an approved physical objective.

### Phase 12: run one evidence-triggered alternative-runtime spike

**Objective:** evaluate LiteRT, ExecuTorch, or direct QNN only when the ORT results expose
a reviewed reason to pay for another runtime.

Before the spike, the plan must identify the trigger: missed energy/latency/footprint
budget, unsupported operators, unusable accelerator access, or a model-ownership shift.
Then it must:

- select exactly one runtime and one representative model; do not build a generic
  multi-runtime abstraction as part of the probe
- use the official export/lowering path to `.tflite`, `.pte`, or vendor artifact with
  pinned tools, hashes, operator/delegation reports, and task-level parity
- build the smallest disposable Android integration that still uses the same inputs,
  stage timing boundaries, APK inspection, release-like settings, and Nord protocol
- prove actual delegate/backend assignment and record fallback or partitioning
- compare sustained latency, energy, thermal drift, memory, model/runtime footprint,
  conversion reliability, developer workflow, licensing, and maintenance surface
- remove or keep the spike isolated if it fails; do not leave a half-supported backend
  in the public enum or manifest

**Outputs:** throwaway integration artifact, reproducible export record, and a normalized
comparison report against the promoted ORT candidate.

**Gate:** promoting the alternative requires a new backend architecture decision and
plan. It must demonstrate parity and a predeclared meaningful Pareto advantage large
enough to justify dual-runtime ownership; API brevity or one warm latency number is not
sufficient.

### Phase 13: promotion, consumer, and support review

**Objective:** turn experimental evidence into a narrowly truthful supported deliverable,
or explicitly conclude that the experiment is not ready for promotion.

The plan must:

- run a clean, pinned release matrix for host regression, both Android ABIs, normal and
  16 KB emulators, the physical Nord, and every promoted provider/model candidate
- build a minimal external Gradle consumer from the released AAR coordinate or locked
  artifact bundle and verify runtime loading outside the source tree
- audit exported symbols, R8/ProGuard needs, native crash symbols, third-party notices,
  SDK/runtime/model licenses, vulnerability/update ownership, and package/model hashes
- publish a support matrix that distinguishes built, emulator-tested, device-tested,
  performance-qualified, power-qualified, optional, legacy, and unsupported behavior
- document model/runtime compatibility, update/rollback procedure, diagnostics, known
  limitations, and exact reproduction commands
- review every generated artifact and decide what is retained locally, archived by CI,
  attached to a release, or deliberately excluded from version control

**Outputs:** release candidate bundle, external-consumer evidence, support matrix,
third-party notices, benchmark report, and explicit promotion recommendation.

**Gate:** support is declared only for exact tested ABIs, device/API envelope, models,
providers, and evidence levels. An unavailable trustworthy power source remains visible
as power-unvalidated rather than being waived.

### Phase 14: optional iOS plan

**Objective:** use Android evidence to write a separate iOS design/implementation plan,
not to claim iOS support.

The plan would identify a Mac/Xcode/signing environment and physical Apple target,
package the same native facade through XCFramework/CocoaPod and Objective-C++/Swift,
qualify XNNPACK/CoreML, and apply the same model provenance, direct-buffer, sustained
performance, and energy boundaries. It must account for Apple-specific lifecycle,
memory, packaging, Instruments, and distribution rather than mechanically porting the
Android Gradle/JNI layer.

**Gate:** no iOS implementation starts without its own approved plan and access to the
required build/signing/device environment.

### Review and sequencing rule

Each phase is a separate coherent review batch unless two adjacent documentation-only
or build-seam steps are demonstrably inseparable. Before preparing a batch, inspect the
complete live index and worktree; stage only an explicit allowlist; review the complete
cached diff; run `git diff --cached --check`; repeat affected validation; report caveats
and the exact proposed commit message; then stop for approval. No commit, tag, push,
remote build, SDK installation, or container publication is implied by this design.

## Acceptance Criteria for the Experiment

- A pinned build produces inspectable `arm64-v8a` and `x86_64` Android artifacts.
- Container tests install and execute the real JNI/facade path on the normal and
  16 KB-page emulator lanes.
- The physical Nord executes the same checked inputs and satisfies approved numerical
  or task-level parity.
- Strict provider runs prove graph assignment or fail; no result relies on silent
  fallback.
- Reports distinguish cold, warm, sustained, and end-to-end behavior and preserve
  complete runtime/device configuration.
- The selected runtime configuration is Pareto-efficient for latency and energy under
  equal correctness.
- Thermal degradation is measured over a sustained run and is not hidden by short
  warmup-only timing.
- If trustworthy energy measurement is unavailable on the Nord, the result is marked
  power-unvalidated rather than passed.
- Existing desktop CPU/CUDA/TensorRT, wrappers, examples, packaging, and ROS behavior
  remain unchanged unless a separately approved prerequisite correction is required.

## Optional iOS Extension

After the Android lane is validated, reuse the same C++ facade in an XCFramework or
local CocoaPod with a narrow Objective-C++/Swift bridge. Start with CPU/XNNPACK and
then CoreML on a physical Apple Neural Engine device. ORT-format/operator-reduction,
artifact provenance, direct-buffer ownership, numerical parity, and sustained power
methodology mirror Android. Xcode Instruments supplies iOS runtime analysis.

iOS requires a Mac, Xcode, code signing, and physical-device evidence. An iOS
simulator build is a correctness lane only and cannot be inferred from Android or
implemented/validated on the current Linux host.

## Primary References

- [OnePlus Nord specifications](https://www.oneplus.com/global/nord-specs)
- [Qualcomm Snapdragon 765G product brief](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/prod_brief_qcom_sd765g_5g_1.pdf)
- [ONNX Runtime mobile deployment](https://onnxruntime.ai/docs/tutorials/mobile/)
- [ONNX Runtime Android build](https://onnxruntime.ai/docs/build/android.html)
- [ONNX Runtime custom/reduced builds](https://onnxruntime.ai/docs/build/custom.html)
- [ONNX Runtime ORT model format](https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html)
- [ONNX Runtime mobile model usability checker](https://onnxruntime.ai/docs/tutorials/mobile/helpers/model-usability-checker.html)
- [ONNX Runtime Java API](https://onnxruntime.ai/docs/api/java/)
- [ONNX Runtime XNNPACK provider](https://onnxruntime.ai/docs/execution-providers/Xnnpack-ExecutionProvider.html)
- [ONNX Runtime QNN provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [LiteRT for Android](https://ai.google.dev/edge/litert/android)
- [LiteRT prebuilt Android C++ integration](https://ai.google.dev/edge/litert/next/android_cpp_sdk)
- [ExecuTorch Android integration](https://docs.pytorch.org/executorch/stable/using-executorch-android.html)
- [ExecuTorch Android backends](https://docs.pytorch.org/executorch/stable/android-backends.html)
- [ExecuTorch Qualcomm backend](https://docs.pytorch.org/executorch/stable/backends-qualcomm.html)
- [Android NNAPI deprecation and migration](https://developer.android.com/ndk/guides/neuralnetworks/migration-guide)
- [Android native AAR/Prefab dependencies](https://developer.android.com/build/native-dependencies)
- [Google Android emulator container scripts](https://github.com/google/android-emulator-container-scripts)
- [Android Macrobenchmark metrics and PowerMetric](https://developer.android.com/topic/performance/benchmarking/macrobenchmark-metrics)
- [Android Power Profiler](https://developer.android.com/studio/profile/power-profiler)
- [Android Dynamic Performance Framework](https://developer.android.com/games/optimize/adpf)
- [Android 16 KB page-size support](https://developer.android.com/guide/practices/page-sizes)
- [ONNX Runtime CoreML provider](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [ONNX Runtime iOS build](https://onnxruntime.ai/docs/build/ios.html)
- [Official ORT Android image-classification example](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/mobile/examples/image_classification/android)
