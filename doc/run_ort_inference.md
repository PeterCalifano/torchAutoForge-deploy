# One-Shot ONNX Inference

`run_ort_inference` loads one `.onnx` artifact through
`ptafdeploy::inference::CInferenceManager` and either prints its tensor contract
or executes one inference from prepared float32 tensors. It is intentionally a
tensor-level utility: PNG decoding, normalization, detection decoding, and
centroid interpretation belong in model integration examples.

## Build And Discovery

ONNX Runtime is required for normal builds. Point CMake at its config package
with either supported form:

```bash
./build_lib.sh \
  -D onnxruntime_DIR=/path/to/onnxruntime/lib/cmake/onnxruntime

export ONNXRUNTIME_ROOT=/path/to/onnxruntime
./build_lib.sh
```

`ONNXRUNTIME_ROOT` must contain
`lib/cmake/onnxruntime/onnxruntimeConfig.cmake`. The executable is normally
written to `build/src/programs/run_ort_inference`.

## Inspect A Model

Metadata-only mode loads the requested execution providers and reports every
input/output name, dtype, and declared shape without requiring tensor values:

```bash
./build/src/programs/run_ort_inference model.onnx --metadata-only
```

Dynamic dimensions are printed as `-1`. They require a concrete `--shape` only
when inference is requested. Metadata-only mode rejects tensor and output
options instead of silently ignoring them.

## Run Prepared Tensor Data

Each model input requires exactly one source:

- `--input [name=]path.f32` reads a dense native-endian IEEE-754 float32 file
- `--fill [name=]value` creates a deterministic tensor filled with one finite
  value, useful for smoke checks

Static input shapes are read from the model. Supply every dynamic dimension as
a positive integer:

```bash
./build/src/programs/run_ort_inference model.onnx \
  --shape 1,3,640,640 \
  --input input.f32
```

Raw files contain values only—no header, shape, strides, dtype tag, or padding.
The file byte count must equal `product(shape) * sizeof(float)`. Values use the
model's dense logical order; for an NCHW image input this means channel-first
row-major data prepared before invoking the CLI.

For a multi-input model, qualify every input, shape, and fill with the exact
model input name reported by `--metadata-only`:

```bash
./build/src/programs/run_ort_inference model.onnx \
  --input image=image.f32 \
  --shape image=1,1,1536,2048 \
  --fill prior=0.0
```

Unqualified specifications are accepted only when the model has one input.
Duplicate or unknown names, missing sources, file/fill conflicts, non-float32
inputs, shape mismatches, and wrong file sizes fail before execution with an
actionable diagnostic.

## Runtime Selection

The runtime options use the same backend-neutral policy as the library:

```text
--targets cpu,cuda,tensorrt  execution-target priority
--device N                   non-negative device index
--no-fallback                reject lower-priority fallback
--intra-op-threads N         ORT intra-operation threads; zero uses its default
--inter-op-threads N         ORT inter-operation threads; zero uses its default
```

For a CPU-only reproducible run:

```bash
./build/src/programs/run_ort_inference model.onnx \
  --shape 1,11 --fill 0 --targets cpu --no-fallback
```

Use `get_available_providers --require-targets cuda` before requesting CUDA on a
new machine. TensorRT in `--targets` means the ONNX Runtime TensorRT Execution
Provider; standalone serialized `.engine` inference remains the separate
TensorRT backend.

## Result And Output Files

Stdout is a stable line-oriented result summary. It includes backend detail,
model tensor metadata, and for each executed output:

```text
output[0].name=prediction
output[0].shape=[1,2]
output[0].elements=2
output[0].values=0.5,0.5
```

`--max-output-values N` bounds each preview and accepts zero to suppress values.
`--output-dir DIR` creates the directory when necessary and writes every output
as `INDEX_SANITIZED_NAME.f32`, for example `0_prediction.f32`. Output files use
the same headerless native-endian float32 representation as input files; shape
and element count remain in stdout.

Human diagnostics go to stderr through `CLogger`, leaving stdout available for
result parsing. Set `PTAFDEPLOY_LOG_LEVEL=debug`, `info`, `warning`, `error`,
`critical`, or `off` to change verbosity.

## Exit Status

- `0`: help/version request, metadata inspection, or successful inference
- `1`: invalid command line, tensor specification, shape, path, or file size
- `2`: model loading, provider setup, inference, or output-write failure

Run `run_ort_inference --help` for the complete option list and short inline
descriptions.
