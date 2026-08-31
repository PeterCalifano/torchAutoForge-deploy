# Plain Image-Only Centroiding

The native C++ CLI and generated Python/MATLAB wrapper demos run a plain
centroiding CNN from one ordinary image. Each integration owns its
model-specific grayscale, resize, and coordinate policy above the generic
`CModelFacade`, and reports the predicted centroid in normalized,
model-input-pixel, and original-image-pixel coordinates.

This example is intentionally separate from the FiLM centroiding integration.
It supplies no prior vector and expects no centre-of-brightness output.

## Verified Model Contract

The example was qualified with the external checkpoint:

```text
best_model_plain_traveling-goat-68_22b61bbd4ddd.pth
```

Provenance and integrity:

- model type: `plain`
- MLflow run: `traveling-goat-68`
- full run ID: `22b61bbd4ddd4d6cb6f503a22f050a25`
- source commit: `ba34bdd8d7fe09e663b7c506f9b4b92afdfc8333`
- checkpoint SHA-256:
  `8022e92f4de8b86788bc523d5e45d5d09ab414161eac85def1d15c10fc39f12d`
- training-config SHA-256:
  `82a19001e2f83a1f53c37567e1f218d4b3c5ae623af1bea1804920aa5c03e4c1`

The verified opset-11 ONNX export has SHA-256
`8ba4f46355b0f6b542ec848b4c1130760bea194f9bf3b16e7b36bad9f380af4b`
and the following runtime contract:

```text
input  image       float32  [batch_size,1,1536,2048]
output prediction  float32  [batch_size,2]
```

`prediction` stores `[x / 2048, y / 1536]`. The batch axis is dynamic, while
this example always submits a batch of one. The original `.pth` checkpoint and
the exported `.onnx` are external artifacts and are not committed to this
repository.

## Export And Artifact Placement

Use the owning `ml-based-centroiding` export utility to build the plain ONNX
from the checkpoint. Its normal command also evaluates a selected dataset and
checks PyTorch/ONNX numerical agreement:

```bash
cd /path/to/ml-based-centroiding
python python/scripts/export_evaluate_onnx.py \
  --checkpoint /path/to/best_model_plain_traveling-goat-68_22b61bbd4ddd.pth \
  --datasets DATASET_NAME \
  --dataset-roots /path/to/datasets \
  --device cpu \
  --onnx-provider CPUExecutionProvider \
  --onnx-export-dir /path/to/onnx_exports
```

The tracked manifest resolves its artifact relative to
`examples/model_configs/`. To use it without editing the manifest, copy or
symlink the external export to the ignored local model tree:

```bash
mkdir -p models/onnx
ln -s /path/to/best_model_plain_traveling-goat-68_22b61bbd4ddd.onnx \
  models/onnx/best_model_plain_traveling-goat-68_22b61bbd4ddd.onnx
```

Alternatively, pass the external `.onnx` directly to the executable. Raw ONNX
loading assigns the `centroiding` role in the example and does not require a
temporary manifest.

## Build

Build the producer library first. ONNX Runtime discovery follows the main
project rules:

```bash
cmake -S . -B build \
  -D CMAKE_BUILD_TYPE=RelWithDebInfo \
  -D onnxruntime_DIR=/path/to/onnxruntime/lib/cmake/onnxruntime
cmake --build build --parallel
```

Then configure the standalone example against that build-tree package:

```bash
cmake -S examples/plain_centroiding -B build-plain-centroiding \
  -D CMAKE_BUILD_TYPE=RelWithDebInfo \
  -D autoforge_deploy_DIR="$PWD/build" \
  -D onnxruntime_DIR=/path/to/onnxruntime/lib/cmake/onnxruntime
cmake --build build-plain-centroiding --parallel
```

Build both generated wrappers with the tracked interfaces and package sources:

```bash
./build_lib.sh -B build-wrappers -p -m \
  -D onnxruntime_DIR=/path/to/onnxruntime/lib/cmake/onnxruntime
```

Use only `-p` or `-m` when one language is needed. Ordinary wrapper builds do
not update the gtwrap checkout. The build helper also runs the existing
target-owned Python and MATLAB facade smokes when their runtimes are available.

OpenCV `core`, `imgproc`, and `imgcodecs` are required by this standalone
integration, not by the core deployment library. The repository-bundled TCLAP
headers remain a private dependency of the executable. Catch2 v3 is needed only
when `BUILD_TESTING=ON`.

## Run

Run the tracked manifest on CPU:

```bash
./build-plain-centroiding/run_plain_centroiding \
  examples/model_configs/ml_based_centroiding_plain.ptafmodel \
  examples/plain_centroiding/test_data/bright_ellipse_320x240.png
```

Run an external raw ONNX on CUDA without fallback:

```bash
./build-plain-centroiding/run_plain_centroiding \
  /path/to/best_model_plain_traveling-goat-68_22b61bbd4ddd.onnx \
  /path/to/image.png \
  --targets cuda \
  --no-fallback
```

Use `--targets cpu,cuda,tensorrt` to provide a priority list, `--device` to
select a runtime device, and `--intra-op-threads` / `--inter-op-threads` to
replace thread settings. Runtime options replace manifest runtime policy only
when an override is explicitly supplied. For an ONNX artifact, `tensorrt`
selects the ONNX Runtime TensorRT Execution Provider rather than the standalone
serialized-engine backend.

For the tracked ellipse fixture, the verified CPU result includes:

```text
input_shape=[1,1,1536,2048]
image_size=[320,240]
output_shape=[1,2]
centroid.normalized_x=0.680052638
centroid.normalized_y=0.371422648
centroid.original_x_px=217.616852
centroid.original_y_px=89.1414337
```

The labeled ellipse center is `(218, 91)` in the original image. The measured
CPU error is 1.90 px; the native optional regression uses a conservative 5 px
tolerance. The CUDA path was also verified and produced `(217.615, 89.144)`.

Set `PTAFDEPLOY_LOG_LEVEL=debug`, `info`, `warning`, `error`, `critical`, or
`off` to control diagnostics. Result fields remain on stdout, while CLogger
diagnostics use stderr.

Run `run_plain_centroiding --help` for the complete option list. Exit status
`0` means help/version output or successful inference, `1` means invalid CLI,
image, or model contract input, and `2` means model loading or inference failed.

## Python Wrapper Demo

The Python demo uses the generated `autoforge_deploy` package, Pillow for image
decoding/resizing, and NumPy for the flat image buffer. Run it directly from a
wrapper build tree with:

```bash
PYTHONPATH="$PWD/build-wrappers/python" \
python3 examples/plain_centroiding/run_plain_centroiding.py \
  examples/model_configs/ml_based_centroiding_plain.ptafmodel \
  examples/plain_centroiding/test_data/bright_ellipse_320x240.png \
  --target cpu
```

Pass `--target manifest` to retain manifest runtime policy, or `--target cuda`
to require the CUDA provider without fallback. A raw `.onnx` path is accepted
in place of the manifest and is assigned the `centroiding` role locally.

The verified Python CPU result for the tracked ellipse was
`(217.6168, 89.1426)`; CUDA produced `(217.6199, 89.1446)` and reported CUDA as
the applied provider.

## MATLAB Wrapper Demo

Start MATLAB with the build-tree core library available to the dynamic loader:

```bash
LD_LIBRARY_PATH="$PWD/build-wrappers/src:$LD_LIBRARY_PATH" matlab
```

Then add the generated wrapper, MEX binary, and demo directories:

```matlab
addpath("build-wrappers/wrap/autoforge_deploy")
addpath("build-wrappers/wrap/autoforge_deploy_mex")
addpath("examples/plain_centroiding")

stResult = RunPlainCentroidingFacadeDemo( ...
    "examples/model_configs/ml_based_centroiding_plain.ptafmodel", ...
    "examples/plain_centroiding/test_data/bright_ellipse_320x240.png", ...
    strTarget="cpu");
```

The function returns role/backend strings, concrete input/output shapes, image
size, and two-element normalized/model-input/original-image coordinate vectors.
Its verified CPU result was `(217.6175, 89.1478)`; CUDA produced the same
rounded coordinates and reported CUDA as the applied provider.

Both wrappers use generic task surfaces. Python consumes `DecodeFeatureRows`,
while MATLAB consumes `DecodeFeatureRowsMatrix`, whose rows are
`[x, y, score]`. Neither wrapper API contains a plain-centroiding decoder or
image type.

## Preprocessing And Coordinate Semantics

Every integration applies these steps in order:

1. the owning image library decodes and converts the source to unsigned 8-bit
   grayscale
2. bilinear interpolation resizes directly to the model's concrete `H x W`
3. the generic HWC-to-NCHW adapter creates `[1,1,H,W]`
4. each grayscale value is multiplied by `1 / 255`

The native CLI uses OpenCV, Python uses Pillow, and MATLAB uses `imresize` with
antialiasing disabled. Their interpolation conventions can differ slightly, so
small output differences are expected even though shape, scaling, role, and
coordinate contracts are identical.

Direct resize does not letterbox or preserve aspect ratio. The normalized x and
y output components are therefore mapped independently to the original width
and height. This reproduces a full-image affine stretch and does not attempt to
undo crop, pad, or camera-model transformations performed outside the example.

The output is decoded through the generic feature-row adapter and then checked
for the selected plain-model contract: exactly one `[1,2]` prediction, finite
coordinates, and values inside `[0,1]`.

## Tests And Limitations

Run the target-owned native tests with the external model enabled:

```bash
PTAFDEPLOY_PLAIN_CENTROIDING_ONNX=/path/to/plain_model.onnx \
  ctest --test-dir build-plain-centroiding --output-on-failure
```

The permanent tests cover this example's grayscale preprocessing, normalized
coordinate mapping, invalid model contracts, CLI help, wrapper-safe generic
feature serialization, and the optional real ONNX ellipse inference. Existing
Python and MATLAB wrapper smokes exercise the generated generic feature APIs.
They do not test TCLAP, CMake, ONNX Runtime, or generic template functionality.

The synthetic ellipse is a deterministic integration regression, not a
replacement for validation on mission imagery. The test remains opt-in because
the qualified ONNX artifact is external. If the environment variable is not
set, only the real-model case is skipped.
