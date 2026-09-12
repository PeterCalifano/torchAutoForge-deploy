# Centroiding models

These demos accept an ordinary image file or a directory of images and run
independent, batch-one inference through `CModelFacade`. The model receives one
float32 image tensor `[1,1,H,W]` and produces normalized `[x,y]` in `[1,2]`.
Models with a prior-vector input, including FiLM models, require a separate adapter.
The word `plain` remains in architecture and checkpoint identifiers.

## Native application

Install torchAutoForge-deploy first. Configure this standalone application with
OpenCV (core, imgproc, imgcodecs) and the installed inference package available:

```bash
cmake -S examples/centroiding_models -B build-centroiding \
  -DCMAKE_PREFIX_PATH=/path/to/autoforge_deploy/install
cmake --build build-centroiding --parallel

build-centroiding/run_centroiding model.ptafmodel image.png
build-centroiding/run_centroiding model.ptafmodel frames/ --output results/
build-centroiding/run_centroiding model.ptafmodel frames/ --output results/ --overlays
```

With the external plain checkpoint present at the manifest's configured path,
the tracked fixture can be used directly:

```bash
build-centroiding/run_centroiding \
  examples/model_configs/ml_based_centroiding_plain.ptafmodel \
  examples/centroiding_models/test_data/bright_ellipse_320x240.png
```

Expected console fields include `role=centroiding`, `image_size=[320,240]`,
`output_shape=[1,2]`, `inference_ms`, and the decoded coordinate fields. Coordinate
values depend on the model; the fixture does not define an accuracy expectation.

Raw `.onnx` artifacts are also accepted. Runtime overrides remain `--targets`,
`--device`, `--intra-op-threads`, `--inter-op-threads`, and `--no-fallback`.
The native executable writes JSON directly using the embedded RapidJSON headers;
Python and MATLAB are not needed. OpenCV, RapidJSON, and TCLAP remain private
application dependencies.

## Python and MATLAB

Use an environment containing the generated `autoforge_deploy` Python wrapper,
NumPy, and Pillow:

```bash
python examples/centroiding_models/run_centroiding.py model.ptafmodel frames/ \
  --output results-python/ --overlays
```

Python retains `--target manifest|cpu|cuda` and `--device N`. With the generated
MATLAB wrapper and Image Processing Toolbox on the MATLAB path:

```matlab
addpath('examples/centroiding_models');
stRun = RunCentroidingFacadeDemo("model.ptafmodel", "frames", ...
    strOutputPath="results-matlab", bOverlays=true, strTarget="cpu");
```

MATLAB output publication requires its JVM for canonical paths and atomic file
replacement. MATLAB returns the report-shaped run struct for either input mode. Its `frames`
cell array retains compact records, not images. Both wrapper demos produce their
own matching JSON reports using native language facilities.

## Selection, image processing, and coordinates

Directory selection is non-recursive and accepts regular `.png`, `.jpg`, `.jpeg`,
`.bmp`, `.tif`, and `.tiff` files, case-insensitively. Explicit single-file inputs
retain the corresponding decoder's supported formats. Empty selections fail.
The selected filenames are fixed before output creation.

Filenames use natural ordering: ASCII digit runs compare by numeric value without
integer conversion, other characters compare case-sensitively by Unicode code
point, and equal natural keys compare by the complete original filename.
For example, `frame02.png`, `frame2.png`, `frame10.png` occur in that order.
Filenames must be valid Unicode; numeric sorting does not change frame contents.

The model is loaded once. Each image is decoded, converted to unsigned 8-bit
grayscale, resized bilinearly to the model extent, and scaled by `1/255` using the
existing generic adapters. OpenCV, Pillow, and MATLAB retain their established
grayscale/resize implementations; numerical equality across languages is not
assumed. The report identifies the preprocessing implementation.

Coordinates have their origin at the upper-left image boundary, with x rightward
and y downward. Normalized coordinates multiply width and height, not width-minus-one
or height-minus-one. Thus `(0.5,0.5)` maps to `(160,120)` for a 320-by-240 image.
A prediction lies inside the image when `0 <= x < width` and `0 <= y < height`.
Normalized 1 is outside the corresponding upper boundary. Finite predictions
outside the image are retained; invalid shapes and non-finite outputs fail.

Overlays are disabled by default. `--overlays` requires `--output`; overlays are
original-resolution PNG files named `overlays/000000_<source-stem>.png`.
Contrasting crosshair strokes are drawn at the predicted location and clipped to
the image. A fully off-image crosshair is invisible. Image pixels outside the
strokes retain their appearance, and recorded coordinates are never clamped.
PNG overlays support unsigned 8-bit and 16-bit samples; unsupported sample types
fail explicitly instead of silently reducing their range.

## Reports and failures

Without `--output`, predictions appear through the existing console fields, with
frame index, source filename, and `inference_ms` added. With output enabled,
`predictions.json` contains:

- `schema_version: 1`, `status: complete|incomplete`, and the requested model path;
- resolved model/config paths, role, effective runtime, backend description,
  declarative pipeline names, and input/output tensor metadata;
- actual preprocessing, input path/kind, selection count, and ordering;
- an ordered `frames` array containing index, relative source filename, original
  dimensions, raw output name/shape/values, normalized/model/image coordinates,
  `inside_image`, inference duration, and relative overlay path or null.

Runtime backend, artifact, and target enum values use the public enum integer
values. The existing Python/MATLAB wrapper does not expose the runtime priority
vector; `execution_target_priority` is null there. Native reports include the
vector. The effective backend description and other exposed runtime settings are
recorded in all three demos. No public facade change is introduced for reporting.
Native JPEG overlays follow the inference decoder's EXIF orientation handling;
Python and MATLAB retain their existing image-decoding conventions.

`inference_ms` measures the facade inference call, including facade/wrapper overhead.
Model loading, preprocessing, overlay generation, and report IO are excluded.
The first frame may include runtime initialization effects; it is not discarded.

Output must be a new directory or an existing empty directory. Nonempty output
locations fail without overwriting prior results. Output cannot equal or contain
the input location. A child output directory is permitted because selection is
fixed and non-recursive. Use a separate empty destination for each language/run.

An initial valid incomplete report is written before loading the model. On a
caught failure, processing stops and publication retains completed frame records
with error stage, frame index/source, and message. Reports are replaced atomically
on the same filesystem; unsupported atomic replacement fails explicitly.
Native and Python demos spool records to disk. MATLAB retains compact return
records and also spools completed frames. On publication failure, the prior
incomplete report and diagnostic files remain in the output directory.
Abrupt termination and concurrent writers to the same output directory are outside
the supported recovery contract.

## Validation phase (pending authorization)

Implementation has not yet been runtime-qualified. Do not infer numerical accuracy
from successful plumbing tests or an arbitrary synthetic blob's geometric centre.

After authorization, run the native CTest suite, Python sequence tests, MATLAB
sequence checks, and the available real ONNX model. Native tests can use
`PTAFDEPLOY_PLAIN_CENTROIDING_ONNX` for the external plain checkpoint.

```bash
ctest --test-dir build-centroiding --output-on-failure
python -m pytest examples/centroiding_models/test/test_centroiding_sequence.py
```

In MATLAB, add the demo directory to the path and run
`runtests("examples/centroiding_models/test/TestCentroidingSequence.m")`.

The primary image targets are the COSMICA Itokawa simulator outputs. Select and
record the intended run under `cosmica-simulator/output_images/images` or
`output_images_ID0`. Supplementary cases come from ml-based-centroiding evaluation
sets and OPERATIVE directories under `$DATASETS`. The user will provide another
folder later. Keep datasets unchanged and generated reports outside source control.
Record model/runtime/image provenance and first-frame versus aggregate timing.
Report coordinate errors only when the label definition and coordinate convention
have been verified.
