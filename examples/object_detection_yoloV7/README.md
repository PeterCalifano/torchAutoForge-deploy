# YOLOv7 Facade Demos

The native, Python, and MATLAB demos run the same object-detection contract:

- load `examples/model_configs/yolov7_640x640.ptafmodel` through
  `ptafdeploy::inference::CModelFacade`;
- resize one RGB image to the model's concrete `[1,3,H,W]` input;
- route HWC-to-NCHW conversion and `1/255` scaling through the shared adapter;
- run wrapper-safe float inference; and
- define the `[1,N,5+C]` YOLO schema in each integration and decode it through
  the generic detection-row adapter.

The first demos intentionally report score-sorted pre-NMS detections. They do
not render an annotated image. NMS, alternate layouts, and visualization remain
deferred until their product contract is selected.

## Local Artifacts

The root `models/` tree and this demo's `sample_data/` directory are gitignored
because model weights may be large or separately licensed. Place a compatible
YOLOv7 ONNX model and input image at:

```text
models/onnx/yolov7_640x640.onnx
examples/object_detection_yoloV7/sample_data/horses.jpg
```

The verified local artifact has this contract:

```text
input:  images float32 [1,3,640,640]
output: output float32 [1,25200,85]
model SHA-256: 99b8030554ed03fa15be185acd58065df6c64c2a79af392a419c87af857aaf63
image SHA-256: c8f0a677a1356569e2ce71d2fa88c1030c0ae57ecf5e14170e02d9a86a20dcb4
```

Use a different artifact path by copying/editing the manifest; do not hardcode
machine-local paths in the library. The sample model's upstream provenance and
redistribution license are not established by this repository, so keep it
external until those are recorded.

## Native C++ CLI

The standalone example requires an installed `autoforge_deploy` package and
OpenCV image components. Its command-line parser uses the repository-bundled
TCLAP headers, so no separate TCLAP package or download is required:

```bash
cmake -S examples/object_detection_yoloV7 -B build-yolov7 \
  -DCMAKE_PREFIX_PATH="/path/to/autoforge_deploy;/path/to/onnxruntime"
cmake --build build-yolov7 --parallel

./build-yolov7/object_detection_yoloV7 \
  examples/model_configs/yolov7_640x640.ptafmodel \
  examples/object_detection_yoloV7/sample_data/horses.jpg \
  --target cpu --score-threshold 0.25 --max-detections 5
```

`--target manifest` uses manifest runtime settings. `--target cpu` and
`--target cuda` force a single target without fallback. If the model and image
are present at configure time, standalone CTest includes one real CPU inference
smoke in addition to the CLI-help smoke:

```bash
ctest --test-dir build-yolov7 --output-on-failure
```

## Python Demo

Build/install the gtwrap Python package and ensure Pillow plus NumPy are
available, then run:

```bash
python3 examples/object_detection_yoloV7/run_object_detection.py \
  examples/model_configs/yolov7_640x640.ptafmodel \
  examples/object_detection_yoloV7/sample_data/horses.jpg \
  --target cpu --score-threshold 0.25 --max-detections 5
```

The script imports `autoforge_deploy`; use the installed package or point
`PYTHONPATH` at the wrapper build directory.

## MATLAB Demo

Add the generated package and MEX directories to the MATLAB path, then run:

```matlab
addpath("build-matlab/wrap/autoforge_deploy")
addpath("build-matlab/wrap/autoforge_deploy_mex")
addpath("examples/object_detection_yoloV7")

stResult = RunYoloFacadeDemo( ...
    "examples/model_configs/yolov7_640x640.ptafmodel", ...
    "examples/object_detection_yoloV7/sample_data/horses.jpg", ...
    strTarget="cpu", ui32MaxDetections=uint32(5));
```

The MATLAB demo requires `imread`/`imresize` support and returns an N-by-6
matrix with rows `[cx cy width height score class_id]`.

## Expected Summary

With the verified local model and image, all three demos report:

```text
role=object_detection
input_shape=[1,3,640,640]
output_shape=[1,25200,85]
detections=5
detection[0]=class=17,...
```

Small numeric differences across CPU, CUDA, OpenCV, Pillow, and MATLAB resize
implementations are expected; the contract, output layout, detection class, and
score ordering should remain coherent.
