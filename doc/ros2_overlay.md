# ROS 2 Inference Overlay

The ROS 2 Jazzy overlay is optional and does not affect the ordinary
`build_lib.sh` workflow.

## Packages

`ptafdeploy` builds and installs the core CMake project inside colcon.
`ptafdeploy_interfaces` defines wrapper-safe float tensors, model status, and
the inference service. `ptafdeploy_ros` owns the lifecycle bridge.
`ptafdeploy_spinup` installs launch descriptions and the small ONNX fixture.

The bridge intentionally calls
`ptafdeploy::inference::CModelFacade::InferFloatTensors`. It does not expose ORT
sessions, TensorRT handles, raw pointers, or byte buffers.

## Lifecycle Contract

The component class is `ptafdeploy_ros::CInferenceLifecycleNode`, with default
node name `ptafdeploy_inference`.

- `model_config_path` is required when configuring.
- Configure loads and validates the `.ptafmodel`.
- Activate enables `~/infer` and the lifecycle `~/status` publisher.
- Inactive requests return `success=false`.
- Conversion or inference exceptions become failure responses and status data.
- Calls are serialized around the single facade instance.
- `inference_count` increments only after successful inference.
- Cleanup releases the facade, service, publisher, and status state.

## Build And Launch

```bash
export ONNXRUNTIME_ROOT=/path/to/onnxruntime
./build_ros2.sh --clean
source ros2/install/setup.bash

ros2 launch ptafdeploy_spinup ptafdeploy.launch.py \
  namespace:=demo node_name:=inference

ros2 launch ptafdeploy_spinup ptafdeploy_composition.launch.py \
  namespace:=demo node_name:=inference
```

`build_ros2.sh` converts a valid `ONNXRUNTIME_ROOT` into the explicit
`onnxruntime_DIR` needed by downstream colcon packages. An explicit
`--cmake-arg -Donnxruntime_DIR=/path/to/lib/cmake/onnxruntime` takes
precedence.

Set `autostart:=false` to leave the node unconfigured. Override
`model_config_path` to load another installed or absolute manifest.

`./build_ros2.sh --cuda --cmake-arg
-DCMAKE_CUDA_ARCHITECTURES=<numeric>` forwards CUDA/PTX support only. There is
no OptiX compatibility flag.

## Tests

The ROS tests are target-specific:

- float-tensor conversion and validation;
- lifecycle loading, inactive rejection, real fixture inference, failure
  translation, status accounting, and cleanup;
- optional real YOLO image preprocessing, service transport, raw-output shape,
  and integration-owned detection decoding; and
- standalone/composed launch at root and under a namespace, using real ONNX
  inference and expected numeric output.

The YOLO ROS test uses the same external model and image documented by the
native/Python/MATLAB demo. When OpenCV is available, the test target is always
built; it reports a GTest skip when either external artifact is absent. With
both artifacts present, run it directly after building the overlay:

```bash
source ros2/install/setup.bash
ctest --test-dir ros2/build/ptafdeploy_ros \
  -R test_yolo_inference_lifecycle_node -V
```

The test sends `[1,3,640,640]` float data through the generic `~/infer`
service, expects raw `[1,25200,85]` output, and defines the YOLO row schema only
inside the test-side integration. The lifecycle node remains model-agnostic.
