# PTAF Model Manifest Schema

`.ptafmodel` files are versioned model manifests. They do not contain model
weights. They bind a model artifact to a role, preprocessing/postprocessing
contract labels, and backend-neutral runtime preferences.

## Syntax

- UTF-8 text.
- One `key = value` assignment per line.
- `#` starts a comment.
- Blank lines are ignored.
- Unknown keys are rejected.
- Relative `artifact_path` values are resolved from the manifest file location.

## Schema Version 1

Required keys:

- `schema_version`: must be `1`.
- `artifact_path`: relative or absolute path to `.onnx`, `.engine`, or `.plan`.

Optional keys:

- `role`: one of `raw_tensor`, `centroiding`, `object_detection`, `feature_matching`, `tracking`, `optical_flow`, `custom`. Default: `raw_tensor`.
- `preprocessing`: contract label for caller/model adapter expectations. Default: `caller_supplied_tensors`.
- `postprocessing`: contract label for output adapter expectations. Default: `raw_model_outputs`.
- `backend`: one of `auto`, `onnxruntime`, `tensorrt_engine`. Default: `auto`.
- `artifact`: one of `auto`, `onnx`, `tensorrt_engine`. Default: `auto`.
- `execution_target_priority`: comma-separated backend-neutral targets: `cpu`, `cuda`, `tensorrt`. Default: backend-defined.
- `allow_fallback`: boolean. Default: `true`.
- `device_id`: non-negative integer device index. Default: `0`.
- `intra_op_num_threads`: backend thread setting. `0` means backend default. Negative values are invalid. Default: `1`.
- `inter_op_num_threads`: backend thread setting. `0` means backend default. Negative values are invalid. Default: `1`.
- `enable_profiling`: boolean. Default: `false`.
- `log_id`: backend log/profiling label. Default: `autoforge_deploy`.
- `tensorrt_optimization_profile_index`: non-negative TensorRT runtime profile index for serialized engines. Default: `0`.

Boolean values accepted: `true`, `false`, `on`, `off`, `yes`, `no`, `1`, `0`.

## Runtime Semantics

- `execution_target_priority` is not an ONNX Runtime provider list.
- ONNX Runtime backend maps targets internally to ORT execution providers.
- TensorRT engine backend validates compatible targets and runs serialized
  `.engine` / `.plan` artifacts when built with `ENABLE_TENSORRT=ON`.
- The TensorRT profile key affects only the standalone serialized-engine
  backend. DLA selection is not exposed. Precision is an engine-build property;
  build FP16/INT8 engines with TensorRT tooling before loading them here.
- `preprocessing` and `postprocessing` are manifest contract labels, not hidden
  executable pipelines. Concrete conversion helpers stay in shared adapters.

## Template

```ini
# ptafdeploy model config v1
schema_version = 1
artifact_path = relative/or/absolute/model.onnx
role = raw_tensor
preprocessing = caller_supplied_tensors
postprocessing = raw_model_outputs
backend = auto
artifact = auto
execution_target_priority = cuda,cpu
allow_fallback = true
device_id = 0
intra_op_num_threads = 1
inter_op_num_threads = 1
enable_profiling = false
log_id = autoforge_deploy
tensorrt_optimization_profile_index = 0
```
