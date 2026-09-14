# PTAF model manifests

A `.ptafmodel` file is a UTF-8 JSON configuration that references an ONNX or
TensorRT artifact. It contains no model weights. The authoritative schema is
[`schemas/ptafmodel.schema.json`](../../schemas/ptafmodel.schema.json).

## Contract

Schema version remains `1`. Required properties are `schema_version`,
`artifact_path`, and `task`. Legacy key-value syntax and the `role` property are
not accepted. Comments, duplicate properties, unknown properties, and invalid
values are rejected. Strings containing `#` are ordinary JSON strings.

`artifact_path` is resolved relative to the manifest directory. `task` selects
one of the existing model tasks. Public C++ facade names and enum types retain
their existing names; serialized reports and command-line task selection use
`task` and `--task`.

Optional settings retain their runtime defaults: automatic backend/artifact
selection, backend-defined execution-target priority, fallback enabled, device
zero, one intra-operation and inter-operation thread, profiling disabled,
`log_id` equal to `autoforge_deploy`, and TensorRT optimization profile zero.
Preprocessing and postprocessing default to `caller_supplied_tensors` and
`raw_model_outputs`. These labels describe an adapter contract; validation does
not execute or verify the corresponding algorithms.

Booleans must be JSON booleans. Execution-target priorities are nonempty arrays
of distinct `cpu`, `cuda`, or `tensorrt` strings. Numeric runtime settings are
non-negative integers representable by the native runtime API. Backend/artifact
combinations must agree with each other and with recognized artifact extensions.

## Example

```json
{
  "schema_version": 1,
  "artifact_path": "models/model.onnx",
  "task": "centroiding",
  "execution_target_priority": ["cpu"],
  "allow_fallback": false
}
```

## Native validation

`ValidatePtafModelConfig(path)` validates a file without loading a model.
`ValidatePtafModelJson(json, path)` validates in-memory text using `path` for
relative artifact resolution and diagnostics. Passing `true` as the second
argument to `ValidatePtafModelConfig` additionally requires an existing regular
artifact file with a supported extension. It does not verify the model contents.
Model loading separately checks providers, tensor metadata, and runtime support.

Validation reports JSON byte offsets or invalid property/schema paths. Duplicate
properties and NUL bytes are rejected to prevent ambiguous parsing or truncated
paths. The compiled schema is shared across calls; each call owns its document
and validator. Validation runs during configuration loading, not inference.

The schema is embedded in the core library and installed under
`share/autoforge_deploy/schemas`. Validation requires no network access, OpenCV,
or optional inference-output component. `GetPtafModelSchemaJson()` returns the
embedded schema for editors and creation tools.

See [manifest creation tools](ptafmodel_tools.md) for command-line and interactive
creation, including overwrite protection and defaults.

Automatic generation during model export will be implemented in PTAF. Exporters
must supply task and preprocessing information explicitly when it is not known;
tensor shapes alone do not establish those semantics.
