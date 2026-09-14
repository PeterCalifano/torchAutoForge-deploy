# Model manifest examples

`template.ptafmodel` is a minimal raw-tensor configuration.
`centroiding_template.ptafmodel` declares the image-only centroiding adapter.
Both reference a placeholder `model.onnx`; set the path before inference.

The plain centroiding example references checkpoint `traveling-goat-68`, run
`22b61bbd4ddd4d6cb6f503a22f050a25`. Its exported ONNX SHA-256 is
`8ba4f46355b0f6b542ec848b4c1130760bea194f9bf3b16e7b36bad9f380af4b`.

The FiLM example references the ONNX export from ml-based-centroiding and requires
an image tensor plus its separate prior input. The image-only centroiding demo
cannot execute that contract.

Plain, FiLM, and YOLO model binaries may be absent from fresh checkouts. Keep
external binaries in their canonical ignored model locations or update the
manifest artifact paths. Paths are relative to the manifest directory.

See the [manifest contract](../../doc/development/ptafmodel_manifest_schema.md)
for schema requirements, defaults, and native validation.

The tracked test manifest `tests/inference/model_configs/cpu_tree_centroid.ptafmodel`
uses a deterministic CPU-only ONNX fixture (IR 9; opsets 17 and `ai.onnx.ml` 3).
It flattens float32 input `[1,1,2,2]` and applies a single-leaf
`TreeEnsembleRegressor` that returns `[0.25,0.75]` for every input.
