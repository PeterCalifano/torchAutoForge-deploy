# Local Model Artifacts

`models/` is the single repository-local root for generated and externally
supplied model artifacts. Its contents are intentionally ignored except for this
layout contract:

- `matlab/` contains MATLAB model artifacts
- `onnx/` contains ONNX graphs consumed by the deployment library
- `pytorch/` contains source checkpoints used by model-owning export workflows
- `tensorrt/` contains hardware- and toolchain-specific engines and sidecars

Do not commit model binaries, conversion reports, benchmark output, or generated
sidecars without an explicit fixture review. Example `.ptafmodel` manifests use
paths relative to their own location and may refer into this tree.
