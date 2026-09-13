# Model manifest creation tools

The tools use the [shared manifest contract](ptafmodel_manifest_schema.md).

## Create and validate

```bash
make-ptafmodel init models/model.onnx --task centroiding --output model.ptafmodel
make-ptafmodel validate model.ptafmodel
make-ptafmodel validate model.ptafmodel --check-artifact
make-ptafmodel schema
```

Creation uses a relative artifact path where representable, explicit CPU
execution, and fallback disabled. It refuses an existing destination and does
not require model weights to be available. The destination parent must exist.
Use `make-ptafmodel init --help` for device, thread, backend, target, fallback, and
preprocessing/postprocessing options. For a serialized TensorRT engine, select
`--targets cuda`; the CPU creation default applies unless overridden.

Validation reports JSON byte offsets or invalid property/schema paths. Ordinary
validation checks the configuration without accessing an inference session.
`--check-artifact` additionally requires a regular file with `.onnx`, `.engine`,
or `.plan` extension; it does not establish that the file contains a valid model.
Model loading separately checks providers, tensor metadata, and runtime support.

## Interactive creation

```bash
python3 scripts/make_ptafmodel.py --ptafmodel-executable build/src/programs/make-ptafmodel
# After installation, from any working directory:
make_ptafmodel.py
```

The Python 3.12 wizard uses only the standard library. It obtains choices from
the native tool's embedded schema, prompts for settings, shows the selected
arguments, and requests confirmation. It delegates serialization, defaults,
validation, and overwrite protection to the native executable. Cancellation or
EOF before confirmation writes nothing. No Bash wizard is maintained.

Both the executable and wizard are installed under `bin`. The schema is also
installed under `share/autoforge_deploy/schemas`; the executable embeds the same
schema and does not depend on that file or network access at runtime. Optional
image and inference-output components are not required.
