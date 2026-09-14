# Native inference evaluation

The optional native programs prepare image selections and summarize version-one
inference reports. They do not change model preprocessing or execute a Python
interpreter. `utils/metrics` belongs to the core library; report parsing belongs
to `inference_output`, whose RapidJSON headers remain private.
`ReadJsonFile` returns an owned JSON value tree; `ReadInferenceReport` validates
the shared report envelope and returns `SInferenceReport`. Image decoding belongs to `images`. Generic index sampling lives in
`utils/sampling.h` as `SelectEvenlySpacedIndices`; `CopyOrLinkFiles` and its
`EFileCreationMode` belong to `utils/filesystem.h`. Both general utilities are
available through the core library without OpenCV.

Enable `autoforge_deploy_ENABLE_INFERENCE_OUTPUT` to build `evaluate_inference`.
Also enable `autoforge_deploy_ENABLE_IMAGE_SUPPORT` to build
`prepare_image_sequence`. Both executables are installed under `bin`.

## Reproducible selections and execution

```bash
prepare_image_sequence /path/to/images data/inputs/run --count 100 --mode symlink
scripts/run_image_evaluation.sh /path/to/run_centroiding /path/to/evaluate_inference \
  model.ptafmodel data/inputs/run/images data/results/run
```

Create `data/inputs` and `data/results` first. Selection accepts one image or a
non-recursive image directory and uses the demo's natural filename ordering.
For N inputs and K selected frames, indices are floor(i*(N-1)/(K-1)); K is
limited to N, and K=1 selects index zero. The selection manifest records absolute
source paths, indices, image dimensions, decoded sample depth, and channel count.
`--mode copy` creates independent files. Existing nonempty destinations and
repeated basenames are rejected. Sources remain unchanged. Partial selections
are retained on failure; concurrent writers are unsupported.

The shell driver requires a new results directory and runs three sequential CPU
passes with one intra-op and one inter-op thread, fallback disabled, and no
overlays. Each pass loads the model once. Summary statistics exclude the first
five frames and record the first-frame duration separately. This exclusion does
not establish that a model has reached a steady state. Avoid concurrent benchmark
processes and record other host load when interpreting results.

A final annotation run reuses the same inputs and writes directly into the same
results directory. A successful run retains only:

- `predictions.json`: frame predictions and run/model metadata from the annotation run
- `overlays/`: one annotated image per selected frame
- `summary.json`: statistics and run metadata for the three timing passes

Intermediate timing reports are temporary and removed only after success. Summary
`report` fields identify their original execution paths, not retained files.
Console predictions are suppressed; errors remain on stderr. Failures retain
intermediate reports at the printed temporary path and any incomplete annotation
report and recovery files in the results directory. No separate per-pass logs,
host dumps, or checksum files are produced by default. For annotations without
repeated timing, invoke `run_centroiding` directly with `--output PATH --overlays`;
this produces only the report and images. See the
[centroiding demo](../examples/centroiding_models/README.md) for coordinate and
failure conventions.

## Report summaries and reference points

```bash
evaluate_inference --discard 5 --output summary.json pass-1/predictions.json
# Only after verifying that predictions and references share units and convention:
evaluate_inference --discard 5 --output errors.json --references references.json \
  --point-field centroid.image_pixels --units pixels --convention upper_left_xy \
  pass-1/predictions.json
```

Reference format:

```json
{"schema_version":1,"units":"pixels","coordinate_convention":"upper_left_xy",
 "points":[{"source":"frame1.png","x":160.0,"y":120.0}]}
```

The point selector traverses nested objects and requires numeric `x` and `y`.
Units and convention must match the reference metadata exactly. These strings
are caller declarations, not automatic verification of dataset semantics.
Matching uses exact source strings and rejects duplicate or empty identities.
Missing predictions and references are listed separately. Signed biases are
prediction minus reference; Euclidean errors and root-mean-square distance use
all matched frames, including frames excluded from latency statistics.
No matched points produce null metrics.

Each summary includes count, mean, median, 95th percentile, minimum, and maximum.
Quantiles linearly interpolate the sorted samples at p*(n-1). Inputs must be
finite; results outside finite double precision raise an error. The library
borrows input spans and does not modify their ordering. Point comparisons use
standard source-key/point pairs with the existing wrapper-safe `SPoint2D` type;
coordinates use float precision, and accumulated metrics use double precision.
No additional point representation is defined. Report reading retains
JSON records in memory, but does not load images.

Output files must not exist, including dangling symlinks. Concurrent writers are
unsupported. Invalid inputs return exit code 1. Incomplete reports produce a
summary marked incomplete and exit code 2; they are ineligible for comparison.
Fewer than six frames with the default discard count produce null latency
statistics. Complete reports return zero even when no latency samples remain;
check `eligible_for_comparison` before comparing runs. This flag checks report
completion and sample availability only, not environmental comparability.

The FiLM regression test is separate from the image-only demo. Set
`PTAFDEPLOY_FILM_CENTROIDING_ONNX` to its compatible image-plus-prior ONNX artifact.
An unset path skips external-model tests; an explicitly invalid path fails.
CUDA is skipped only when the runtime reports that target unavailable. Synthetic
images test the contract and finite outputs, not learned centroid accuracy.

## Recorded sequence validation, 2026-09-12

Release CPU execution used `best_model_plain_traveling-goat-68_22b61bbd4ddd.onnx`
(SHA-256 `8ba4f46355b0f6b542ec848b4c1130760bea194f9bf3b16e7b36bad9f380af4b`).
Each group used 100 evenly spaced images, three sequential JSON-only passes,
and ten separately selected overlays. All 900 timed frames and 30 overlay frames
completed. Coordinate scaling, inside-image flags, original extents, and unchanged
pixels outside marker strokes were checked for all overlays. No accuracy was
computed. Other MATLAB workloads were active; these timings describe that host
load and are not isolated performance comparisons.

| Group | Median ms, passes 1/2/3 | p95 ms, passes 1/2/3 | First frame ms, passes 1/2/3 |
|---|---|---|---|
| cosmica | 32.52/32.38/32.36 | 47.39/66.90/58.14 | 55.43/46.25/47.25 |
| itokawa | 31.99/33.04/33.04 | 48.75/51.94/40.83 | 45.43/45.13/43.54 |
| operative | 42.05/42.01/41.93 | 51.73/50.22/59.65 | 52.96/46.88/44.52 |

The following source directories were linked into ignored `data/inputs` selections.
Selection manifests record exact frame identities and decoded dimensions/depth;
the original input checksums and host diagnostics are retained in the temporary
archive described below.

- `cosmica`: `/home/peterc/devDir/projects-DART/cosmica-simulator/output_images/images` (7905 available images)

- `itokawa`: `/media/peterc/DatasetsArchive/datasets/UniformlyScatteredPointCloudsDatasets/Itokawa/Dataset_UniformPointCloud_Itokawa_SPECTRAL_OPTIX_RT_1000_WFOV_Farinella_evaluation_ID99/images` (1000 available images)

- `operative`: `/media/peterc/DatasetsArchive/datasets/TrajectoriesDatasets/Moon/OPERATIVE_trajectory_test/images` (27106 available images)

COSMICA's target identity follows the user's identification of this output folder.
No co-located run manifest established its exact simulation configuration; the
separate `output_images_ID0` directory was not substituted. The supplementary
Itokawa YAML labels provide both `dCentreOfFigure` and
`dObjProjectedEllipsoidCentre`. OPERATIVE labels provide `bodyCenter` and camera
geometry. Their origin conventions and correspondence to the learned target have
not been verified. These fields were not used as accuracy references.

Fresh build, test, installed-consumer, and known-answer CLI evidence resides at
`/tmp/ptaf-evaluation-9mtake_3`. Prior Python/MATLAB six-frame schema, mapping,
overlay, and partial-failure evidence remains at
`/tmp/ptaf-utils-validation-qnfz4s88`; those unchanged paths were not requalified
as performance benchmarks. Temporary evidence and ignored data are local artifacts,
not distributed test fixtures. A subsequent search of `$WS_ML_CEN/onnx_checkpoints`
located the compatible FiLM artifact; its CPU and CUDA contract tests both passed
(see the development tracker). The additional user folder remains pending.

The layout was subsequently consolidated to one 100-image input selection and
one output directory per group. `data/results/<group>` now contains only
`predictions.json`, `summary.json`, and `overlays/` with 100 annotated images.
Prior timing reports, diagnostic files, and ten-frame runs were moved to
`/tmp/ptaf-previous-results-mr73w5o0`. Their execution metadata was preserved;
the three timing summaries are unchanged. Minimal-output script checks passed
for a real-model run, collision protection, temporary-file cleanup, and failed
image decoding with an incomplete report retained. Evidence resides at
`/tmp/ptaf-minimal-output-test-mfpt_lui`.
