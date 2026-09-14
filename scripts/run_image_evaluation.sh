#!/usr/bin/env bash
# Keep one prediction report, annotations, and a timing summary per image sequence.
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "Usage: $0 RUN_CENTROIDING EVALUATE_INFERENCE MODEL INPUT_DIRECTORY NEW_RESULTS_DIRECTORY" >&2
    exit 1
fi

centroiding_executable=$1
evaluation_executable=$2
model_path=$3
input_directory=$4
output_directory=$5

[[ -d "$input_directory" && -f "$model_path" ]] || { echo 'Missing model or input directory' >&2; exit 1; }
mkdir -- "$output_directory"
temporary_directory=$(mktemp -d "${TMPDIR:-/tmp}/ptaf-image-evaluation.XXXXXXXX")
run_completed=false

# Failed runs retain their original reports and spool files for diagnosis.
cleanup_temporary_reports() {
    if [[ "$run_completed" == true ]]; then
        rm -rf -- "$temporary_directory"
    else
        printf 'Evaluation failed; partial output: %s; intermediate reports: %s\n' "$output_directory" "$temporary_directory" >&2
    fi
}
trap cleanup_temporary_reports EXIT

# Collect three independent timing passes before generating annotated images.
timing_reports=()
for pass_index in 1 2 3; do
    pass_directory="$temporary_directory/pass-$pass_index"
    "$centroiding_executable" "$model_path" "$input_directory" --targets cpu --intra-op-threads 1 \
        --inter-op-threads 1 --no-fallback --output "$pass_directory" > /dev/null
    timing_reports+=("$pass_directory/predictions.json")
done

# Annotation uses the same inputs after timing passes; no extra input selection is needed.
"$centroiding_executable" "$model_path" "$input_directory" --targets cpu --intra-op-threads 1 \
    --inter-op-threads 1 --no-fallback --output "$output_directory" --overlays > /dev/null

# Publish the summary before removing intermediate reports.
"$evaluation_executable" --discard 5 --output "$output_directory/summary.json" "${timing_reports[@]}"
run_completed=true
